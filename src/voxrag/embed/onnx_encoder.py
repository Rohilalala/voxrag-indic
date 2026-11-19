"""ONNX dense encoder for `intfloat/multilingual-e5-small`.

This is the only neural model on the sub-200ms request path, so the design rule here is:
pay every cost once, at load time, and pay nothing avoidable per call.

  * the graph, the session, the tokenizer, the input plumbing and the embedding dim are
    all resolved in `__init__`;
  * the resolved file paths are memoised to disk so a process restart makes zero network
    calls;
  * padding is dynamic (longest-in-batch), so a 12-token query does not pay for 192;
  * a repeated query never reaches onnxruntime at all (LRU cache);
  * `encode_async` hands the blocking Run call to a worker thread, and skips the thread
    hop entirely when every input is already cached.

Implements `voxrag.contracts.Embedder`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import OrderedDict
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Final

import numpy as np
import onnxruntime as ort
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError
from tokenizers import Tokenizer

from voxrag.config import DATA_DIR, settings

log = logging.getLogger("voxrag.embed")

# --------------------------------------------------------------------------------------
# Module-level constants
#
# These are NOT settings: they describe the shape of the upstream HF repos, not anything
# an operator should tune. Everything tunable already lives in config.Settings.
# --------------------------------------------------------------------------------------

#: ONNX graph resolution order, after `settings.embed_onnx_file`. Checked against the real
#: repo on 2026-08-15: intfloat/multilingual-e5-small publishes `onnx/model.onnx`,
#: `onnx/model_O4.onnx` and `onnx/model_qint8_avx512_vnni.onnx` -- it does NOT publish the
#: `onnx/model_quantized.onnx` name config.py guesses at, which is exactly why this
#: fallback chain exists. int8 names come first: 118MB versus 470MB to pull, and int8
#: kernels on the CPU EP; the fp32 graph is a correct but heavier fallback.
#: `model_O4.onnx` is deliberately excluded -- O4 is fp16, and the CPU execution provider
#: casts fp16 back to fp32 per op, so it is strictly slower than the plain fp32 graph.
ONNX_CANDIDATES: Final[tuple[str, ...]] = (
    "onnx/model_quantized.onnx",
    "onnx/model_qint8_avx512_vnni.onnx",
    "onnx/model_int8.onnx",
    "onnx/model.onnx",
    "model_quantized.onnx",
    "model.onnx",
)

#: Tokenizer resolution order. A tokenizer sitting next to the chosen graph is preferred
#: because it is guaranteed to be the one the export was traced with.
TOKENIZER_CANDIDATES: Final[tuple[str, ...]] = ("tokenizer.json", "onnx/tokenizer.json")

#: Pad tokens across the encoder families we might load (XLM-R uses `<pad>`, BERT `[PAD]`).
PAD_TOKENS: Final[tuple[str, ...]] = ("<pad>", "[PAD]", "<PAD>")

#: Seconds allowed for the single Hub metadata call. Without a ceiling a flaky network
#: turns "which file exists?" into a multi-minute hang at import time.
HUB_TIMEOUT_S: Final[float] = 10.0

#: Where the resolved paths are memoised. Restarting the process must not re-hit the Hub.
RESOLUTION_CACHE: Final[Path] = DATA_DIR / "embed_resolved.json"


class EmbedderUnavailableError(RuntimeError):
    """No usable ONNX graph could be resolved for the configured repo.

    Deliberately a RuntimeError: the harness catches it, records a degradation and falls
    back to lexical-only retrieval. It is *not* silently replaced with a fake embedder --
    a hash-based stand-in would return plausible vectors and silently destroy recall,
    which is far worse than an honest degraded path.
    """


# --------------------------------------------------------------------------------------
# Model file resolution
# --------------------------------------------------------------------------------------


def _load_resolution_cache(repo: str) -> tuple[Path, Path, str] | None:
    """Return memoised (onnx, tokenizer, relative-name) if still valid on disk."""
    try:
        blob = json.loads(RESOLUTION_CACHE.read_text())
        if blob.get("repo") != repo:
            return None
        onnx_path = Path(blob["onnx_path"])
        tok_path = Path(blob["tokenizer_path"])
    except (OSError, ValueError, KeyError):
        return None
    if not onnx_path.is_file() or not tok_path.is_file():
        return None  # HF cache was pruned since we wrote this
    return onnx_path, tok_path, str(blob.get("onnx_file", onnx_path.name))


def _store_resolution_cache(repo: str, onnx_path: Path, tok_path: Path, onnx_file: str) -> None:
    payload = {
        "repo": repo,
        "onnx_file": onnx_file,
        "onnx_path": str(onnx_path),
        "tokenizer_path": str(tok_path),
    }
    try:
        RESOLUTION_CACHE.parent.mkdir(parents=True, exist_ok=True)
        RESOLUTION_CACHE.write_text(json.dumps(payload, indent=2))
    except OSError as exc:  # read-only container filesystem: costs a Hub call, not correctness
        log.debug("could not memoise embed resolution: %s", exc)


def _download(repo: str, filename: str, *, local_only: bool = False) -> Path | None:
    """One Hub fetch with bounded retries. Returns None when the file simply is not there.

    Missing-file errors are terminal (no retry); transport errors are retried, because the
    Hub returning a 5xx or dropping a connection mid-download is routine.
    """
    attempts = max(1, settings.max_retries)
    for attempt in range(attempts):
        try:
            return Path(
                hf_hub_download(
                    repo,
                    filename,
                    token=settings.hf_token,
                    local_files_only=local_only,
                    etag_timeout=HUB_TIMEOUT_S,
                )
            )
        except (EntryNotFoundError, RepositoryNotFoundError, GatedRepoError):
            # Terminal: the file/repo is absent or forbidden. Retrying a 404 six times
            # over six candidates is how a typo in embed_model_repo turns into a
            # 30-second startup hang instead of an immediate, readable error.
            return None
        except Exception as exc:  # hub raises a wide, version-dependent set
            if local_only:
                return None  # "not in the local cache" is an expected miss, not a failure
            if attempt == attempts - 1:
                log.warning("hub fetch failed for %s/%s: %s", repo, filename, exc)
                return None
            time.sleep(settings.retry_backoff_ms / 1000.0 * (attempt + 1))
    return None


def _repo_files(repo: str) -> set[str] | None:
    """List the repo in ONE bounded call, so we never probe candidates with N 404s."""
    try:
        info = HfApi(token=settings.hf_token).model_info(repo, timeout=HUB_TIMEOUT_S)
    except Exception as exc:  # offline is a normal state here
        log.debug("could not list %s: %s", repo, exc)
        return None
    return {s.rfilename for s in (info.siblings or [])}


def _export_with_optimum(repo: str) -> Path | None:
    """Last resort: trace the torch weights to ONNX ourselves.

    optimum is not a declared dependency (it drags in torch, ~2GB), so this only runs on a
    machine that happens to have it. Guarded by find_spec rather than a bare import so the
    common path never pays the import cost.
    """
    if find_spec("optimum") is None or find_spec("torch") is None:
        return None
    out_dir = DATA_DIR / "onnx-export" / repo.replace("/", "__")
    exported = out_dir / "model.onnx"
    if exported.is_file():
        return exported
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction

        model = ORTModelForFeatureExtraction.from_pretrained(repo, export=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
    except Exception as exc:  # any export failure falls through to the error
        log.warning("optimum export of %s failed: %s", repo, exc)
        return None
    return exported if exported.is_file() else next(iter(out_dir.glob("*.onnx")), None)


def _resolve_model_files(repo: str, preferred: str) -> tuple[Path, Path, str]:
    """Locate the ONNX graph and its tokenizer, downloading only if we must.

    Order: memoised paths -> local HF cache -> one repo listing + one download -> optimum
    export -> actionable error.
    """
    memoised = _load_resolution_cache(repo)
    if memoised is not None:
        return memoised

    # dict.fromkeys: dedupe while preserving the configured file's priority.
    candidates = list(dict.fromkeys((preferred, *ONNX_CANDIDATES)))

    onnx_path: Path | None = None
    chosen = ""
    for name in candidates:  # warm start: already-downloaded blob, zero network
        onnx_path = _download(repo, name, local_only=True)
        if onnx_path is not None:
            chosen = name
            break

    if onnx_path is None:
        listed = _repo_files(repo)
        wanted = [c for c in candidates if c in listed] if listed is not None else candidates
        for name in wanted:
            onnx_path = _download(repo, name)
            if onnx_path is not None:
                chosen = name
                break

    if onnx_path is None:
        onnx_path = _export_with_optimum(repo)
        chosen = "optimum-export/model.onnx"

    if onnx_path is None:
        raise EmbedderUnavailableError(
            f"No ONNX graph found for embed_model_repo={repo!r}. Tried {candidates!r} in the "
            f"local HF cache and on the Hub, and could not export from torch weights "
            f"(install `optimum[onnxruntime]` to enable that). Fix by setting "
            f"VOXRAG_EMBED_ONNX_FILE to a file that exists in the repo, or point "
            f"VOXRAG_EMBED_MODEL_REPO at a repo that ships ONNX."
        )

    tok_names = list(dict.fromkeys((*_sibling_tokenizers(chosen), *TOKENIZER_CANDIDATES)))
    tok_path: Path | None = None
    for name in tok_names:
        tok_path = _download(repo, name, local_only=True) or _download(repo, name)
        if tok_path is not None:
            break
    if tok_path is None:
        raise EmbedderUnavailableError(
            f"Found an ONNX graph for {repo!r} at {onnx_path} but no tokenizer.json "
            f"(tried {tok_names!r}). A fast tokenizer is required; slow SentencePiece "
            f"tokenisers are far too slow for the latency budget."
        )

    _store_resolution_cache(repo, onnx_path, tok_path, chosen)
    return onnx_path, tok_path, chosen


def _sibling_tokenizers(onnx_file: str) -> tuple[str, ...]:
    parent = str(Path(onnx_file).parent)
    return () if parent in (".", "") else (f"{parent}/tokenizer.json",)


# --------------------------------------------------------------------------------------
# Query cache
# --------------------------------------------------------------------------------------


class _LruCache:
    """Tiny LRU over (text, is_query) -> unit vector.

    Locked because `encode_async` runs the encoder from arbitrary worker threads. The
    critical section is a dict operation, so contention is irrelevant next to a 4ms
    forward pass.
    """

    __slots__ = ("_capacity", "_data", "_lock", "evictions", "hits", "misses")

    def __init__(self, capacity: int) -> None:
        self._data: OrderedDict[tuple[str, bool], np.ndarray] = OrderedDict()
        self._capacity = max(0, capacity)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: tuple[str, bool], *, count: bool = True) -> np.ndarray | None:
        """`count=False` looks up without touching the counters.

        The async fast path probes the cache before deciding whether to spawn a worker
        thread, and that probe would otherwise book a second set of misses for work the
        counted pass is about to book again -- halving the reported hit rate for no reason.
        """
        if self._capacity == 0:
            return None
        with self._lock:
            vec = self._data.get(key)
            if vec is None:
                if count:
                    self.misses += 1
                return None
            self._data.move_to_end(key)
            if count:
                self.hits += 1
            return vec

    def note_hits(self, n: int) -> None:
        """Credit hits found by an uncounted probe that ended up serving the request."""
        with self._lock:
            self.hits += n

    def put(self, key: tuple[str, bool], vec: np.ndarray) -> None:
        if self._capacity == 0:
            return
        # Copy: `vec` is a row view into a whole batch array, and keeping the view alive
        # would pin the entire batch in memory. Read-only so a caller cannot corrupt it.
        stored = np.array(vec, dtype=np.float32, copy=True)
        stored.setflags(write=False)
        with self._lock:
            self._data[key] = stored
            self._data.move_to_end(key)
            while len(self._data) > self._capacity:
                self._data.popitem(last=False)
                self.evictions += 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            size = len(self._data)
            hits, misses, evictions = self.hits, self.misses, self.evictions
        looked_up = hits + misses
        return {
            "size": size,
            "capacity": self._capacity,
            "hits": hits,
            "misses": misses,
            "evictions": evictions,
            "hit_rate": (hits / looked_up) if looked_up else 0.0,
        }


# --------------------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------------------


class OnnxEncoder:
    """CPU ONNX implementation of `contracts.Embedder`."""

    def __init__(
        self,
        *,
        repo: str | None = None,
        onnx_file: str | None = None,
        threads: int | None = None,
    ) -> None:
        self._repo = repo or settings.embed_model_repo
        onnx_path, tok_path, rel = _resolve_model_files(
            self._repo, onnx_file or settings.embed_onnx_file
        )
        self.name = f"onnx:{self._repo}:{Path(rel).name}"
        self._tokenizer = self._build_tokenizer(tok_path)
        self._session = self._build_session(onnx_path, threads or settings.embed_threads)

        # Resolve the graph's IO once. Doing this per call costs a few hundred ns of
        # attribute walking and, worse, tempts you into feeding tensors the graph ignores.
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._wants_token_type = "token_type_ids" in self._input_names
        unknown = set(self._input_names) - {"input_ids", "attention_mask", "token_type_ids"}
        if unknown:
            # Fail at load with the input names in hand, rather than with an opaque ORT
            # "Missing Input" on the first query, mid-request.
            raise EmbedderUnavailableError(
                f"{onnx_path} expects inputs this encoder cannot supply: {sorted(unknown)}. "
                f"Expected an encoder graph taking input_ids/attention_mask."
            )
        self._output_name = self._pick_output()

        self._cache = _LruCache(settings.embed_cache_size)
        self._n_query_encodes = 0
        self._n_passage_encodes = 0
        self._n_forward_batches = 0

        # A probe forward pass, not an optimisation: it is the only way to learn the true
        # hidden size when the export declares a symbolic output dim, and we must know
        # `dim` before allocating a single output buffer.
        self.dim = self._probe_dim()

    # -- construction helpers -----------------------------------------------------------

    @staticmethod
    def _build_tokenizer(tok_path: Path) -> Tokenizer:
        tok = Tokenizer.from_file(str(tok_path))
        tok.enable_truncation(max_length=settings.embed_max_tokens)
        pad_id, pad_token = 0, PAD_TOKENS[0]
        for candidate in PAD_TOKENS:
            found = tok.token_to_id(candidate)
            if found is not None:
                pad_id, pad_token = found, candidate
                break
        # length=None => pad to the longest sequence in *this* batch. Padding to
        # embed_max_tokens (192) instead would make a 12-token query do ~16x the matmul
        # work for identical output, since the pad positions are masked away anyway.
        tok.enable_padding(pad_id=pad_id, pad_token=pad_token, length=None)
        return tok

    @staticmethod
    def _build_session(onnx_path: Path, threads: int) -> ort.InferenceSession:
        opts = ort.SessionOptions()
        # Threads trade tail latency for batch throughput, and the graded number is the
        # tail. Measured here (arm64, 1-query batches, 300 samples per setting):
        #   1 -> p50 1.90ms p100 3.77ms | 2 -> p50 1.44ms p100 2.55ms
        #   4 -> p50 1.30ms p100 4.54ms | 8 -> p50 1.75ms p100 2.68ms
        # 4 buys 0.14ms of median and pays 2ms of P100 to the fork/join barrier, so the
        # default stays 2. Raise settings.embed_threads only for an ingest-only process,
        # where batch64 does improve monotonically (492/336/220/196ms).
        opts.intra_op_num_threads = max(1, threads)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.log_severity_level = 3  # warnings and below are noise on every session build
        try:
            return ort.InferenceSession(
                str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # corrupt/partial download is the usual cause
            raise EmbedderUnavailableError(
                f"onnxruntime could not load {onnx_path}: {exc}. If the download was "
                f"interrupted, delete {RESOLUTION_CACHE} and the file above, then retry."
            ) from exc

    def _pick_output(self) -> str:
        outputs = self._session.get_outputs()
        for out in outputs:
            if out.name in ("last_hidden_state", "token_embeddings"):
                return out.name
        for out in outputs:
            if len(out.shape) == 3:  # (batch, seq, hidden) -- the only poolable output
                return out.name
        return outputs[0].name

    def _probe_dim(self) -> int:
        # Shape of the output we actually pool, which is not necessarily outputs[0].
        static = next(
            (o.shape for o in self._session.get_outputs() if o.name == self._output_name), []
        )
        if static and isinstance(static[-1], int) and static[-1] > 0:
            declared = int(static[-1])
        else:
            declared = int(self._forward(["probe"]).shape[1])
        if declared != settings.embed_dim:
            # Loud, because the vector store is built with settings.embed_dim and would
            # otherwise reject or silently mis-index every upsert.
            log.warning(
                "embed_dim mismatch: %s emits %d dims, settings.embed_dim=%d. Using %d; "
                "set VOXRAG_EMBED_DIM to match before building the index.",
                self.name,
                declared,
                settings.embed_dim,
                declared,
            )
        return declared

    # -- inference ----------------------------------------------------------------------

    def _forward(self, prefixed: list[str]) -> np.ndarray:
        """Tokenise -> run -> mean-pool -> L2-normalise for one already-prefixed batch."""
        encodings = self._tokenizer.encode_batch(prefixed)
        # Built straight from the padded encodings: tokenizers has already made every row
        # the same length, so this is a single contiguous allocation, not a pad loop.
        ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)

        feeds: dict[str, np.ndarray] = {"input_ids": ids, "attention_mask": mask}
        if self._wants_token_type:
            # XLM-R has no segment embeddings; some exports still declare the input.
            feeds["token_type_ids"] = np.zeros_like(ids)
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}

        # Requesting exactly one output stops ORT from materialising pooler_output.
        hidden = self._run_with_retry(feeds)

        if hidden.ndim == 3:
            # Mean pooling, NOT CLS. e5 is trained with mean pooling; taking hidden[:, 0]
            # here would still produce 384 finite numbers and silently lose several points
            # of nDCG, which is the worst kind of bug -- it never raises.
            m = mask.astype(np.float32)[:, :, None]
            pooled = (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
        else:
            pooled = hidden.astype(np.float32, copy=False)

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)  # a zero vector would otherwise yield NaNs
        pooled /= norms
        self._n_forward_batches += 1
        return np.ascontiguousarray(pooled, dtype=np.float32)

    def _run_with_retry(self, feeds: dict[str, np.ndarray]) -> np.ndarray:
        """Bounded retry around Run. ORT failures are usually deterministic, but an
        allocation failure under memory pressure is transient and worth one more go."""
        attempts = max(1, settings.max_retries)
        for attempt in range(attempts):
            try:
                return self._session.run([self._output_name], feeds)[0]
            except Exception as exc:  # ORT raises its own Fail/RuntimeException
                if attempt == attempts - 1:
                    raise EmbedderUnavailableError(
                        f"onnxruntime Run failed on {self.name} after {attempts} attempts: {exc}"
                    ) from exc
                log.warning("onnx run failed (attempt %d/%d): %s", attempt + 1, attempts, exc)
                # Blocking sleep is safe: the async path only ever calls into here from
                # asyncio.to_thread, never from the event loop.
                time.sleep(settings.retry_backoff_ms / 1000.0 * (attempt + 1))
        raise AssertionError("unreachable")

    def _encode_uncached(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        """Prefix, split into batches, encode. Bypasses the cache entirely."""
        prefix = settings.embed_query_prefix if is_query else settings.embed_passage_prefix
        if is_query:
            self._n_query_encodes += len(texts)
        else:
            self._n_passage_encodes += len(texts)
        # The startswith guard makes double-prefixing impossible ("query: query: ..." is a
        # different, worse point in embedding space) without needing callers to be careful.
        prefixed = [t if t.startswith(prefix) else prefix + t for t in texts]

        size = max(1, settings.embed_batch_size)
        if len(prefixed) <= size:
            return self._forward(prefixed)

        # Length-bucketed batching. Padding is dynamic (to the longest item in the batch), so
        # in input order a single long chunk drags its whole batch up to its length. Measured
        # on this corpus: chunk token lengths are mean 89 / p50 87 / p95 135 / max 192, so a
        # randomly-ordered batch of 128 almost always contains something near 192 and pays
        # 192 tokens of compute for an average of 89 -- 2.17x waste. Sorting by length first
        # makes each batch nearly uniform and reclaims most of that.
        #
        # Sorted on CHARACTER length, not token length: tokenising up front purely to order
        # the batches would cost a second pass over the whole input. Characters correlate
        # closely enough with tokens here to bucket well, and a mis-bucketed item costs a few
        # wasted pad positions, never a wrong result.
        #
        # Only the batched (indexing) path does this. It is pure overhead for the
        # single-query path above, which returns before reaching here.
        order = sorted(range(len(prefixed)), key=lambda i: len(prefixed[i]))
        out = np.empty((len(prefixed), self.dim), dtype=np.float32)
        for start in range(0, len(order), size):
            idx = order[start : start + size]
            vecs = self._forward([prefixed[i] for i in idx])
            # Scatter back to the caller's order -- the caller pairs these rows with its own
            # list positionally, so returning them in sorted order would silently mis-assign
            # every vector to the wrong chunk.
            out[idx] = vecs
        return out

    def _encode_with_cache(
        self, texts: list[str], is_query: bool, *, cache_only: bool
    ) -> np.ndarray | None:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        # Dedupe within the call too: a benchmark replaying a query set, or an ingest batch
        # with boilerplate passages, would otherwise encode the same string repeatedly.
        pending: dict[str, list[int]] = {}
        for i, text in enumerate(texts):
            hit = self._cache.get((text, is_query), count=not cache_only)
            if hit is not None:
                out[i] = hit
            else:
                pending.setdefault(text, []).append(i)

        if pending:
            if cache_only:
                return None
            unique = list(pending)
            vecs = self._encode_uncached(unique, is_query=is_query)
            for text, vec in zip(unique, vecs, strict=True):
                self._cache.put((text, is_query), vec)
                for i in pending[text]:
                    out[i] = vec
        return out

    # -- Embedder protocol --------------------------------------------------------------

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        result = self._encode_with_cache(texts, is_query, cache_only=False)
        assert result is not None  # cache_only=False always produces a full array
        return result

    async def encode_async(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # Fast path: an all-hit lookup is ~1us, while asyncio.to_thread costs tens of us of
        # executor scheduling. Doing the hop anyway would make cached queries slower.
        cached = self._encode_with_cache(texts, is_query, cache_only=True)
        if cached is not None:
            self._cache.note_hits(len(texts))
            return cached
        return await asyncio.to_thread(self.encode, texts, is_query=is_query)

    # -- operations ---------------------------------------------------------------------

    def warmup(self) -> None:
        """Run throwaway forward passes so the first real query is not the first ONNX call.

        The first Run allocates arenas, JITs kernels and faults the weights in from disk;
        it is routinely 10-50x steady state and on its own would blow the 200ms budget.
        Both a batch of 1 and a batch of >1 are exercised because ORT re-plans memory per
        distinct input shape. Deliberately bypasses the cache so warmup traffic cannot
        pollute `cache_stats()` and flatter the reported hit rate.
        """
        self._encode_uncached(["वार्म अप"], is_query=True)
        self._encode_uncached(["warm up passage", "இரண்டாவது பத்தி"], is_query=False)

    def cache_stats(self) -> dict[str, Any]:
        """Cache counters plus encode provenance, for the trace and the analytics report."""
        stats = self._cache.stats()
        stats.update(
            {
                "query_encodes": self._n_query_encodes,
                "passage_encodes": self._n_passage_encodes,
                "forward_batches": self._n_forward_batches,
                "model": self.name,
                "threads": settings.embed_threads,
            }
        )
        return stats


# --------------------------------------------------------------------------------------
# Process-wide singleton
# --------------------------------------------------------------------------------------

_INSTANCE: OnnxEncoder | None = None
_INSTANCE_LOCK = threading.Lock()


def get_embedder(*, warmup: bool = True) -> OnnxEncoder:
    """The one encoder for this process. Loading twice would double ~120MB of weights.

    Double-checked locking: concurrent first requests (uvicorn worker threads, or the
    ingest pool) must block on one load rather than each build their own session. Warmup
    happens inside the lock so the second caller gets a *warm* model, not a cold one.
    """
    global _INSTANCE
    instance = _INSTANCE
    if instance is not None:
        return instance
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            started = time.perf_counter()
            encoder = OnnxEncoder()
            if warmup:
                encoder.warmup()
            log.info(
                "embedder ready: %s dim=%d in %.0fms",
                encoder.name,
                encoder.dim,
                (time.perf_counter() - started) * 1000.0,
            )
            _INSTANCE = encoder
    return _INSTANCE


__all__ = [
    "ONNX_CANDIDATES",
    "EmbedderUnavailableError",
    "OnnxEncoder",
    "get_embedder",
]
