"""In-process HNSW index -- the latency floor the rest of the system is measured against.

hnswlib runs inside this process: no socket, no JSON, no protobuf, no cross-process copy.
Its search time is the ANN graph traversal and essentially nothing else, which makes it the
honest lower bound for the <200ms budget. Whatever Qdrant costs above this number is the
price of running a real vector database, and the benchmark reports both so that price is
visible rather than hidden.

Persistence is the index file plus a JSONL sidecar holding the full Chunk per label, so a
hit returns a hydrated ScoredChunk with no second lookup.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import hnswlib
import numpy as np
from pydantic import ValidationError

from voxrag.config import settings
from voxrag.contracts import Chunk, ChunkStrategyName, Lang, ScoredChunk

_LOG = logging.getLogger(__name__)

_INDEX_FILE = "index.bin"
_CHUNKS_FILE = "chunks.jsonl"
_META_FILE = "meta.json"

#: hnswlib pre-allocates its graph and `resize_index` copies the whole thing, so start
#: large enough that a laptop-sized ingest never pays for a resize. 65k * 384 * 4B = 100MB.
_INITIAL_CAPACITY = 65_536

#: Filtered searches over-fetch by this factor. hnswlib applies the predicate *during*
#: traversal, so a selective filter makes the graph walk drift away from the true top-k;
#: asking for more candidates and trimming afterwards recovers most of that recall.
_FILTER_OVERFETCH = 4

#: Vectors added since the last flush before the store writes itself to disk. Saving is a
#: full rewrite of the graph file, so it must not happen per batch during a large ingest.
_AUTOSAVE_EVERY = 50_000

#: Compact integer codes for the two filterable fields. The hnswlib filter callback runs
#: once per visited node inside the C++ search loop, so it must be the cheapest possible
#: Python: one list index and one int compare. String comparison here is measurably slower.
_LANG_CODE: dict[Lang, int] = {lang: i for i, lang in enumerate(Lang)}
_STRAT_CODE: dict[ChunkStrategyName, int] = {s: i for i, s in enumerate(ChunkStrategyName)}
_UNKNOWN_CODE = -1


class _RWLock:
    """Many concurrent searches, one exclusive writer.

    hnswlib tolerates concurrent `knn_query` calls but not a query racing `add_items`, and
    `resize_index` reallocates the graph out from under any in-flight reader. A plain mutex
    would serialise every search and destroy the concurrency the latency benchmark exercises,
    so reads share and writes exclude. Writer-preferring: a long ingest must not be starved
    by a steady query load.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writing = False
        self._waiting_writers = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        with self._cond:
            while self._writing or self._waiting_writers:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        with self._cond:
            self._waiting_writers += 1
            while self._writing or self._readers:
                self._cond.wait()
            self._waiting_writers -= 1
            self._writing = True
        try:
            yield
        finally:
            with self._cond:
                self._writing = False
                self._cond.notify_all()


class HnswStore:
    """VectorStore backed by hnswlib. Implements `voxrag.contracts.VectorStore`."""

    name = "hnsw"

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else Path(settings.hnsw_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dim = settings.embed_dim
        #: Mirrors QdrantStore.mode so /api/health can report both stores uniformly.
        self.mode = "in-process"

        self._lock = _RWLock()
        self._index = hnswlib.Index(space="cosine", dim=self.dim)
        self._chunks: dict[int, Chunk] = {}
        self._label_of: dict[str, int] = {}
        #: Parallel arrays indexed by label; see _LANG_CODE for why they are ints.
        self._lang_key: list[int] = []
        self._strat_key: list[int] = []
        self._key_counts: Counter[tuple[int, int]] = Counter()
        self._next_label = 0
        self._dirty = 0
        #: False until the sidecar on disk matches our labels; controls append-vs-truncate.
        self._sidecar_valid = False

        loaded = self._load()
        if not loaded:
            self._index.init_index(
                max_elements=_INITIAL_CAPACITY,
                ef_construction=settings.hnsw_ef_construct,
                M=settings.hnsw_m,
            )
        # ef is not restored by load_index (it comes back as the build-time default), so it
        # must be set explicitly every time or search silently runs at the wrong recall.
        self._index.set_ef(max(1, settings.hnsw_ef_search))
        self._ef = self._index.ef

        _LOG.info(
            "hnsw store ready: mode=in-process path=%s vectors=%d dim=%d M=%d ef_construct=%d ef_search=%d",
            self.path,
            len(self._chunks),
            self.dim,
            settings.hnsw_m,
            settings.hnsw_ef_construct,
            self._ef,
        )
        atexit.register(self._flush_at_exit)

    # -- VectorStore ---------------------------------------------------------------------

    async def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        vecs = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if vecs.ndim != 2 or vecs.shape[0] != len(chunks):
            raise ValueError(
                f"vectors shape {vecs.shape} does not match {len(chunks)} chunks"
            )
        if vecs.shape[1] != self.dim:
            raise ValueError(f"vectors have dim {vecs.shape[1]}, index expects {self.dim}")
        # add_items releases the GIL but still costs milliseconds-to-seconds per batch;
        # holding the event loop for that would stall every concurrent request.
        await asyncio.to_thread(self._upsert_sync, chunks, vecs)

    async def search(
        self,
        vector: np.ndarray,
        *,
        top_k: int = 20,
        lang: Lang | None = None,
        strategy: ChunkStrategyName | None = None,
    ) -> list[ScoredChunk]:
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError(f"query vector has dim {query.shape[1]}, index expects {self.dim}")
        return await asyncio.to_thread(self._search_sync, query, top_k, lang, strategy)

    async def count(self) -> int:
        return len(self._chunks)

    async def health(self) -> bool:
        # A query against an uninitialised hnswlib index segfaults rather than raising, so
        # health is a structural check, never a probe.
        return self._index.dim == self.dim and self._index.get_max_elements() > 0

    # -- extras used by the API and the benchmark ----------------------------------------

    async def warmup(self) -> None:
        """Touch the search path once so the first real query pays no first-call cost."""
        if not self._chunks:
            return
        zero = np.zeros((1, self.dim), dtype=np.float32)
        zero[0, 0] = 1.0  # cosine space rejects an all-zero probe
        await asyncio.to_thread(self._search_sync, zero, 1, None, None)

    async def save(self) -> None:
        """Force a flush of the graph + sidecar to disk."""
        await asyncio.to_thread(self._save_sync)

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "path": str(self.path),
            "vectors": len(self._chunks),
            "dim": self.dim,
            "m": settings.hnsw_m,
            "ef_construct": settings.hnsw_ef_construct,
            "ef_search": self._ef,
            "exact": False,
        }

    # -- internals -----------------------------------------------------------------------

    def _upsert_sync(self, chunks: list[Chunk], vecs: np.ndarray) -> None:
        labels = np.empty(len(chunks), dtype=np.int64)
        with self._lock.write():
            for i, chunk in enumerate(chunks):
                lang_code = _LANG_CODE.get(chunk.lang, _UNKNOWN_CODE)
                strat_code = _STRAT_CODE.get(chunk.strategy, _UNKNOWN_CODE)
                label = self._label_of.get(chunk.chunk_id)
                if label is None:
                    # Labels are handed out densely from 0, which is what lets the filter
                    # callback index a plain list instead of hashing a dict.
                    label = self._next_label
                    self._next_label += 1
                    self._label_of[chunk.chunk_id] = label
                    self._lang_key.append(lang_code)
                    self._strat_key.append(strat_code)
                else:
                    self._key_counts[(self._lang_key[label], self._strat_key[label])] -= 1
                    self._lang_key[label] = lang_code
                    self._strat_key[label] = strat_code
                self._key_counts[(lang_code, strat_code)] += 1
                self._chunks[label] = chunk
                labels[i] = label

            needed = self._next_label
            capacity = self._index.get_max_elements()
            if needed > capacity:
                # Double rather than grow-to-fit: resize copies the entire graph, so a
                # per-batch exact resize turns ingest into O(n^2) memcpy.
                self._index.resize_index(max(needed, capacity * 2))
            self._index.add_items(vecs, labels)

            self._append_sidecar(chunks, labels)
            self._dirty += len(chunks)
            if self._dirty >= _AUTOSAVE_EVERY:
                self._save_locked()

    def _search_sync(
        self,
        query: np.ndarray,
        top_k: int,
        lang: Lang | None,
        strategy: ChunkStrategyName | None,
    ) -> list[ScoredChunk]:
        if top_k <= 0:
            return []
        lang_code = _LANG_CODE.get(lang, _UNKNOWN_CODE) if lang is not None else None
        strat_code = _STRAT_CODE.get(strategy, _UNKNOWN_CODE) if strategy is not None else None

        with self._lock.read():
            total = len(self._chunks)
            if total == 0:
                return []

            if lang_code is None and strat_code is None:
                predicate = None
                k = min(top_k, total)
            else:
                matching = self._matching_count(lang_code, strat_code)
                if matching == 0:
                    return []
                # Over-fetch, then trim. Clamped by the exact number of matching chunks
                # because hnswlib raises rather than returning fewer neighbours than asked.
                k = min(top_k * _FILTER_OVERFETCH, matching)
                predicate = self._predicate(lang_code, strat_code)
            k = max(1, k)

            labels, dists = self._knn(query, k, predicate)
            hits: list[ScoredChunk] = []
            for rank, (label, dist) in enumerate(zip(labels[0], dists[0], strict=True), start=1):
                chunk = self._chunks.get(int(label))
                if chunk is None:
                    continue  # label survived in the graph but not in the sidecar
                # hnswlib's cosine "distance" is 1 - cosine similarity.
                score = float(1.0 - dist)
                hits.append(
                    ScoredChunk(chunk=chunk, score=score, dense_score=score, dense_rank=rank)
                )
                if len(hits) >= top_k:
                    break
            return hits

    def _knn(
        self, query: np.ndarray, k: int, predicate: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """knn_query with the short-result guard.

        hnswlib raises RuntimeError ("cannot return the results in a contiguous 2D array")
        whenever it finds fewer than k neighbours -- which a selective filter can cause even
        after the count-based clamp, because the traversal may simply not reach the matching
        region. Halving k and retrying degrades the result instead of failing the request.
        num_threads=1 because hnswlib parallelises across query *rows* and we always send one:
        extra threads would only add pool overhead to a sub-millisecond call.
        """
        while True:
            try:
                if predicate is None:
                    return self._index.knn_query(query, k=k, num_threads=1)
                return self._index.knn_query(query, k=k, filter=predicate, num_threads=1)
            except RuntimeError:
                if k <= 1:
                    return np.empty((1, 0), dtype=np.uint64), np.empty((1, 0), dtype=np.float32)
                k //= 2

    def _predicate(self, lang_code: int | None, strat_code: int | None) -> Any:
        """Build the cheapest closure that expresses this filter. Bound locally on purpose:
        a local variable lookup beats an attribute lookup, and this runs per visited node."""
        lang_key = self._lang_key
        strat_key = self._strat_key
        if lang_code is not None and strat_code is not None:
            return lambda label: lang_key[label] == lang_code and strat_key[label] == strat_code
        if lang_code is not None:
            return lambda label: lang_key[label] == lang_code
        return lambda label: strat_key[label] == strat_code

    def _matching_count(self, lang_code: int | None, strat_code: int | None) -> int:
        """Exact number of indexed chunks the filter admits, in O(distinct keys) (<= 15*8)."""
        if lang_code is None and strat_code is None:
            return len(self._chunks)
        return sum(
            n
            for (lc, sc), n in self._key_counts.items()
            if (lang_code is None or lc == lang_code) and (strat_code is None or sc == strat_code)
        )

    # -- persistence ---------------------------------------------------------------------

    def _append_sidecar(self, chunks: list[Chunk], labels: np.ndarray) -> None:
        """Append-only JSONL. A full rewrite per batch would dominate ingest time; on load,
        a later record for the same label simply overwrites the earlier one."""
        mode = "a" if self._sidecar_valid else "w"
        with (self.path / _CHUNKS_FILE).open(mode, encoding="utf-8") as fh:
            # Splicing model_dump_json() straight into the line avoids a second
            # dict -> json round trip; at a million chunks that difference is minutes.
            fh.writelines(
                f'{{"label":{int(label)},"chunk":{chunk.model_dump_json()}}}\n'
                for chunk, label in zip(chunks, labels, strict=True)
            )
        self._sidecar_valid = True

    def _save_sync(self) -> None:
        with self._lock.write():
            self._save_locked()

    def _save_locked(self) -> None:
        self._index.save_index(str(self.path / _INDEX_FILE))
        (self.path / _META_FILE).write_text(
            json.dumps(
                {
                    "space": "cosine",
                    "dim": self.dim,
                    "m": settings.hnsw_m,
                    "ef_construct": settings.hnsw_ef_construct,
                    "count": len(self._chunks),
                    "next_label": self._next_label,
                }
            ),
            encoding="utf-8",
        )
        self._dirty = 0

    def _load(self) -> bool:
        index_file = self.path / _INDEX_FILE
        meta_file = self.path / _META_FILE
        chunks_file = self.path / _CHUNKS_FILE
        if not (index_file.exists() and meta_file.exists() and chunks_file.exists()):
            return False
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("hnsw meta unreadable (%s); rebuilding empty index", exc)
            return False
        # load_index reads the geometry from the file but keeps the dim we constructed with,
        # so a stale index of the wrong dimension loads and then segfaults on query. The meta
        # file exists purely to make that mismatch detectable before we touch the C++ side.
        if meta.get("dim") != self.dim or meta.get("space") != "cosine":
            _LOG.warning(
                "hnsw index at %s is dim=%s/%s but settings want dim=%d/cosine; rebuilding",
                self.path,
                meta.get("dim"),
                meta.get("space"),
                self.dim,
            )
            return False

        try:
            self._index.load_index(str(index_file))
        except (RuntimeError, OSError) as exc:
            _LOG.warning("hnsw index load failed (%s); rebuilding empty index", exc)
            self._index = hnswlib.Index(space="cosine", dim=self.dim)
            return False

        key_by_label: dict[int, tuple[int, int]] = {}
        try:
            with chunks_file.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        chunk = Chunk.model_validate(record["chunk"])
                    except (json.JSONDecodeError, KeyError, ValidationError):
                        continue  # a torn last line from a killed ingest must not be fatal
                    label = int(record["label"])
                    self._chunks[label] = chunk
                    self._label_of[chunk.chunk_id] = label
                    key_by_label[label] = (
                        _LANG_CODE.get(chunk.lang, _UNKNOWN_CODE),
                        _STRAT_CODE.get(chunk.strategy, _UNKNOWN_CODE),
                    )
        except OSError as exc:
            _LOG.warning("hnsw sidecar unreadable (%s); rebuilding empty index", exc)
            self._index = hnswlib.Index(space="cosine", dim=self.dim)
            self._chunks.clear()
            self._label_of.clear()
            return False

        # The filter callback indexes these lists with whatever label hnswlib hands back, so
        # they must cover every label in the graph, not just the ones the sidecar knew about.
        graph_labels = self._index.get_ids_list()
        highest = max(
            [int(meta.get("next_label", 0)) - 1, -1]
            + ([max(graph_labels)] if graph_labels else [])
            + ([max(key_by_label)] if key_by_label else [])
        )
        size = highest + 1
        self._lang_key = [_UNKNOWN_CODE] * size
        self._strat_key = [_UNKNOWN_CODE] * size
        for label, (lang_code, strat_code) in key_by_label.items():
            self._lang_key[label] = lang_code
            self._strat_key[label] = strat_code
            self._key_counts[(lang_code, strat_code)] += 1
        self._next_label = size
        self._sidecar_valid = True

        if len(graph_labels) != len(self._chunks):
            _LOG.warning(
                "hnsw index holds %d vectors but sidecar holds %d chunks; "
                "vectors without a chunk will be skipped at search time",
                len(graph_labels),
                len(self._chunks),
            )
        return True

    def _flush_at_exit(self) -> None:
        if self._dirty == 0:
            return
        try:
            self._save_sync()
        except Exception as exc:  # interpreter shutdown must never raise out of atexit
            _LOG.warning("hnsw flush on exit failed: %s", exc)


__all__ = ["HnswStore"]
