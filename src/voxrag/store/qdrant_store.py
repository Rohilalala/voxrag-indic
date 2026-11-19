"""Qdrant-backed vector store -- the "real vector database" arm of the system.

Two very different things ship under the name "Qdrant" and conflating them would make the
latency numbers a lie:

* SERVER mode (`settings.qdrant_url`, i.e. `docker compose up -d`) is the benchmarked
  configuration. It builds an HNSW graph from `hnsw_m` / `hnsw_ef_construct`, honours
  `SearchParams(hnsw_ef=...)` per query, and supports KEYWORD payload indexes so a
  `lang`/`strategy` filter is an indexed lookup rather than a scan.
* LOCAL mode (`settings.qdrant_path`) is the embedded, zero-setup dev path. Qdrant itself
  warns that it "performs exact (brute-force) search, so search_params is ignored" and that
  "payload indexes have no effect in the local Qdrant". There is no graph and no ef knob:
  every query scans every vector, so cost grows linearly with the corpus. Perfectly fine for
  a 5k-chunk dev loop, dishonest as an ANN latency claim.

So the store detects which one it is at construction, skips the calls the local backend
would only warn about, says so in the log, and exposes `mode` so /api/health and the
benchmark can label their numbers correctly.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from voxrag.config import settings
from voxrag.contracts import Chunk, ChunkStrategyName, Lang, ScoredChunk
from voxrag.store.quantization import normalise_kind, quantization_config, search_params

_LOG = logging.getLogger(__name__)

#: Points per upsert request. Large enough to amortise the HTTP round trip, small enough
#: that one request body stays a few MB with 384-dim float vectors plus chunk payloads.
_UPSERT_BATCH = 256

#: Socket timeout for the client, in seconds. Deliberately far above the harness's 80ms
#: retrieve_dense budget: the harness owns request deadlines, this only catches a dead peer.
_CLIENT_TIMEOUT_S = 10

#: gRPC shaves ~1ms off a round trip versus REST, but only when 6334 is published. Left off
#: because the default `docker run -p 6333:6333` does not expose it and a silent connection
#: failure is worse than a millisecond. Not a `settings` field; flip here if 6334 is open.
PREFER_GRPC = False

#: chunk_id -> point id. Qdrant only accepts unsigned ints or UUIDs as point ids, and uuid5
#: is deterministic, so re-indexing the same chunk overwrites it instead of duplicating it.
_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

#: Transport-level failures worth retrying. Semantic errors (4xx, bad vector shape) are not.
#: qdrant_client funnels every httpx failure through ResponseHandlingException, so that one
#: entry covers connect/read/timeout; the rest guard the gRPC and embedded paths.
_RETRYABLE: tuple[type[BaseException], ...] = (
    ResponseHandlingException,
    TimeoutError,
    OSError,
)

#: Ceiling on an honoured Retry-After. A cloud Qdrant under quota pressure can ask for
#: minutes, which is a legitimate answer for an ingest and an absurd one inside a request.
_MAX_RETRY_AFTER_S = 2.0


class QdrantStore:
    """VectorStore backed by Qdrant. Implements `voxrag.contracts.VectorStore`."""

    name = "qdrant"

    def __init__(
        self,
        *,
        url: str | None = None,
        path: Path | str | None = None,
        collection: str | None = None,
        local: bool = False,
    ) -> None:
        resolved_url = None if local else (url if url is not None else settings.qdrant_url)
        self.collection = collection or settings.qdrant_collection
        #: Validated here rather than at query time: a typo must fail the process at start-up,
        #: not silently serve float32 under a benchmark labelled "scalar".
        self.quantization = normalise_kind(settings.quantization)
        self._ready = False
        self._init_lock = asyncio.Lock()
        #: Filters are rebuilt for every query otherwise; there are at most 15*8 distinct
        #: combinations, so caching the pydantic objects removes an allocation per request.
        self._filter_cache: dict[tuple[str | None, str | None], models.Filter] = {}

        if resolved_url:
            self.is_local = False
            self.mode = "server"
            self.location = resolved_url
            self._client = AsyncQdrantClient(
                url=resolved_url,
                api_key=settings.qdrant_api_key,
                prefer_grpc=PREFER_GRPC,
                timeout=_CLIENT_TIMEOUT_S,
            )
            _LOG.info(
                "qdrant store ready: mode=server url=%s collection=%s "
                "hnsw(m=%d, ef_construct=%d, ef_search=%d), quantization=%s, "
                "keyword payload indexes on lang+strategy",
                self.location,
                self.collection,
                settings.hnsw_m,
                settings.hnsw_ef_construct,
                settings.hnsw_ef_search,
                self.quantization,
            )
        else:
            store_path = Path(path) if path is not None else Path(settings.qdrant_path)
            store_path.mkdir(parents=True, exist_ok=True)
            self.is_local = True
            self.mode = "local"
            self.location = str(store_path)
            self._client = AsyncQdrantClient(
                path=str(store_path),
                # Embedded Qdrant keeps sqlite handles that would otherwise refuse to be
                # touched from the threads uvicorn hands requests to.
                force_disable_check_same_thread=True,
            )
            _LOG.warning(
                "qdrant store ready: mode=local path=%s collection=%s -- embedded Qdrant does "
                "EXACT BRUTE-FORCE search: no HNSW graph, hnsw_ef_search=%d ignored, payload "
                "indexes skipped. Query cost grows linearly with corpus size, so any latency "
                "measured here is not an ANN number. Set VOXRAG_QDRANT_URL to a real server "
                "before benchmarking.",
                self.location,
                self.collection,
                settings.hnsw_ef_search,
            )

    # -- VectorStore ---------------------------------------------------------------------

    async def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] != len(chunks):
            raise ValueError(f"vectors shape {vecs.shape} does not match {len(chunks)} chunks")
        if vecs.shape[1] != settings.embed_dim:
            raise ValueError(
                f"vectors have dim {vecs.shape[1]}, collection expects {settings.embed_dim}"
            )
        await self._ensure()

        for start in range(0, len(chunks), _UPSERT_BATCH):
            batch = chunks[start : start + _UPSERT_BATCH]
            rows = vecs[start : start + _UPSERT_BATCH].tolist()
            points = [
                models.PointStruct(
                    id=_point_id(chunk.chunk_id), vector=row, payload=_payload(chunk)
                )
                for chunk, row in zip(batch, rows, strict=True)
            ]
            # wait=False: do not block the ingest loop on the indexing optimiser. The
            # collection converges shortly after the last batch; `count()` reflects that.
            await self._call(
                "upsert",
                lambda p=points: self._client.upsert(self.collection, points=p, wait=False),
                backoff_ms=settings.retry_backoff_ms,
            )

    async def search(
        self,
        vector: np.ndarray,
        *,
        top_k: int = 20,
        lang: Lang | None = None,
        strategy: ChunkStrategyName | None = None,
    ) -> list[ScoredChunk]:
        if top_k <= 0:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != settings.embed_dim:
            raise ValueError(
                f"query vector has dim {query.shape[0]}, collection expects {settings.embed_dim}"
            )
        await self._ensure()

        query_filter = self._filter_for(lang, strategy)
        # Passing search_params in local mode only earns a UserWarning; the backend is
        # brute-force and has no ef to tune (and no quantization either).
        #
        # Built per request rather than cached at construction so that reassigning
        # `store.quantization` takes effect immediately -- that is how
        # measure_quantization.py compares kinds over one collection without rebuilding the
        # store, i.e. on the code path that actually serves traffic.
        params = (
            None
            if self.is_local
            else search_params(
                self.quantization,
                rescore=settings.quantization_rescore,
                oversampling=settings.quantization_oversampling,
                hnsw_ef=settings.hnsw_ef_search,
            )
        )
        payload = query.tolist()
        response = await self._call(
            "search",
            lambda: self._client.query_points(
                self.collection,
                query=payload,
                limit=top_k,
                query_filter=query_filter,
                search_params=params,
                with_payload=True,
                with_vectors=False,
            ),
            # No sleep on the read path: the failure this retry actually recovers from is a
            # keep-alive socket the server closed, which succeeds again immediately. Anything
            # slower than that is the harness's stage timeout to deal with, not ours.
            backoff_ms=0.0,
        )

        hits: list[ScoredChunk] = []
        for rank, point in enumerate(response.points, start=1):
            raw = (point.payload or {}).get("chunk")
            if not raw:
                continue
            try:
                chunk = Chunk.model_validate(raw)
            except ValidationError:
                _LOG.warning("qdrant point %s holds an unparseable chunk payload", point.id)
                continue
            score = float(point.score)
            hits.append(ScoredChunk(chunk=chunk, score=score, dense_score=score, dense_rank=rank))
        return hits

    async def count(self) -> int:
        await self._ensure()
        result = await self._call(
            "count",
            lambda: self._client.count(self.collection, exact=True),
            backoff_ms=settings.retry_backoff_ms,
        )
        return int(result.count)

    async def health(self) -> bool:
        try:
            await self._ensure()
            await self._client.get_collection(self.collection)
            return True
        except Exception as exc:
            _LOG.warning("qdrant health check failed (%s mode, %s): %s", self.mode, self.location, exc)
            return False

    # -- extras used by the API and the benchmark ----------------------------------------

    async def warmup(self) -> None:
        """Create the collection and open a connection before the first user request.

        Otherwise request #1 pays collection creation plus a TCP+TLS handshake inside the
        200ms budget and shows up as a fake latency outlier at P100.
        """
        try:
            await self._ensure()
            probe = np.zeros(settings.embed_dim, dtype=np.float32)
            probe[0] = 1.0
            await self.search(probe, top_k=1)
            await self.check_language_coverage()
        except Exception as exc:
            _LOG.warning("qdrant warmup failed (continuing): %s", exc)

    async def check_language_coverage(self) -> list[Lang]:
        """Log, loudly, every configured language this collection cannot serve.

        A `lang` filter for a language the collection does not hold matches nothing, and
        Qdrant answers an empty result -- not an error, so nothing retries, nothing degrades
        and nothing is logged. The system then serves that language from the lexical arm
        alone and reports itself healthy. That is exactly how this collection was pointed at
        the earlier three-language index while claiming fourteen, and it cost the dense arm
        of eleven languages. It is a startup-time question, so it is asked at startup.
        """
        missing: list[Lang] = []
        for lang in settings.ingest_langs:
            result = await self._call(
                "count",
                lambda lang=lang: self._client.count(
                    self.collection, count_filter=self._filter_for(lang, None), exact=False
                ),
                backoff_ms=settings.retry_backoff_ms,
            )
            if int(result.count) == 0:
                missing.append(lang)
        if missing:
            _LOG.error(
                "collection %r holds NO passages in %s; the dense arm will return empty "
                "results for those languages and the system will silently serve them from "
                "the lexical arm alone. Point VOXRAG_QDRANT_COLLECTION at an index that "
                "covers every configured language.",
                self.collection,
                ", ".join(lang.value for lang in missing),
            )
        return missing

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "location": self.location,
            "collection": self.collection,
            "dim": settings.embed_dim,
            "m": settings.hnsw_m,
            "ef_construct": settings.hnsw_ef_construct,
            "ef_search": None if self.is_local else settings.hnsw_ef_search,
            #: Reported because shipping a quantized index without saying so would make the
            #: published retrieval numbers describe a different collection than the served one.
            "quantization": "none" if self.is_local else self.quantization,
            #: True means brute force: results are exact, latency is O(corpus).
            "exact": self.is_local,
            "payload_indexed": not self.is_local,
        }

    async def close(self) -> None:
        await self._client.close()
        self._ready = False

    # -- internals -----------------------------------------------------------------------

    async def _ensure(self) -> None:
        """Idempotent, concurrency-safe collection bootstrap.

        `_ready` is only latched on success, so a server that was down when the process
        started heals on the next request instead of poisoning the store forever.
        """
        if self._ready:
            return
        async with self._init_lock:
            if self._ready:
                return
            await self._create_collection()
            self._ready = True

    async def _create_collection(self) -> None:
        exists = await self._call(
            "collection_exists",
            lambda: self._client.collection_exists(self.collection),
            backoff_ms=settings.retry_backoff_ms,
        )
        if not exists:
            try:
                await self._client.create_collection(
                    self.collection,
                    vectors_config=models.VectorParams(
                        size=settings.embed_dim, distance=models.Distance.COSINE
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=settings.hnsw_m, ef_construct=settings.hnsw_ef_construct
                    ),
                    # None when quantization is off, which is what an unquantized collection
                    # looks like. Only applies to a collection this call creates: an existing
                    # 2.27M index is patched with quantization.apply_to_collection instead,
                    # which re-encodes in place rather than re-embedding.
                    quantization_config=quantization_config(self.quantization),
                )
            except UnexpectedResponse as exc:
                # 409: another worker created it between the check and the create.
                if exc.status_code != 409:
                    raise

        if self.is_local:
            return  # payload indexes have no effect in embedded mode; calling only warns

        for field in ("lang", "strategy"):
            try:
                await self._client.create_payload_index(
                    self.collection,
                    field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception as exc:
                # A missing index costs recall-preserving speed, not correctness: Qdrant
                # falls back to filtering during traversal. Never fail startup over it.
                _LOG.warning("could not create payload index on %r: %s", field, exc)

    def _filter_for(
        self, lang: Lang | None, strategy: ChunkStrategyName | None
    ) -> models.Filter | None:
        if lang is None and strategy is None:
            return None
        key = (lang.value if lang else None, strategy.value if strategy else None)
        cached = self._filter_cache.get(key)
        if cached is None:
            conditions = [
                models.FieldCondition(key=field, match=models.MatchValue(value=value))
                for field, value in (("lang", key[0]), ("strategy", key[1]))
                if value is not None
            ]
            cached = models.Filter(must=conditions)
            self._filter_cache[key] = cached
        return cached

    async def _call(self, what: str, fn: Any, *, backoff_ms: float) -> Any:
        """Run one client call with bounded retries on transport-level failure.

        There is no second vector database to fall back to, so the contract here is: retry
        what is worth retrying, then raise a clear error. The harness turns that into a
        recorded degradation rather than a dead request.
        """
        attempts = max(1, settings.max_retries)
        last: BaseException | None = None
        for attempt in range(attempts):
            sleep_s = backoff_ms * (attempt + 1) / 1000.0
            try:
                return await fn()
            except UnexpectedResponse as exc:
                if exc.status_code < 500:
                    raise
                last = exc
            except ResourceExhaustedResponse as exc:
                # The server told us exactly how long to wait; ignoring it just burns quota.
                last = exc
                sleep_s = min(float(exc.retry_after_s), _MAX_RETRY_AFTER_S)
            except _RETRYABLE as exc:
                last = exc
            if attempt + 1 < attempts and sleep_s > 0:
                await asyncio.sleep(sleep_s)
        raise RuntimeError(
            f"qdrant {what} failed after {attempts} attempts ({self.mode} mode, {self.location}): {last}"
        ) from last


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def _payload(chunk: Chunk) -> dict[str, Any]:
    """Payload carrying the whole Chunk plus the two filterable fields at the top level.

    The duplication is deliberate: top-level scalars are what `create_payload_index` can
    index, and storing the full chunk means a search returns hydrated ScoredChunks with no
    second round trip to a document store -- which would cost more than the ANN search does.
    """
    dumped = chunk.model_dump(mode="json")
    return {"lang": dumped["lang"], "strategy": dumped["strategy"], "chunk": dumped}


__all__ = ["QdrantStore"]
