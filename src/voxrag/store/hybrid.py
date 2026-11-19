"""Reciprocal Rank Fusion and the two-arm retriever that feeds it.

Why RRF and not a weighted sum of the raw scores: BM25 is unbounded and its scale moves with
document length, vocabulary size and query length, while cosine sits in [-1,1]. Any attempt to
put them on one axis needs a per-query normalisation (min-max over the candidate list is the
usual one), and that normalisation is itself unstable -- it stretches a set of near-identical
scores across the full range and rewards whichever arm happened to return a flat result set.
Ranks have none of those problems: they are already comparable, already bounded, and a
retriever that is confidently wrong cannot shout its way to the top of the fused list.

The two arms are independent, so they run concurrently. Dense retrieval crosses a socket to
Qdrant (tens of milliseconds); BM25 is an in-process sparse matrix pass (single-digit).
Serialising them would add the cheap arm's latency to the request for no reason, and the
200ms budget covers both.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Final

import numpy as np

from voxrag.config import settings
from voxrag.contracts import (
    ChunkStrategyName,
    Lang,
    LexicalIndex,
    ScoredChunk,
    StageName,
    VectorStore,
)
from voxrag.telemetry import Trace

_LOG = logging.getLogger(__name__)

#: Fallbacks used only if a key is absent from settings.stage_timeout_ms.
_DENSE_TIMEOUT_MS: Final = 80.0
_LEXICAL_TIMEOUT_MS: Final = 60.0


def reciprocal_rank_fusion(
    dense: Sequence[ScoredChunk],
    lexical: Sequence[ScoredChunk],
    k: int | None = None,
    dense_weight: float | None = None,
) -> list[ScoredChunk]:
    """Fuse two ranked lists into one, sorted best-first.

        score(c) = dense_weight/(k + dense_rank(c)) + (1 - dense_weight)/(k + lexical_rank(c))

    with a term contributed only by the arm that actually retrieved the chunk, so agreement
    between the two arms is what pushes a chunk to the top. `k` damps the head of the curve:
    at k=60 the gap between rank 1 and rank 2 is 1.6%, which stops a single arm's top hit from
    being unbeatable and is the whole point of the constant.

    Ranks come from `dense_rank`/`lexical_rank` when the retriever set them and from list
    position otherwise, so a store that forgets to annotate still fuses correctly. Every raw
    score and rank is carried onto the merged ScoredChunk -- the trace is meant to explain
    *why* a chunk won, which needs both arms' numbers side by side.
    """
    # Resolved here rather than in the signature: a default argument is evaluated once, at
    # import, so `k: int = settings.rrf_k` froze both knobs at whatever they were when this
    # module was first imported. Every runtime override was silently discarded -- which is
    # why the tuning sweep scored dense_weight 0.35 and 1.0 identically to four decimals and
    # concluded that fusion weighting does not matter.
    k = settings.rrf_k if k is None else k
    dense_weight = settings.dense_weight if dense_weight is None else dense_weight
    w_dense = min(max(dense_weight, 0.0), 1.0)
    w_lexical = 1.0 - w_dense
    k = max(1, k)

    # chunk_id -> the field updates that will be applied in one model_copy at the end.
    updates: dict[str, dict[str, Any]] = {}
    base: dict[str, ScoredChunk] = {}

    for hits, weight, score_field, rank_field in (
        (dense, w_dense, "dense_score", "dense_rank"),
        (lexical, w_lexical, "lexical_score", "lexical_rank"),
    ):
        seen: set[str] = set()
        for position, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.chunk_id
            if chunk_id in seen:
                continue  # a duplicate inside one arm must not be paid twice
            seen.add(chunk_id)
            rank = getattr(hit, rank_field) or position
            update = updates.get(chunk_id)
            if update is None:
                update = updates[chunk_id] = {"score": 0.0}
                base[chunk_id] = hit
            update["score"] += weight / (k + rank)
            # Fall back to `score` for stores that report the raw similarity there only.
            raw = getattr(hit, score_field)
            update[score_field] = hit.score if raw is None else raw
            update[rank_field] = rank

    # model_copy(update=...) skips pydantic validation, which a fresh ScoredChunk(...) would
    # re-run for every candidate on every request. At fuse_top_k=12 that is small; it is on
    # the hot path of a 200ms budget, so it is still not free.
    fused = [base[chunk_id].model_copy(update=update) for chunk_id, update in updates.items()]
    fused.sort(key=lambda hit: hit.score, reverse=True)
    return fused


def normalized_confidence(fused: Sequence[ScoredChunk], k: int | None = None) -> float:
    """Evidence strength of the best fused hit, in [0,1].

    Raw RRF scores are tiny (~1/61) and move with `k`, so no fixed threshold could ever be
    written against them -- `settings.min_retrieval_score = 0.28` would reject literally every
    result. We divide by the theoretical maximum instead: 1/(k+1), which is what a chunk
    ranked first by *both* arms scores. The scale is then interpretable:

        1.00  rank 1 in both arms
        0.65  rank 1 dense only          (== dense_weight)
        0.35  rank 1 lexical only        (== 1 - dense_weight)
        0.27  rank ~20 lexical only      -> below the threshold, refuse

    which is exactly the shape the no-relevant-context guardrail needs: single-arm hits from
    deep in the tail cannot clear the bar, agreement always can.
    """
    if not fused:
        return 0.0
    k = settings.rrf_k if k is None else k
    top = max(hit.score for hit in fused)
    best_possible = 1.0 / (max(1, k) + 1)
    return min(1.0, max(0.0, top / best_possible))


class HybridRetriever:
    """Dense + lexical retrieval, fused. Both arms are optional at runtime.

    Failure policy: an arm that times out, raises, or is disabled contributes an empty list
    and a note in the trace. One dead dependency degrades result quality; it never fails the
    request, because the other arm alone is still a working retriever. Both dead returns an
    empty candidate list, which the no-relevant-context guardrail turns into an honest refusal
    rather than a 500.
    """

    name = "hybrid"

    def __init__(
        self,
        store: VectorStore | None = None,
        lexical: LexicalIndex | None = None,
        *,
        enable_lexical: bool | None = None,
    ) -> None:
        self._store = store
        self._lexical = lexical
        self.enable_lexical = settings.enable_lexical if enable_lexical is None else enable_lexical
        self._dense_timeout = settings.stage_timeout_ms.get(
            StageName.RETRIEVE_DENSE.value, _DENSE_TIMEOUT_MS
        )
        self._lexical_timeout = settings.stage_timeout_ms.get(
            StageName.RETRIEVE_LEXICAL.value, _LEXICAL_TIMEOUT_MS
        )

    # Both backends are built lazily and then held: constructing either one loads an index
    # from disk or opens a connection pool, which must never happen inside a request.
    @property
    def store(self) -> VectorStore:
        if self._store is None:
            from voxrag.store import get_vector_store

            self._store = get_vector_store()
        return self._store

    @property
    def lexical(self) -> LexicalIndex:
        if self._lexical is None:
            from voxrag.store.bm25 import get_bm25_index

            self._lexical = get_bm25_index()
        return self._lexical

    async def retrieve(
        self,
        query: str,
        query_vector: np.ndarray,
        *,
        trace: Trace,
        lang: Lang | None = None,
        strategy: ChunkStrategyName | None = None,
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Run both arms concurrently, fuse, and return the top `settings.fuse_top_k`.

        `trace` gets one stage per arm plus FUSE. Because the arms overlap, their stage times
        sum to more than the wall-clock they consumed together -- that is intentional, the
        report needs each arm's own cost, and TOTAL/CORE hold the real elapsed time.
        """
        limit = top_k if top_k is not None else settings.fuse_top_k
        dense_hits, lexical_hits = await asyncio.gather(
            self._dense_arm(query_vector, lang, strategy, trace),
            self._lexical_arm(query, lang, trace),
        )

        with trace.stage(StageName.FUSE) as detail:
            fused = reciprocal_rank_fusion(dense_hits, lexical_hits)
            confidence = normalized_confidence(fused)
            top = fused[:limit]
            detail.update(
                dense=len(dense_hits),
                lexical=len(lexical_hits),
                fused=len(fused),
                returned=len(top),
                confidence=round(confidence, 4),
                rrf_k=settings.rrf_k,
                dense_weight=settings.dense_weight,
                # Overlap is the single most useful fusion diagnostic: near zero means the two
                # arms disagree completely and the weights are doing all the work.
                overlap=sum(
                    1 for hit in fused if hit.dense_rank is not None and hit.lexical_rank is not None
                ),
            )
        return top

    async def warmup(self) -> None:
        """Touch both arms once so the first real query pays no first-call cost."""
        try:
            warm = getattr(self.store, "warmup", None)
            if warm is not None:
                await warm()
        except Exception as exc:  # a cold cache is not worth failing startup over
            _LOG.warning("dense warmup failed: %s", exc)
        if not self.enable_lexical:
            return
        try:
            warm_lexical = getattr(self.lexical, "warmup", None)
            if warm_lexical is not None:
                await asyncio.to_thread(warm_lexical)
        except Exception as exc:
            _LOG.warning("lexical warmup failed: %s", exc)

    # -- arms ----------------------------------------------------------------------------

    async def _dense_arm(
        self,
        query_vector: np.ndarray,
        lang: Lang | None,
        strategy: ChunkStrategyName | None,
        trace: Trace,
    ) -> list[ScoredChunk]:
        with trace.stage(StageName.RETRIEVE_DENSE) as detail:
            hits = await self._guarded(
                lambda: self.store.search(
                    query_vector, top_k=settings.retrieve_top_k, lang=lang, strategy=strategy
                ),
                budget_ms=self._dense_timeout,
                arm="dense",
                trace=trace,
                detail=detail,
            )
            detail["candidates"] = len(hits)
            return hits

    async def _lexical_arm(
        self, query: str, lang: Lang | None, trace: Trace
    ) -> list[ScoredChunk]:
        if not self.enable_lexical:
            trace.skip(StageName.RETRIEVE_LEXICAL, "disabled")
            return []
        with trace.stage(StageName.RETRIEVE_LEXICAL) as detail:
            hits = await self._guarded(
                # LexicalIndex.search is sync and CPU-bound (a pass over the score matrix);
                # running it inline would block the loop and stall the dense arm it is
                # supposed to be overlapping with.
                lambda: asyncio.to_thread(
                    self.lexical.search, query, top_k=settings.lexical_top_k, lang=lang
                ),
                budget_ms=self._lexical_timeout,
                arm="lexical",
                trace=trace,
                detail=detail,
            )
            detail["candidates"] = len(hits)
            return hits

    async def _guarded(
        self,
        call: Callable[[], Awaitable[list[ScoredChunk]]],
        *,
        budget_ms: float,
        arm: str,
        trace: Trace,
        detail: dict[str, Any],
    ) -> list[ScoredChunk]:
        """Run one arm under a deadline, retrying only while the deadline still allows it.

        The retry exists for the dense arm's socket: a refused connection or a reset fails in
        about a millisecond, and a second attempt inside the same budget often lands. There is
        deliberately no backoff sleep -- `retry_backoff_ms` is 40ms against an 80ms arm budget,
        so sleeping would guarantee the retry misses its own deadline. Stage-level backoff is
        the harness's job; this is a sub-100ms in-flight recovery.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + budget_ms / 1000.0
        attempts = max(1, settings.max_retries + 1)
        cause = "deadline exhausted"
        for attempt in range(attempts):
            remaining = deadline - loop.time()
            if remaining <= 0.0:
                break
            try:
                return await asyncio.wait_for(call(), timeout=remaining)
            except TimeoutError:
                # wait_for cancels the await, not the worker thread behind to_thread; the
                # orphan finishes into nothing, which is the right trade against the budget.
                cause = f"timeout after {budget_ms:.0f}ms"
                break  # the budget is spent by definition, so retrying cannot help
            except Exception as exc:
                cause = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < attempts:
                    trace.retries += 1
                    detail["retried"] = detail.get("retried", 0) + 1
        note = f"{arm} retrieval failed ({cause}); serving from the other arm"
        _LOG.warning("%s", note)
        trace.degrade(note)
        detail["failed"] = True
        detail["error"] = cause
        return []


_INSTANCE: HybridRetriever | None = None
_INSTANCE_LOCK = threading.Lock()


def get_hybrid_retriever() -> HybridRetriever:
    """Process-wide retriever, so the vector store and BM25 index are opened once."""
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = HybridRetriever()
    return _INSTANCE


__all__ = [
    "HybridRetriever",
    "get_hybrid_retriever",
    "normalized_confidence",
    "reciprocal_rank_fusion",
]
