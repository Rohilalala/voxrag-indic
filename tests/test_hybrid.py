"""Reciprocal Rank Fusion and the two-arm retriever.

RRF is where rank arithmetic goes wrong, and every way it goes wrong is silent. A 0-based
rank makes the top hit score 1/k instead of 1/(k+1): nothing crashes, the ordering inside a
single arm is even preserved, but the fused score no longer matches the scale
`normalized_confidence` divides by, so the no-relevant-context guardrail starts refusing
good answers. Likewise, running the two arms serially costs 50ms of the 200ms budget and no
functional assertion would ever notice.

So the tests below pin the arithmetic to exact expected values derived from the documented
formula, and time the concurrency directly.
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from voxrag.config import settings
from voxrag.contracts import (
    Chunk,
    ChunkStrategyName,
    Lang,
    LexicalIndex,
    ScoredChunk,
    StageName,
    VectorStore,
)
from voxrag.store.hybrid import HybridRetriever, normalized_confidence, reciprocal_rank_fusion
from voxrag.telemetry import Trace

K = settings.rrf_k
W_DENSE = settings.dense_weight
W_LEXICAL = 1.0 - W_DENSE
DIM = settings.embed_dim


def _chunk(chunk_id: str, lang: Lang = Lang.HIN) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=f"passage {chunk_id}",
        lang=lang,
        strategy=ChunkStrategyName.PASSAGE_NATIVE,
    )


def _dense_hits(*chunk_ids: str) -> list[ScoredChunk]:
    """A dense result list annotated the way a real VectorStore annotates one."""
    return [
        ScoredChunk(
            chunk=_chunk(cid),
            score=1.0 - 0.05 * i,
            dense_score=1.0 - 0.05 * i,
            dense_rank=i + 1,
        )
        for i, cid in enumerate(chunk_ids)
    ]


def _lexical_hits(*chunk_ids: str) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=_chunk(cid),
            score=9.0 - i,
            lexical_score=9.0 - i,
            lexical_rank=i + 1,
        )
        for i, cid in enumerate(chunk_ids)
    ]


def _by_id(fused: list[ScoredChunk]) -> dict[str, ScoredChunk]:
    return {hit.chunk.chunk_id: hit for hit in fused}


# ======================================================================================
# Rank arithmetic
# ======================================================================================


def test_ranks_are_one_based_when_the_retriever_annotates_them() -> None:
    """rank 1 must contribute w/(k+1). w/k would be the 0-based bug: it never raises."""
    fused = reciprocal_rank_fusion(_dense_hits("a"), [])
    assert fused[0].score == pytest.approx(W_DENSE / (K + 1))
    assert fused[0].score != pytest.approx(W_DENSE / K)


def test_ranks_are_one_based_when_they_fall_back_to_list_position() -> None:
    """A store that forgets to annotate must still fuse as rank 1, 2, 3 -- not 0, 1, 2."""
    bare = [ScoredChunk(chunk=_chunk("a"), score=0.9), ScoredChunk(chunk=_chunk("b"), score=0.8)]
    fused = reciprocal_rank_fusion(bare, [])
    scores = _by_id(fused)
    assert scores["a"].score == pytest.approx(W_DENSE / (K + 1))
    assert scores["b"].score == pytest.approx(W_DENSE / (K + 2))
    assert scores["a"].dense_rank == 1
    assert scores["b"].dense_rank == 2


def test_reported_rank_wins_over_list_position() -> None:
    """Ranks come from the retriever when it set them: a store may hand back a list whose
    order is not its own ranking (a filtered re-slice, say) and the rank is the truth."""
    hits = [
        ScoredChunk(chunk=_chunk("second"), score=0.5, dense_score=0.5, dense_rank=7),
        ScoredChunk(chunk=_chunk("first"), score=0.9, dense_score=0.9, dense_rank=1),
    ]
    fused = reciprocal_rank_fusion(hits, [])
    assert [hit.chunk.chunk_id for hit in fused] == ["first", "second"]
    assert _by_id(fused)["second"].score == pytest.approx(W_DENSE / (K + 7))


def test_agreement_between_arms_beats_either_arm_alone() -> None:
    """The whole point of RRF: a chunk both retrievers found outranks the same chunk found
    by one, even when the single-arm sighting is at a better rank."""
    fused = reciprocal_rank_fusion(_dense_hits("solo", "both"), _lexical_hits("both"))
    scores = _by_id(fused)
    assert scores["both"].score == pytest.approx(W_DENSE / (K + 2) + W_LEXICAL / (K + 1))
    assert scores["solo"].score == pytest.approx(W_DENSE / (K + 1))
    assert scores["both"].score > scores["solo"].score
    assert [hit.chunk.chunk_id for hit in fused] == ["both", "solo"]

    # And against itself: the same chunk, same dense rank, with and without the lexical arm.
    single = reciprocal_rank_fusion(_dense_hits("both"), [])
    assert scores["both"].score > _by_id(single)["both"].score


def test_dense_weight_shifts_the_ordering() -> None:
    """Two chunks that swap ranks between the arms: the weight alone decides the winner."""
    dense = _dense_hits("dense_favourite", "x", "y", "z", "lexical_favourite")
    lexical = _lexical_hits("lexical_favourite", "x", "y", "z", "dense_favourite")

    dense_heavy = reciprocal_rank_fusion(dense, lexical, dense_weight=0.9)
    lexical_heavy = reciprocal_rank_fusion(dense, lexical, dense_weight=0.1)

    assert dense_heavy[0].chunk.chunk_id == "dense_favourite"
    assert lexical_heavy[0].chunk.chunk_id == "lexical_favourite"
    # Same candidate set either way -- only the order moved.
    assert {h.chunk.chunk_id for h in dense_heavy} == {h.chunk.chunk_id for h in lexical_heavy}


def test_fusion_reads_its_knobs_at_call_time_not_at_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """`k` and `dense_weight` used to be default ARGUMENTS reading settings, which Python
    evaluates once, at import. Every runtime override was discarded, and the tuning sweep
    consequently scored dense_weight 0.35 and 1.0 identically to four decimals and reported
    that fusion weighting does not matter. Anything that reads a tunable must read it per
    call, or the sweep that tunes it is measuring nothing."""
    dense = _dense_hits("dense_favourite", "x", "lexical_favourite")
    lexical = _lexical_hits("lexical_favourite", "x", "dense_favourite")

    monkeypatch.setattr(settings, "dense_weight", 0.9)
    assert reciprocal_rank_fusion(dense, lexical)[0].chunk.chunk_id == "dense_favourite"
    monkeypatch.setattr(settings, "dense_weight", 0.1)
    assert reciprocal_rank_fusion(dense, lexical)[0].chunk.chunk_id == "lexical_favourite"

    single = _dense_hits("only")
    monkeypatch.setattr(settings, "rrf_k", 10)
    monkeypatch.setattr(settings, "dense_weight", 1.0)
    assert normalized_confidence(reciprocal_rank_fusion(single, [])) == pytest.approx(1.0)
    assert reciprocal_rank_fusion(single, [])[0].score == pytest.approx(1.0 / 11)


def test_dense_weight_is_clamped_to_zero_one() -> None:
    fused = reciprocal_rank_fusion(_dense_hits("a"), _lexical_hits("b"), dense_weight=5.0)
    scores = _by_id(fused)
    assert scores["a"].score == pytest.approx(1.0 / (K + 1))
    assert scores["b"].score == pytest.approx(0.0)


def test_k_damps_the_head_of_the_curve() -> None:
    """The constant exists so a single arm's top hit is not unbeatable: cross-arm agreement
    at a worse rank must still be able to overturn it.

    Asserted as that property rather than as a percentage. The gap between rank 1 and rank 2
    is set by k -- 1.6% at the paper's k=60, 9.1% at the k=10 this corpus measured -- so a
    hard-coded percentage tests which constant is configured, not what the constant is for.
    """
    fused = reciprocal_rank_fusion(_dense_hits("first", "second"), [])
    scores = _by_id(fused)
    ratio = scores["first"].score / scores["second"].score
    assert ratio == pytest.approx((K + 2) / (K + 1))

    single_arm_top = reciprocal_rank_fusion(_dense_hits("dense_only"), [])[0].score
    agreed_second = _by_id(
        reciprocal_rank_fusion(_dense_hits("x", "both"), _lexical_hits("y", "both"))
    )["both"].score
    assert agreed_second > single_arm_top, "agreement must be able to beat a single arm's best"


def test_every_raw_score_and_rank_survives_fusion() -> None:
    """The trace has to explain *why* a chunk won, which needs both arms' numbers."""
    dense = _dense_hits("a", "both")
    lexical = _lexical_hits("both", "c")
    fused = _by_id(reciprocal_rank_fusion(dense, lexical))

    both = fused["both"]
    assert both.dense_rank == 2 and both.lexical_rank == 1
    assert both.dense_score == pytest.approx(dense[1].dense_score)
    assert both.lexical_score == pytest.approx(lexical[0].lexical_score)
    assert both.chunk is dense[1].chunk  # the payload is carried, not rebuilt

    # Single-arm hits keep the other arm's fields empty rather than inventing a zero.
    assert fused["a"].lexical_rank is None and fused["a"].lexical_score is None
    assert fused["c"].dense_rank is None and fused["c"].dense_score is None


def test_raw_score_falls_back_to_score_for_unannotated_stores() -> None:
    bare_dense = [ScoredChunk(chunk=_chunk("a"), score=0.77)]
    bare_lexical = [ScoredChunk(chunk=_chunk("a"), score=4.2)]
    fused = reciprocal_rank_fusion(bare_dense, bare_lexical)[0]
    assert fused.dense_score == pytest.approx(0.77)
    assert fused.lexical_score == pytest.approx(4.2)


def test_a_duplicate_inside_one_arm_is_not_paid_twice() -> None:
    dupe = _dense_hits("a", "a", "b")
    fused = reciprocal_rank_fusion(dupe, [])
    assert len(fused) == 2
    scores = _by_id(fused)
    assert scores["a"].score == pytest.approx(W_DENSE / (K + 1))  # not 2x, not rank 2 as well
    assert scores["b"].score == pytest.approx(W_DENSE / (K + 3))  # position is not renumbered


def test_fusing_with_one_empty_arm_returns_the_other_arm_unchanged() -> None:
    dense = _dense_hits("a", "b", "c")
    lexical = _lexical_hits("x", "y")

    dense_only = reciprocal_rank_fusion(dense, [])
    assert [h.chunk.chunk_id for h in dense_only] == ["a", "b", "c"]
    assert [h.dense_rank for h in dense_only] == [1, 2, 3]

    lexical_only = reciprocal_rank_fusion([], lexical)
    assert [h.chunk.chunk_id for h in lexical_only] == ["x", "y"]
    assert [h.lexical_rank for h in lexical_only] == [1, 2]
    assert lexical_only[0].score == pytest.approx(W_LEXICAL / (K + 1))


def test_both_arms_empty_returns_empty_rather_than_raising() -> None:
    assert reciprocal_rank_fusion([], []) == []
    assert normalized_confidence([]) == 0.0


def test_output_is_sorted_best_first() -> None:
    fused = reciprocal_rank_fusion(_dense_hits("a", "b", "c", "d"), _lexical_hits("d", "c"))
    assert [h.score for h in fused] == sorted((h.score for h in fused), reverse=True)


# ======================================================================================
# Confidence
# ======================================================================================


def test_normalized_confidence_matches_its_documented_scale() -> None:
    """The scale is what `settings.min_retrieval_score` is written against, so the anchor
    points are load-bearing, not cosmetic."""
    both = reciprocal_rank_fusion(_dense_hits("a"), _lexical_hits("a"))
    assert normalized_confidence(both) == pytest.approx(1.0)

    dense_only = reciprocal_rank_fusion(_dense_hits("a"), [])
    assert normalized_confidence(dense_only) == pytest.approx(W_DENSE)

    lexical_only = reciprocal_rank_fusion([], _lexical_hits("a"))
    assert normalized_confidence(lexical_only) == pytest.approx(W_LEXICAL)

    # A lone lexical hit from deep in the tail must fall under the refusal threshold.
    tail = reciprocal_rank_fusion([], _lexical_hits(*(f"t{i}" for i in range(20))))
    assert normalized_confidence(tail[-1:]) < settings.min_retrieval_score


def test_normalized_confidence_is_bounded_and_monotone_in_the_top_score() -> None:
    def conf(score: float) -> float:
        return normalized_confidence([ScoredChunk(chunk=_chunk("a"), score=score)])

    raw = [-1.0, 0.0, 0.001, 0.005, 0.010, 1.0 / (K + 1), 5.0]
    confidences = [conf(s) for s in raw]
    assert all(0.0 <= c <= 1.0 for c in confidences)
    assert confidences == sorted(confidences)
    assert confidences[0] == 0.0  # a negative score cannot become negative confidence
    assert confidences[-1] == 1.0  # and an over-max score is clamped, not >1


def test_normalized_confidence_reads_the_best_hit_not_the_first() -> None:
    unsorted = [
        ScoredChunk(chunk=_chunk("a"), score=0.001),
        ScoredChunk(chunk=_chunk("b"), score=1.0 / (K + 1)),
    ]
    assert normalized_confidence(unsorted) == pytest.approx(1.0)


# ======================================================================================
# HybridRetriever
# ======================================================================================


class _FakeStore:
    """VectorStore that sleeps like a network hop. Satisfies contracts.VectorStore."""

    name = "fake-dense"

    def __init__(
        self, hits: list[ScoredChunk], *, delay_s: float = 0.0, error: Exception | None = None
    ) -> None:
        self.hits = hits
        self.delay_s = delay_s
        self.error = error
        self.calls = 0
        self.last_kwargs: dict[str, object] = {}

    async def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError

    async def search(
        self,
        vector: np.ndarray,
        *,
        top_k: int = 20,
        lang: Lang | None = None,
        strategy: ChunkStrategyName | None = None,
    ) -> list[ScoredChunk]:
        self.calls += 1
        self.last_kwargs = {"top_k": top_k, "lang": lang, "strategy": strategy}
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        return list(self.hits)

    async def count(self) -> int:
        return len(self.hits)

    async def health(self) -> bool:
        return True


class _FakeLexical:
    """LexicalIndex whose `search` is synchronous and CPU-bound, like the real one."""

    name = "fake-lexical"

    def __init__(
        self, hits: list[ScoredChunk], *, delay_s: float = 0.0, error: Exception | None = None
    ) -> None:
        self.hits = hits
        self.delay_s = delay_s
        self.error = error
        self.calls = 0
        self.thread_names: list[str] = []

    def index(self, chunks: list[Chunk]) -> None:  # pragma: no cover
        raise NotImplementedError

    def search(
        self, query: str, *, top_k: int = 20, lang: Lang | None = None
    ) -> list[ScoredChunk]:
        self.calls += 1
        self.thread_names.append(threading.current_thread().name)
        if self.delay_s:
            time.sleep(self.delay_s)  # blocking on purpose: the real index is a matrix pass
        if self.error is not None:
            raise self.error
        return list(self.hits)


def test_fakes_satisfy_the_protocols() -> None:
    assert isinstance(_FakeStore([]), VectorStore)
    assert isinstance(_FakeLexical([]), LexicalIndex)


async def _warm_thread_pool() -> None:
    """asyncio's default executor spawns its first worker lazily; that one-off cost would
    otherwise land inside the concurrency measurement below."""
    await asyncio.to_thread(lambda: None)


async def test_the_two_arms_run_concurrently() -> None:
    """Serialising the arms adds the cheap arm's latency to every request for no reason.
    Nothing functional would catch it, so it is measured."""
    await _warm_thread_pool()
    arm_delay = 0.05
    store = _FakeStore(_dense_hits("dense-only"), delay_s=arm_delay)
    lexical = _FakeLexical(_lexical_hits("lexical-only"), delay_s=arm_delay)
    retriever = HybridRetriever(store, lexical, enable_lexical=True)
    trace = Trace(request_id="concurrency")

    start = time.perf_counter()
    fused = await retriever.retrieve("क्वेरी", np.zeros(DIM, dtype=np.float32), trace=trace)
    elapsed = time.perf_counter() - start

    # Both arms really ran and really finished -- otherwise a timed-out arm would make a
    # serialisation regression look like a pass.
    assert {hit.chunk.chunk_id for hit in fused} == {"dense-only", "lexical-only"}
    assert not trace.degradations
    # 1.5x the arm delay, i.e. 75ms against a 100ms serial floor. Measured over 20 runs the
    # concurrent path takes 50.8-55.6ms, so this leaves ~20ms of slack for a loaded CI box
    # while still being 25ms clear of the cheapest possible serialised implementation.
    assert elapsed < arm_delay * 1.5, f"arms look serialised: {elapsed * 1000:.0f}ms"
    assert elapsed >= arm_delay  # sanity: the sleeps were not skipped


async def test_lexical_arm_does_not_block_the_event_loop() -> None:
    """The lexical index is sync; running it inline would stall the dense arm it is meant to
    overlap with. It must be handed to a worker thread."""
    lexical = _FakeLexical(_lexical_hits("a"), delay_s=0.01)
    retriever = HybridRetriever(_FakeStore([]), lexical, enable_lexical=True)
    await retriever.retrieve("q", np.zeros(DIM, dtype=np.float32), trace=Trace(request_id="r"))
    assert lexical.thread_names and threading.current_thread().name not in lexical.thread_names


async def test_retrieve_passes_configured_top_k_to_the_dense_arm() -> None:
    store = _FakeStore(_dense_hits("a"))
    retriever = HybridRetriever(store, _FakeLexical([]), enable_lexical=True)
    await retriever.retrieve(
        "q",
        np.zeros(DIM, dtype=np.float32),
        trace=Trace(request_id="r"),
        lang=Lang.TAM,
        strategy=ChunkStrategyName.PARENT_CHILD,
    )
    assert store.last_kwargs == {
        "top_k": settings.retrieve_top_k,
        "lang": Lang.TAM,
        "strategy": ChunkStrategyName.PARENT_CHILD,
    }


async def test_retrieve_truncates_to_the_requested_limit() -> None:
    store = _FakeStore(_dense_hits(*(f"d{i}" for i in range(10))))
    retriever = HybridRetriever(store, _FakeLexical([]), enable_lexical=True)
    trace = Trace(request_id="r")
    fused = await retriever.retrieve(
        "q", np.zeros(DIM, dtype=np.float32), trace=trace, top_k=3
    )
    assert [hit.chunk.chunk_id for hit in fused] == ["d0", "d1", "d2"]
    fuse_detail = next(t for t in trace.timings if t.stage is StageName.FUSE).detail
    assert fuse_detail["fused"] == 10 and fuse_detail["returned"] == 3


async def test_a_dead_dense_arm_degrades_to_the_lexical_one() -> None:
    """One dead dependency degrades result quality; it never fails the request."""
    store = _FakeStore([], error=ConnectionRefusedError("qdrant is down"))
    lexical = _FakeLexical(_lexical_hits("survivor"))
    retriever = HybridRetriever(store, lexical, enable_lexical=True)
    trace = Trace(request_id="degrade")

    fused = await retriever.retrieve("q", np.zeros(DIM, dtype=np.float32), trace=trace)

    assert [hit.chunk.chunk_id for hit in fused] == ["survivor"]
    assert trace.degradations and "dense" in trace.degradations[0]
    dense_detail = next(t for t in trace.timings if t.stage is StageName.RETRIEVE_DENSE).detail
    assert dense_detail["failed"] is True
    assert "ConnectionRefusedError" in dense_detail["error"]
    # Retried inside its own budget rather than failing on the first refused socket.
    assert store.calls == settings.max_retries + 1
    assert trace.retries == settings.max_retries


async def test_a_dead_lexical_arm_degrades_to_the_dense_one() -> None:
    store = _FakeStore(_dense_hits("survivor"))
    lexical = _FakeLexical([], error=RuntimeError("index corrupt"))
    retriever = HybridRetriever(store, lexical, enable_lexical=True)
    trace = Trace(request_id="degrade")

    fused = await retriever.retrieve("q", np.zeros(DIM, dtype=np.float32), trace=trace)

    assert [hit.chunk.chunk_id for hit in fused] == ["survivor"]
    assert trace.degradations and "lexical" in trace.degradations[0]


async def test_both_arms_dead_returns_empty_instead_of_erroring() -> None:
    """An empty candidate list is what the guardrail turns into an honest refusal; a raise
    here would be a 500."""
    retriever = HybridRetriever(
        _FakeStore([], error=OSError("no route to host")),
        _FakeLexical([], error=RuntimeError("boom")),
        enable_lexical=True,
    )
    trace = Trace(request_id="both-dead")
    fused = await retriever.retrieve("q", np.zeros(DIM, dtype=np.float32), trace=trace)
    assert fused == []
    assert len(trace.degradations) == 2
    assert normalized_confidence(fused) == 0.0


async def test_a_slow_arm_is_cut_off_at_its_budget() -> None:
    """The arm timeout is what keeps a hung dependency out of the 200ms budget."""
    budget_s = settings.stage_timeout_ms[StageName.RETRIEVE_DENSE.value] / 1000.0
    store = _FakeStore(_dense_hits("too-late"), delay_s=budget_s * 4)
    retriever = HybridRetriever(store, _FakeLexical(_lexical_hits("fast")), enable_lexical=True)
    trace = Trace(request_id="slow")

    start = time.perf_counter()
    fused = await retriever.retrieve("q", np.zeros(DIM, dtype=np.float32), trace=trace)
    elapsed = time.perf_counter() - start

    assert [hit.chunk.chunk_id for hit in fused] == ["fast"]
    assert elapsed < budget_s * 3
    assert "timeout" in trace.degradations[0]


async def test_disabled_lexical_arm_is_skipped_not_failed() -> None:
    lexical = _FakeLexical(_lexical_hits("never"))
    retriever = HybridRetriever(_FakeStore(_dense_hits("a")), lexical, enable_lexical=False)
    trace = Trace(request_id="no-lexical")

    fused = await retriever.retrieve("q", np.zeros(DIM, dtype=np.float32), trace=trace)

    assert [hit.chunk.chunk_id for hit in fused] == ["a"]
    assert lexical.calls == 0
    assert not trace.degradations  # disabled is a configuration choice, not a failure
    lexical_stage = next(t for t in trace.timings if t.stage is StageName.RETRIEVE_LEXICAL)
    assert lexical_stage.skipped is True
    assert lexical_stage.detail == {"why": "disabled"}


async def test_fuse_stage_reports_the_overlap_diagnostic() -> None:
    """Overlap near zero means the arms disagree completely and the weights are doing all
    the work -- it is the one number that says whether hybrid retrieval is earning its keep."""
    store = _FakeStore(_dense_hits("shared", "dense-only"))
    lexical = _FakeLexical(_lexical_hits("shared", "lexical-only"))
    retriever = HybridRetriever(store, lexical, enable_lexical=True)
    trace = Trace(request_id="overlap")

    await retriever.retrieve("q", np.zeros(DIM, dtype=np.float32), trace=trace)

    detail = next(t for t in trace.timings if t.stage is StageName.FUSE).detail
    assert detail["dense"] == 2 and detail["lexical"] == 2
    assert detail["fused"] == 3 and detail["overlap"] == 1
    assert 0.0 <= detail["confidence"] <= 1.0
    assert detail["rrf_k"] == K and detail["dense_weight"] == W_DENSE


async def test_fuse_trace_describes_the_arithmetic_that_actually_ran() -> None:
    """The trace is the audit record, so the parameters it reports must be the parameters
    that produced the scores next to them.

    These are two independent reads today: `reciprocal_rank_fusion` takes `k`/`dense_weight`
    as *default arguments*, which Python evaluates once at import, while the FUSE detail
    reads `settings` live at call time (store/hybrid.py:48-49 against :204-205). They agree
    as long as nothing mutates `settings` after import -- which nothing in the service does --
    but if that ever stops being true the trace would quietly describe arithmetic that did
    not happen. One dense hit at rank 1 makes the reported parameters reproduce the score
    exactly, so a divergence fails here instead of silently mis-explaining a ranking.
    """
    store = _FakeStore(_dense_hits("only"))
    retriever = HybridRetriever(store, _FakeLexical([]), enable_lexical=True)
    trace = Trace(request_id="trace-honesty")

    fused = await retriever.retrieve("q", np.zeros(DIM, dtype=np.float32), trace=trace)

    detail = next(t for t in trace.timings if t.stage is StageName.FUSE).detail
    reported = detail["dense_weight"] / (detail["rrf_k"] + 1)
    assert fused[0].score == pytest.approx(reported)
    # And the confidence it reports is that same score on the scale it claims to use.
    assert detail["confidence"] == pytest.approx(
        fused[0].score / (1.0 / (detail["rrf_k"] + 1)), abs=1e-4
    )


async def test_warmup_tolerates_backends_that_cannot_warm() -> None:
    """Warmup is best-effort: a cold cache is not worth failing startup over."""
    store = _FakeStore([])
    lexical = _FakeLexical([])
    retriever = HybridRetriever(store, lexical, enable_lexical=True)
    await retriever.warmup()  # neither fake defines warmup(); this must simply do nothing
    assert store.calls == 0 and lexical.calls == 0
