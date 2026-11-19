"""The harness: one request, ten stages, every one of them timed, bounded and recoverable.

This is deliberately not a prompt-in/text-out call. A voice question walks through:

    stt -> input rails -> embed -> dense + lexical retrieval -> fuse -> post-retrieval
    rails -> optional rerank -> generation (via the validated tool layer) -> output rails

and each step is (a) wrapped in `telemetry.Trace.stage` so it appears in the P50/P70/P100
report, (b) bounded by `settings.stage_timeout_ms` so one hung dependency costs its own
slice of the budget and nothing more, and (c) given a degraded path so it can fail without
failing the request:

    no embedder     -> lexical-only retrieval
    no lexical index-> dense-only retrieval
    no candidates   -> refuse with NO_RELEVANT_CONTEXT
    no generator    -> extractive answer from the top passage
    quality fails   -> fast mode
    ungrounded      -> the answer is replaced by a refusal

Two rules the shape of this file follows from:

**A refusal is a result.** Every early exit still runs `trace.finish()` and still lands in
`telemetry.TRACES`. Dropping fast refusals from the latency ring would flatter every
percentile in the report by removing its cheapest requests, which is the sort of number
that falls apart the moment anyone asks how it was measured.

**The trace shape is constant.** Stages that did not run are recorded as skipped rather
than omitted, so `report()` compares like with like across requests and a missing stage in
the waterfall means something actually went wrong.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final
from uuid import uuid4

from voxrag.config import settings
from voxrag.contracts import (
    Answer,
    GenerationMode,
    GuardrailVerdict,
    Lang,
    PipelineResult,
    RefusalReason,
    ScoredChunk,
    StageName,
    Transcript,
)
from voxrag.harness.errors import StageTimeout, VoxRAGError
from voxrag.harness.retry import stage_budget_ms, with_retry, with_timeout
from voxrag.harness.tools import RetrievalResult, call_tool
from voxrag.telemetry import TRACES, Trace

_LOG = logging.getLogger(__name__)

#: Progress callback. Called as `on_event(kind, name, value)`; may be sync or async.
#: Kinds emitted: "stage" (name, ms) | "transcript" (text, lang) | "guardrail" (rail, allowed)
#: | "retrieval" (n_chunks, confidence) | "token" (text, None) | "answer" (text, mode)
#: | "error" (message, None). "token" carries one generated segment; the non-streaming
#: generators this system ships produce exactly one segment per answer.
EventCallback = Callable[[str, Any, Any], Awaitable[None] | None]

# --------------------------------------------------------------------------------------
# Module-level constants (not in config.py -- they describe this file's own mechanics)
# --------------------------------------------------------------------------------------

DEFAULT_MIME: Final[str] = "audio/wav"

#: Outer ceiling on the whole STT stage, fallback hops included. Generous because the
#: keyless local backend downloads Whisper weights on its very first call and cutting that
#: short would mean the fallback can never become usable; every later call is seconds. STT
#: sits outside the 200ms CORE budget by definition (see telemetry.Trace.finish).
STT_CEILING_MS: Final[float] = 210_000.0

#: Warmup probe budget for the STT vendors -- a network health check, not a transcription.
WARMUP_STT_BUDGET_S: Final[float] = 10.0

#: The warmup query. Real Devanagari, so it exercises the multilingual tokeniser, the
#: script-based language detector and the BM25 Indic path rather than an ASCII fast path.
WARMUP_QUERY: Final[str] = "भारत की राजधानी क्या है"

#: Canonical stage order. `_seal` records anything unvisited as skipped.
_STAGE_ORDER: Final[tuple[StageName, ...]] = (
    StageName.STT,
    StageName.INPUT_GUARDRAIL,
    StageName.QUERY_EMBED,
    StageName.RETRIEVE_DENSE,
    StageName.RETRIEVE_LEXICAL,
    StageName.FUSE,
    StageName.RERANK,
    StageName.GENERATE,
    StageName.OUTPUT_GUARDRAIL,
)

# --------------------------------------------------------------------------------------
# Guardrails: the real module if it is importable, the built-ins below if it is not
# --------------------------------------------------------------------------------------

_GUARDRAILS: Any | None = None
_GUARDRAILS_RESOLVED = False
_GUARDRAILS_LOCK = threading.Lock()


def _guardrails() -> Any | None:
    """`voxrag.guardrails`, or None if it cannot be imported.

    Resolved once, negative result included: importing it compiles a few hundred lexicon
    patterns, and rediscovering per request that it is absent would cost an import attempt
    inside the 200ms budget.
    """
    global _GUARDRAILS, _GUARDRAILS_RESOLVED
    if _GUARDRAILS_RESOLVED:
        return _GUARDRAILS
    with _GUARDRAILS_LOCK:
        if not _GUARDRAILS_RESOLVED:
            try:
                _GUARDRAILS = importlib.import_module("voxrag.guardrails")
            except Exception as exc:
                _LOG.error(
                    "voxrag.guardrails unavailable (%r); falling back to the harness's own "
                    "minimal rails -- policy checks (safety, injection, off-topic) are OFF",
                    exc,
                )
                _GUARDRAILS = None
            _GUARDRAILS_RESOLVED = True
    return _GUARDRAILS


def reset_rail_cache() -> None:
    """Forget the resolved guardrail module. For tests that patch or reload it."""
    global _GUARDRAILS, _GUARDRAILS_RESOLVED
    with _GUARDRAILS_LOCK:
        _GUARDRAILS = None
        _GUARDRAILS_RESOLVED = False


def detect_lang(text: str, hint: Lang | None = None) -> Lang | None:
    """Script-based language identification, delegated to `voxrag.guardrails`.

    The detector lives with the rails because `LanguageRail` is its main consumer, and one
    implementation is the point: two script tables that disagree would let the harness
    retrieve in a language the rail just refused.
    """
    module = _guardrails()
    if module is None:
        return hint
    try:
        return module.detect_lang(text, hint=hint).lang or hint
    except Exception as exc:  # never let detection fail a request
        _LOG.debug("language detection failed (%r); using the hint", exc)
        return hint


def _coerce_lang(value: Any) -> Lang | None:
    """Accept a Lang, an ISO code, a 3-letter stem or a BCP-47 tag. Never raises: a bad hint
    from a caller should cost detection accuracy, not the request."""
    if value is None or isinstance(value, Lang):
        return value
    raw = str(value).strip().lower()
    if not raw:
        return None
    for candidate in (raw, raw.split("-")[0], raw.split("_")[0]):
        try:
            return Lang(candidate)
        except ValueError:
            continue
    from voxrag.contracts import LANG_FILE_STEM  # local: only needed on this slow path

    for lang, stem in LANG_FILE_STEM.items():
        if stem == raw:
            return lang
    _LOG.debug("unrecognised language hint %r; falling back to detection", value)
    return None


def _coerce_mode(value: Any) -> GenerationMode:
    if isinstance(value, GenerationMode):
        return value
    if value is None:
        return settings.default_mode
    try:
        return GenerationMode(str(value).strip().lower())
    except ValueError:
        _LOG.debug("unrecognised generation mode %r; using %s", value, settings.default_mode)
        return settings.default_mode


# --------------------------------------------------------------------------------------
# Built-in rails: what runs when `voxrag.guardrails` cannot be imported
# --------------------------------------------------------------------------------------
#
# These are structural, not policy: empty input, over-long input, no retrieved context, and
# lexical groundedness. They exist so the harness is still *correct* without the guardrail
# module -- it will not answer from nothing and it will not ship an answer that shares no
# vocabulary with its sources -- while being honest that safety, injection and off-topic
# detection are gone. Every one of them is superseded by the real rail of the same purpose.


def _builtin_input_verdicts(query: str) -> list[GuardrailVerdict]:
    """Empty-transcript and length checks. Superseded by `guardrails.EmptyQueryRail`."""
    stripped = query.strip()
    if len(stripped) < settings.min_query_chars:
        return [
            GuardrailVerdict(
                name="harness.empty_query",
                allowed=False,
                reason=RefusalReason.EMPTY_TRANSCRIPT,
                score=1.0,
                detail=f"{len(stripped)} characters, minimum is {settings.min_query_chars}",
            )
        ]
    if len(stripped) > settings.max_query_chars:
        return [
            GuardrailVerdict(
                name="harness.query_length",
                allowed=True,
                score=1.0,
                detail=f"truncated from {len(stripped)} to {settings.max_query_chars} characters",
                rewritten_text=stripped[: settings.max_query_chars],
            )
        ]
    return []


def _tokens(text: str, lang: Lang | None) -> set[str]:
    """Script-aware token set, reusing the retrieval tokeniser so the groundedness score is
    measured in the same units the lexical index scores in."""
    try:
        from voxrag.store.bm25 import tokenize

        return set(tokenize(text, lang))
    except Exception:  # bm25s/PyStemmer missing -- a crude split still measures overlap
        return {piece for piece in text.lower().split() if piece}


def _builtin_output_verdicts(
    answer: Answer, context: Sequence[ScoredChunk], lang: Lang | None
) -> list[GuardrailVerdict]:
    """Groundedness by lexical overlap. Superseded by `guardrails.GroundednessRail`.

    Overlap is a weak proxy for entailment and is stated as such, but it catches the failure
    that actually matters here -- a model answering from its own weights, whose wording then
    shares almost nothing with the retrieved text.
    """
    if answer.refused:
        return []
    answer_tokens = _tokens(answer.text, lang)
    if not answer_tokens:
        return []
    context_tokens: set[str] = set()
    for scored in context:
        context_tokens |= _tokens(scored.chunk.generation_text, scored.chunk.lang)
    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    score = max(answer.groundedness, overlap)
    allowed = score >= settings.min_groundedness
    return [
        GuardrailVerdict(
            name="harness.groundedness",
            allowed=allowed,
            reason=None if allowed else RefusalReason.UNGROUNDED_ANSWER,
            score=round(score, 4),
            detail=(
                f"{overlap:.0%} of answer tokens appear in the cited passages "
                f"(threshold {settings.min_groundedness:.0%})"
            ),
        )
    ]


def _builtin_retrieval_verdicts(result: RetrievalResult) -> list[GuardrailVerdict]:
    """No-relevant-context check. Superseded by `guardrails.OffTopicRail`.

    Without candidates there is nothing to ground an answer in, and answering anyway is the
    definition of hallucinating -- so this one has to exist even in the degraded mode.
    """
    if not result.chunks:
        return [
            GuardrailVerdict(
                name="harness.no_context",
                allowed=False,
                reason=RefusalReason.NO_RELEVANT_CONTEXT,
                score=1.0,
                detail="retrieval returned no candidates",
            )
        ]
    allowed = result.confidence >= settings.min_retrieval_score
    return [
        GuardrailVerdict(
            name="harness.retrieval_confidence",
            allowed=allowed,
            reason=None if allowed else RefusalReason.NO_RELEVANT_CONTEXT,
            score=round(result.confidence, 4),
            detail=(
                f"best fused hit scores {result.confidence:.2f}, "
                f"threshold {settings.min_retrieval_score:.2f}"
            ),
        )
    ]


#: Head-room between the cross-language rail's own ceiling and the ceiling `run_rails` puts on
#: the set that contains it. It exists because the two policies disagree on purpose: the rail
#: refuses when its verifier runs out of time, `run_rails` ALLOWS when a rail does, so whichever
#: fires first decides whether an unverifiable answer is served. The rail has to win.
#:
#: 25ms -- the input_guardrail stage budget, tried first -- was measured to be too little. With
#: the set ceiling at rail+25ms, a first QUALITY query hit a cold model load and `run_rails`
#: logged "guardrail cross_language exceeded 1525.0ms" and let the answer through. Abandoning an
#: in-flight entailment request and returning the refusal costs 1.8ms and 6.4ms in the warm case
#: (measured at 200ms and 50ms rail ceilings), so this is ~40x the warm cost of the thing it is
#: covering, and it is pure head-room: nothing waits for it unless the rail has already timed out.
_CROSS_LANGUAGE_SLACK_MS: Final = 250.0


def _post_retrieval_timeout_ms(mode: GenerationMode) -> float | None:
    """Budget for the post-retrieval rail set. None means "use the stage default".

    That default is the 25ms input-guardrail ceiling, which is right for two regex rails and
    wrong for the one rail that can call a model: it would silently convert the fail-CLOSED
    cross-language rail into an allow on every English query -- the exact unverified answer it
    exists to stop, reported as if the rail had run.

    Only the model-backed modes (QUALITY, AGENTIC) with verification on can make that call at
    all; every other path returns None, stays on the 25ms ceiling and pays nothing.
    """
    if mode is GenerationMode.FAST or not settings.cross_language_verification:
        return None
    return settings.cross_language_timeout_ms + _CROSS_LANGUAGE_SLACK_MS


# --------------------------------------------------------------------------------------
# Per-request state
# --------------------------------------------------------------------------------------


@dataclass
class _Request:
    """Everything one in-flight request carries. Passed between stages instead of ten
    positional arguments, and never shared between requests."""

    trace: Trace
    request_id: str
    mode: GenerationMode
    on_event: EventCallback | None = None
    record: bool = True
    query_text: str = ""
    lang: Lang | None = None
    transcript: Transcript | None = None
    verdicts: list[GuardrailVerdict] = field(default_factory=list)
    retrieved: list[ScoredChunk] = field(default_factory=list)
    #: How many trace timings have already been pushed to `on_event`.
    cursor: int = 0
    #: Set by `_seal`. Guards against a second `trace.finish()` if sealing itself failed and
    #: the error path sealed again -- that would append a second TOTAL/CORE pair and double
    #: this request's weight in every percentile.
    sealed: bool = False


# --------------------------------------------------------------------------------------
# The harness
# --------------------------------------------------------------------------------------


class Harness:
    """Orchestrates one request end to end. One instance per process (see `get_harness`)."""

    def __init__(self) -> None:
        self._warm = False
        self._warm_report: dict[str, Any] = {}
        # Constructed outside a running loop on purpose -- asyncio.Lock has not bound to a
        # loop at construction since 3.10, so a module-level singleton is safe.
        self._warm_lock = asyncio.Lock()

    # -- public API ----------------------------------------------------------------------

    async def run(
        self,
        *,
        audio: bytes | None = None,
        mime: str | None = None,
        text: str | None = None,
        lang: Lang | str | None = None,
        mode: GenerationMode | str | None = None,
        request_id: str | None = None,
        on_event: EventCallback | None = None,
    ) -> PipelineResult:
        """Run the full pipeline for one voice (or typed) question.

        Always returns a PipelineResult. The only exceptions that escape are genuinely
        unrecoverable ones -- a bug in this file, or a cancelled task.
        """
        return await self._run(
            audio=audio,
            mime=mime,
            text=text,
            lang=lang,
            mode=mode,
            request_id=request_id,
            on_event=on_event,
            record=True,
        )

    async def warmup(self) -> dict[str, Any]:
        """Pay every first-call cost before a user is waiting on it.

        Loading the ONNX encoder is ~1-2s of disk and arena allocation, the first Qdrant call
        is a TCP handshake plus collection metadata, and the first BM25 search builds the
        scoring matrices. Against a 200ms budget, doing any of that inside request one is the
        difference between a demo that looks fast and one that looks broken.

        Idempotent and safe to call concurrently; the dummy query is deliberately *not*
        recorded into `telemetry.TRACES` so it cannot flatter the latency percentiles.
        """
        if self._warm:
            return self._warm_report
        async with self._warm_lock:
            if self._warm:
                return self._warm_report
            report: dict[str, Any] = {}

            try:
                from voxrag.harness.tools import _embedder as resolve_embedder

                # Through the tool layer's own cache, not `voxrag.embed` directly: the
                # request path checks *that* reference and would otherwise treat the first
                # real query as a cold start and degrade it to lexical-only.
                embedder = await resolve_embedder(wait=True)
                report["embedder"] = {
                    "model": getattr(embedder, "name", "unknown"),
                    "dim": getattr(embedder, "dim", 0),
                }
            except Exception as exc:
                report["embedder"] = {"error": f"{type(exc).__name__}: {exc}"}

            try:
                from voxrag.store import get_vector_store

                store = get_vector_store()
                report["store"] = {"name": store.name, "chunks": await store.count()}
            except Exception as exc:
                report["store"] = {"error": f"{type(exc).__name__}: {exc}"}

            try:
                from voxrag.store.hybrid import get_hybrid_retriever

                await get_hybrid_retriever().warmup()
                report["retriever"] = "ready"
            except Exception as exc:
                report["retriever"] = f"{type(exc).__name__}: {exc}"

            try:
                from voxrag.stt import warmup as stt_warmup

                report["stt"] = await asyncio.wait_for(stt_warmup(), timeout=WARMUP_STT_BUDGET_S)
            except Exception as exc:
                # STT health is a network probe; failing it must not block the text path.
                report["stt"] = {"error": f"{type(exc).__name__}: {exc}"}

            try:
                import voxrag.generate as generate_module
                from voxrag.harness.tools import _generator as resolve_generator

                # Resolve through the tool layer's cache so the *request path* is what got
                # warmed, then let the module probe its own QUALITY dependency.
                generator = await resolve_generator(settings.default_mode)
                report["generator"] = {
                    "default": getattr(generator, "name", type(generator).__name__),
                    "probe": await generate_module.warmup(),
                    # The line above resolves settings.default_mode, which is FAST, so it
                    # warms the EXTRACTIVE generator; the probe only asks /api/tags whether
                    # gemma4 exists. Neither loads it. That left the cross-language verifier
                    # permanently cold, because its 1500ms ceiling cancels a 6089.7ms load
                    # and takes the load down with it -- measured: three consecutive QUALITY
                    # cross-language queries each refused at ~1.53s with `ollama ps` empty
                    # throughout. So startup loads it explicitly, where waiting is free.
                    "verifier": await generate_module.warm_verifier(),
                }
            except Exception as exc:
                report["generator"] = f"unavailable ({type(exc).__name__}); extractive fallback"

            # Importing the rails compiles a few hundred lexicon patterns; that belongs in
            # startup, not in the first request's 25ms input-guardrail budget.
            report["guardrails"] = "ready" if _guardrails() is not None else "unavailable"

            probe = await self._run(text=WARMUP_QUERY, record=False)
            report["probe"] = {
                "core_ms": round(probe.core_ms, 2),
                "total_ms": round(probe.ms(StageName.TOTAL), 2),
                "retrieved": len(probe.retrieved),
                "refused": probe.answer.refused,
            }

            self._warm = True
            self._warm_report = report
            _LOG.info("harness warm: %s", report)
            return report

    # -- pipeline ------------------------------------------------------------------------

    async def _run(
        self,
        *,
        audio: bytes | None = None,
        mime: str | None = None,
        text: str | None = None,
        lang: Lang | str | None = None,
        mode: GenerationMode | str | None = None,
        request_id: str | None = None,
        on_event: EventCallback | None = None,
        record: bool = True,
    ) -> PipelineResult:
        rid = request_id or uuid4().hex[:12]
        req = _Request(
            trace=Trace(request_id=rid),
            request_id=rid,
            mode=_coerce_mode(mode),
            on_event=on_event,
            record=record,
            lang=_coerce_lang(lang),
        )

        try:
            # 1. speech to text ------------------------------------------------------------
            if audio:
                req.transcript = await self._stt(req, audio, mime or DEFAULT_MIME)
                req.query_text = req.transcript.text.strip()
                req.lang = req.transcript.lang or req.lang
            else:
                req.trace.skip(StageName.STT, "text input")
                req.query_text = (text or "").strip()
            # Detect before the rails rather than relying on LanguageRail's verdict alone:
            # with `enable_guardrails=False` no rail runs at all, and the pipeline must still
            # know which language to retrieve and answer in.
            req.lang = detect_lang(req.query_text, req.lang)
            await self._flush(req)
            await self._emit(req, "transcript", req.query_text, req.lang.value if req.lang else None)

            # 2. input rails ---------------------------------------------------------------
            blocked = await self._input_phase(req)
            if blocked is not None:
                return await self._refuse(req, blocked)

            # 3-7. AGENTIC runs the stages below as a graph that can loop over them.
            if req.mode is GenerationMode.AGENTIC:
                try:
                    from voxrag.harness.agentic import run_graph
                except ModuleNotFoundError:  # langgraph is the optional `agent` extra
                    req.trace.degrade("agentic mode needs the `agent` extra; serving quality")
                    req.mode = GenerationMode.QUALITY
                else:
                    outcome = await run_graph(self, req)
                    if isinstance(outcome, GuardrailVerdict):
                        return await self._refuse(req, outcome)
                    await self._emit(req, "answer", outcome.text, outcome.mode.value)
                    return await self._seal(req, outcome)

            # 3. retrieval -----------------------------------------------------------------
            retrieval = await self._retrieve(req)
            req.retrieved = list(retrieval.chunks)
            await self._flush(req)
            await self._emit(req, "retrieval", len(retrieval.chunks), round(retrieval.confidence, 4))

            # 4. post-retrieval rails ------------------------------------------------------
            blocked = await self._retrieval_phase(req, retrieval)
            if blocked is not None:
                return await self._refuse(req, blocked)

            # 5. optional rerank -----------------------------------------------------------
            # READ THIS BEFORE ADDING A RERANKER. The rails above verified the FUSED order;
            # `_generate` answers from `req.retrieved[:context_top_k]`, and so does
            # `OllamaGenerator.entails`. Those are the same four chunks only while this stage
            # is a no-op -- which it is today, in both configurations: `enable_rerank` is
            # False, and with it True `_resolve_reranker()` returns None (verified: no
            # `voxrag.rerank` or `voxrag.store.rerank` module exists in this tree), so the
            # fused order is passed straight through and the trace records "no reranker
            # available". A real cross-encoder here promotes chunks out of the fused top-12
            # into the top-4, and the cross-language rail would then have said YES about
            # passages the answer is not built from -- an unverified answer whose verdict
            # detail reads "the verifier confirms the passages answer it". Whoever adds one
            # owns that: either verify the post-rerank slice, or cap generation to the chunk
            # ids that were verified.
            req.retrieved = await self._rerank(req)
            await self._flush(req)

            # 6. generation ----------------------------------------------------------------
            answer = await self._generate(req)
            await self._emit(req, "token", answer.text, None)
            await self._flush(req)

            # 7. output rails --------------------------------------------------------------
            answer = await self._output_phase(req, answer)
            await self._flush(req)
            await self._emit(req, "answer", answer.text, answer.mode.value)

            return await self._seal(req, answer)

        except asyncio.CancelledError:
            raise  # the client hung up; nothing to report to
        except VoxRAGError as exc:
            # A typed error that no stage could absorb. Still a result: the trace is real and
            # the latency it measured is real, so it belongs in the percentiles.
            _LOG.error("request %s failed: %s", rid, exc.message)
            req.trace.degrade(f"unrecovered: {exc.message}")
            await self._emit(req, "error", exc.message, None)
            return await self._refuse(
                req,
                GuardrailVerdict(
                    name="harness.failure",
                    allowed=False,
                    reason=RefusalReason.LOW_CONFIDENCE,
                    detail=exc.message,
                ),
            )
        except Exception as exc:  # a bug in this file, or a dependency doing something new
            _LOG.exception("request %s crashed", rid)
            req.trace.degrade(f"unhandled: {type(exc).__name__}: {exc}")
            await self._emit(req, "error", str(exc), None)
            return await self._refuse(
                req,
                GuardrailVerdict(
                    name="harness.failure",
                    allowed=False,
                    reason=RefusalReason.LOW_CONFIDENCE,
                    detail=f"{type(exc).__name__}: {exc}",
                ),
            )

    # -- stages --------------------------------------------------------------------------

    async def _stt(self, req: _Request, audio: bytes, mime: str) -> Transcript:
        """Transcribe. Never raises: a dead STT chain becomes an empty transcript, which the
        empty-query rail turns into a clean refusal instead of an HTTP 500."""
        from voxrag.stt import transcribe_with_fallback

        try:
            return await with_timeout(
                transcribe_with_fallback(
                    audio, mime_type=mime, lang_hint=req.lang, trace=req.trace
                ),
                STT_CEILING_MS,
                StageName.STT,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            note = (
                f"stt exceeded the {STT_CEILING_MS / 1000:.0f}s harness ceiling"
                if isinstance(exc, StageTimeout)
                else f"stt failed ({type(exc).__name__}: {exc})"
            )
            req.trace.degrade(note)
            return Transcript(text="", lang=req.lang, provider="none", raw={"detail": note})

    async def _rails(
        self,
        req: _Request,
        rails_name: str | Sequence[Any],
        stage: StageName,
        fallback: Callable[[], list[GuardrailVerdict]],
        **kwargs: Any,
    ) -> tuple[list[GuardrailVerdict], GuardrailVerdict | None]:
        """Run one rail set, recording it as `stage`, and return (verdicts, blocking).

        `rails_name` is the name of a constant on `voxrag.guardrails`, or an already-built
        sequence of rails for a set that cannot be a constant -- the post-retrieval set
        depends on the request's `GenerationMode` (see `guardrails.post_retrieval_rails`).

        `voxrag.guardrails.run_rails` owns the stage span, the per-rail timeout and the
        fail-open policy for a rail that raises -- duplicating any of that here would mean
        two implementations of the same decision. This wrapper only handles the case that
        module cannot handle: not being importable at all.
        """
        module = _guardrails()
        if module is not None:
            try:
                rails = getattr(module, rails_name) if isinstance(rails_name, str) else rails_name
                verdicts, blocking = await module.run_rails(
                    rails, trace=req.trace, stage=stage, **kwargs
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # run_rails is written not to raise, so this is a bug in it or a bad
                # signature after a refactor. Fail open to the built-ins rather than turning
                # a guardrail defect into a total outage.
                label = rails_name if isinstance(rails_name, str) else stage.value
                _LOG.exception("run_rails(%s) raised", label)
                req.trace.degrade(f"{label} unusable ({type(exc).__name__}); built-in rails")
                verdicts = fallback()
                blocking = next((v for v in verdicts if not v.allowed), None)
        else:
            with req.trace.stage(stage, rails="builtin") as detail:
                verdicts = fallback()
                blocking = next((v for v in verdicts if not v.allowed), None)
                detail.update(
                    checked=[v.name for v in verdicts],
                    blocked_by=blocking.name if blocking else None,
                )

        req.verdicts.extend(verdicts)
        # Stage timing first, then the per-rail verdicts: a live waterfall wants the bar
        # closed before the annotations that belong to it start arriving.
        await self._flush(req)
        for verdict in verdicts:
            await self._emit(req, "guardrail", verdict.name, verdict.allowed)
        return verdicts, blocking

    async def _input_phase(self, req: _Request) -> GuardrailVerdict | None:
        """Run the input rails. Returns the blocking verdict, or None to continue."""
        verdicts, blocking = await self._rails(
            req,
            "INPUT_RAILS",
            StageName.INPUT_GUARDRAIL,
            lambda: _builtin_input_verdicts(req.query_text),
            query=req.query_text,
            lang=req.lang,
        )
        for verdict in verdicts:
            # A rail may rewrite rather than block -- control-character stripping, whitespace
            # collapse, PII redaction. The rewritten text is what everything downstream
            # retrieves and answers on, so it has to be applied before the next stage.
            if verdict.allowed and verdict.rewritten_text is not None:
                req.query_text = verdict.rewritten_text
        # LanguageRail already ran a full script analysis; reading its answer back is free,
        # while calling the detector again would repeat the whole histogram.
        module = _guardrails()
        if module is not None:
            req.lang = module.detected_lang(verdicts) or req.lang
        return blocking

    async def _retrieve(self, req: _Request, query: str | None = None) -> RetrievalResult:
        """Embed + dense + lexical + fuse, through the validated tool boundary.

        `query` overrides the request's own text; the agentic graph retrieves with rewrites.
        """
        indexed = set(settings.ingest_langs)
        # Filter by language only when the index is known to hold that language. A wrong
        # guess inside a script family (Hindi vs Marathi) would otherwise filter every
        # candidate away and turn a good query into a refusal.
        filter_lang = req.lang if req.lang in indexed else None
        payload = {
            # Clamped, not rejected: with guardrails disabled nothing has trimmed this yet,
            # and RetrieveInput enforces the same ceiling by raising. The encoder truncates
            # at 192 tokens anyway, so the tail could not have influenced retrieval.
            "query": (query or req.query_text)[: settings.max_query_chars],
            "lang": filter_lang,
            "top_k": settings.fuse_top_k,
        }

        try:
            result = await with_retry(
                lambda: call_tool("retrieve", payload, req.trace, ctx=None),
                # Two attempts, no backoff: the retry that pays for itself here is a refused
                # socket, which fails in ~1ms and often succeeds immediately after. Sleeping
                # `retry_backoff_ms` would guarantee the second attempt misses the budget.
                attempts=2,
                backoff_ms=0.0,
                trace=req.trace,
                stage=StageName.RETRIEVE_DENSE,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            note = f"retrieval failed ({type(exc).__name__}: {exc}); answering is not possible"
            _LOG.warning("%s", note)
            req.trace.degrade(note)
            return RetrievalResult(chunks=[], confidence=0.0, n_candidates=0, degraded=True, note=note)

        assert isinstance(result, RetrievalResult)  # guaranteed by call_tool's output check
        if result.degraded and result.note:
            req.trace.degrade(result.note)
        return result

    async def _retrieval_phase(
        self, req: _Request, retrieval: RetrievalResult
    ) -> GuardrailVerdict | None:
        """Post-retrieval rails: is there anything here worth answering from?

        Timed as a second INPUT_GUARDRAIL span. The stage enum is frozen and has no
        post-retrieval member; two cheap spans under one name is a smaller distortion of the
        report than inventing a stage the contract does not have, and TOTAL/CORE are
        unaffected either way.
        """
        module = _guardrails()
        # Built per request rather than read off the module: the cross-language rail's
        # behaviour depends on this request's mode. Falls back to the constant name so the
        # no-guardrails-module path is unchanged.
        rails: str | Sequence[Any] = (
            module.post_retrieval_rails(req.mode) if module is not None else "POST_RETRIEVAL_RAILS"
        )
        _verdicts, blocking = await self._rails(
            req,
            rails,
            StageName.INPUT_GUARDRAIL,
            lambda: _builtin_retrieval_verdicts(retrieval),
            query=req.query_text,
            context=retrieval.chunks,
            lang=req.lang,
            timeout_ms=_post_retrieval_timeout_ms(req.mode),
        )
        return blocking

    async def _rerank(self, req: _Request) -> list[ScoredChunk]:
        """Optional cross-encoder precision pass over the fused candidates."""
        if not settings.enable_rerank:
            req.trace.skip(StageName.RERANK, "disabled")
            return req.retrieved
        reranker = _resolve_reranker()
        if reranker is None:
            req.trace.skip(StageName.RERANK, "no reranker available")
            return req.retrieved

        with req.trace.stage(StageName.RERANK, model=getattr(reranker, "name", "unknown")) as detail:
            try:
                ranked = await with_timeout(
                    # Cross-encoders are heavy CPU work: never inline on the event loop.
                    asyncio.to_thread(
                        reranker.rerank,
                        req.query_text,
                        req.retrieved,
                        top_k=settings.context_top_k,
                    ),
                    stage_budget_ms(StageName.RERANK),
                    StageName.RERANK,
                )
                detail["reranked"] = len(ranked)
                return list(ranked)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Reranking is a precision optimisation, not a correctness requirement: the
                # fused order is already a valid ranking.
                detail.update(failed=True, error=f"{type(exc).__name__}: {exc}")
                req.trace.degrade(f"rerank failed ({type(exc).__name__}); using fused order")
                return req.retrieved

    async def _generate(
        self,
        req: _Request,
        *,
        query: str | None = None,
        context: list[ScoredChunk] | None = None,
    ) -> Answer:
        """Generate through the tool layer, degrading QUALITY -> FAST -> extractive.

        `query` and `context` default to the request's own; the agentic graph passes the
        question with the output rails' feedback, and the passages its grader kept.
        """
        context = req.retrieved[: settings.context_top_k] if context is None else context
        payload: dict[str, Any] = {
            "query": query or req.query_text,
            "chunk_ids": [scored.chunk.chunk_id for scored in context],
            "mode": req.mode,
            "lang": req.lang,
        }

        with req.trace.stage(StageName.GENERATE, mode=req.mode.value, context=len(context)) as detail:
            try:
                answer = await with_retry(
                    lambda: call_tool("answer", payload, req.trace, ctx=context),
                    # The abstractive path talks to a model server over HTTP and is worth
                    # retrying; the fast path is local, deterministic, and a second identical
                    # attempt would only spend budget.
                    attempts=settings.max_retries + 1 if req.mode is not GenerationMode.FAST else 1,
                    trace=req.trace,
                    stage=StageName.GENERATE,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                detail.update(failed_mode=req.mode.value, error=f"{type(exc).__name__}: {exc}")
                req.trace.degrade(
                    f"{req.mode.value} generation failed ({type(exc).__name__}); serving fast mode"
                )
                payload.update(mode=GenerationMode.FAST, query=req.query_text)
                try:
                    # The fast path falls back to the harness's own extractive answer
                    # internally, so this call effectively cannot fail -- but if it does,
                    # a refusal is still a result.
                    answer = await call_tool("answer", payload, req.trace, ctx=context)
                except asyncio.CancelledError:
                    raise
                except Exception as inner:
                    detail["fallback_error"] = f"{type(inner).__name__}: {inner}"
                    req.trace.degrade(f"fast generation also failed ({type(inner).__name__})")
                    answer = await self._refusal_answer(req, RefusalReason.LOW_CONFIDENCE)

            assert isinstance(answer, Answer)  # guaranteed by call_tool's output check
            detail.update(
                chars=len(answer.text),
                citations=len(answer.citations),
                model=answer.model,
                refused=answer.refused,
            )
        return answer

    async def _output_phase(
        self, req: _Request, answer: Answer, context: list[ScoredChunk] | None = None
    ) -> Answer:
        """Output rails. A blocking verdict replaces the answer with a refusal -- the whole
        point of a groundedness check is that the ungrounded text never reaches the user."""
        context = req.retrieved[: settings.context_top_k] if context is None else context
        verdicts, blocking = await self._rails(
            req,
            "OUTPUT_RAILS",
            StageName.OUTPUT_GUARDRAIL,
            lambda: _builtin_output_verdicts(answer, context, req.lang),
            query=req.query_text,
            answer=answer,
            context=context,
            lang=req.lang,
        )

        if blocking is not None:
            module = _guardrails()
            if module is not None:
                # One substitution, defined in one place: refusal text, refused flag, reason,
                # and citations dropped (provenance for text nobody sees is provenance for
                # nothing).
                return module.apply_verdict(answer, blocking)
            return await self._refusal_answer(
                req, blocking.reason or RefusalReason.UNGROUNDED_ANSWER
            )

        if answer.groundedness == 0.0:
            # Surface the rail's own measurement when the generator did not compute one, so
            # the API response always carries a groundedness number. Only the groundedness
            # rail's score means groundedness -- the others score citation coverage and
            # refusal consistency on the same [0,1] scale.
            measured = next(
                (v.score for v in verdicts if "ground" in v.name.lower()), None
            )
            if measured is not None:
                answer = answer.model_copy(update={"groundedness": measured})
        return answer

    # -- terminal paths ------------------------------------------------------------------

    async def _refusal_answer(self, req: _Request, reason: RefusalReason) -> Answer:
        """Build a refusal through the tool layer, so refusals are structured output too."""
        answer = await call_tool(
            "refuse", {"reason": reason, "lang": req.lang}, req.trace, ctx=None
        )
        assert isinstance(answer, Answer)
        return answer

    async def _refuse(self, req: _Request, verdict: GuardrailVerdict) -> PipelineResult:
        """Terminate the request with a refusal, keeping the full timing record."""
        if verdict not in req.verdicts:
            req.verdicts.append(verdict)
        reason = verdict.reason or RefusalReason.LOW_CONFIDENCE
        try:
            answer = await self._refusal_answer(req, reason)
        except Exception as exc:  # the refusal tool is local and pure; this is paranoia
            _LOG.error("refusal tool failed: %r", exc)
            answer = Answer(
                text=settings.refusal_message,
                mode=GenerationMode.FAST,
                refused=True,
                refusal_reason=reason,
                model="harness",
                lang=req.lang,
            )
        await self._emit(req, "answer", answer.text, answer.mode.value)
        return await self._seal(req, answer)

    async def _seal(self, req: _Request, answer: Answer) -> PipelineResult:
        """Close the trace, publish it, and build the response."""
        if not req.sealed:
            req.sealed = True
            seen = {timing.stage for timing in req.trace.timings}
            for stage in _STAGE_ORDER:
                if stage not in seen:
                    req.trace.skip(stage, "not reached")
            req.trace.finish()
            if req.record:
                TRACES.add(req.trace)
            await self._flush(req)
        timings = req.trace.timings

        return PipelineResult(
            request_id=req.request_id,
            transcript=req.transcript,
            query_text=req.query_text,
            detected_lang=req.lang,
            answer=answer,
            retrieved=req.retrieved,
            guardrails=req.verdicts,
            timings=timings,
            degradations=req.trace.degradations,
            retries=req.trace.retries,
        )

    # -- events --------------------------------------------------------------------------

    async def _emit(self, req: _Request, kind: str, name: Any, value: Any = None) -> None:
        """Push one progress event. A broken consumer must never fail the query."""
        if req.on_event is None:
            return
        try:
            result = req.on_event(kind, name, value)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _LOG.debug("on_event(%s) raised %r; dropping the event", kind, exc)

    async def _flush(self, req: _Request) -> None:
        """Emit a "stage" event for every timing recorded since the last flush.

        Stages are timed in several places (this file, HybridRetriever, the STT failover),
        so the trace is the single source of what happened -- draining it is simpler and
        more complete than making every one of those call sites know about the callback.
        """
        if req.on_event is None:
            req.cursor = len(req.trace.timings)
            return
        for timing in req.trace.timings[req.cursor :]:
            # Skipped stages carry 0.0ms and would draw a phantom bar in a waterfall. They
            # are still in the returned PipelineResult, where they can be rendered as absent.
            if not timing.skipped:
                await self._emit(req, "stage", timing.stage.value, round(timing.ms, 3))
        req.cursor = len(req.trace.timings)


# --------------------------------------------------------------------------------------
# Reranker resolution (optional dependency, resolved once)
# --------------------------------------------------------------------------------------

_RERANKER: Any | None = None
_RERANKER_RESOLVED = False


def _resolve_reranker() -> Any | None:
    """Find a Reranker implementation, or None. Cached including the negative result: the
    import cost must not be paid per request just to rediscover that it is absent."""
    global _RERANKER, _RERANKER_RESOLVED
    if _RERANKER_RESOLVED:
        return _RERANKER
    for module_name, attr_name in (
        ("voxrag.rerank", "get_reranker"),
        ("voxrag.store.rerank", "get_reranker"),
        ("voxrag.rerank", "build_reranker"),
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        factory = getattr(module, attr_name, None)
        if not callable(factory):
            continue
        try:
            candidate = factory()
        except Exception as exc:
            _LOG.warning("reranker %s.%s failed to build: %r", module_name, attr_name, exc)
            continue
        if candidate is not None and hasattr(candidate, "rerank"):
            _RERANKER = candidate
            break
    else:
        _LOG.info("rerank enabled but no reranker module found; fused order will be used")
    _RERANKER_RESOLVED = True
    return _RERANKER


# --------------------------------------------------------------------------------------
# Process-wide singleton
# --------------------------------------------------------------------------------------

_HARNESS: Harness | None = None
_HARNESS_LOCK = threading.Lock()


def get_harness() -> Harness:
    """The one harness for this process. It owns warmup state, so a second instance would
    re-pay every cold start."""
    global _HARNESS
    if _HARNESS is not None:
        return _HARNESS
    with _HARNESS_LOCK:
        if _HARNESS is None:
            _HARNESS = Harness()
    return _HARNESS


__all__ = [
    "DEFAULT_MIME",
    "STT_CEILING_MS",
    "WARMUP_QUERY",
    "EventCallback",
    "Harness",
    "detect_lang",
    "get_harness",
    "reset_rail_cache",
]
