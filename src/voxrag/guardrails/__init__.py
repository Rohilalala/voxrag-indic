"""Guardrails: the part of the system that decides when *not* to answer.

Three sets, three points in the pipeline:

    INPUT_RAILS           before retrieval  -- is this a query we should spend work on?
    POST_RETRIEVAL_RAILS  after fusion      -- did we find anything that could answer it?
    OUTPUT_RAILS          after generation  -- is the answer we produced actually supported?

`run_rails` runs a set concurrently and times the whole set as one trace stage. Concurrency
matters for the shape of the budget rather than for CPU: these rails are microsecond-scale
regex and set work, so `gather` does not make them faster on one core -- what it buys is that
the set costs one stage timing instead of five, and that each rail carries its own timeout, so
a single pathological check cannot eat the 25ms input-guardrail budget on its own. If a rail
later grows an out-of-process dependency (a moderation API, a small classifier), the call site
does not change and the concurrency starts paying rent.

Failure policy: a rail that raises or times out is recorded as a degradation and treated as
non-blocking. That is fail-open, which is the right call *here* and worth stating plainly --
these rails are pure functions over a string, so an exception means a bug in our own regex, and
letting a bug refuse every request would be a self-inflicted outage. Defence in depth covers
the gap: the other rails in the set still ran, and the output rails still stand between a bad
input and a delivered answer. A rail with a network dependency should not inherit this policy:
`cross_language.CrossLanguageRail` is the one such rail, and it catches its own timeouts and
exceptions and turns them into refusals *before* `_run_one` below can convert them into an
allow. Its module docstring says why the inversion is right there and wrong here.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, Final

from voxrag.config import settings
from voxrag.contracts import (
    Answer,
    GenerationMode,
    Guardrail,
    GuardrailVerdict,
    Lang,
    ScoredChunk,
    StageName,
)
from voxrag.guardrails.cross_language import (
    CROSS_LANGUAGE_RAIL,
    CrossLanguageRail,
)
from voxrag.guardrails.grounding import (
    GROUNDING_RAIL,
    EntityGroundingRail,
    distinctive_terms,
    same_language_hits,
)
from voxrag.guardrails.input_rails import (
    EMPTY_QUERY_RAIL,
    INJECTION_RAIL,
    LANGUAGE_RAIL,
    OFF_TOPIC_RAIL,
    PERSONAL_DATA_RAIL,
    SAFETY_RAIL,
    EmptyQueryRail,
    InjectionRail,
    LangDetection,
    LanguageRail,
    OffTopicRail,
    PersonalDataRail,
    SafetyRail,
    detect_lang,
    detected_lang,
)
from voxrag.guardrails.output_rails import (
    CITATION_RAIL,
    GROUNDEDNESS_RAIL,
    REFUSAL_RAIL,
    CitationRail,
    GroundednessRail,
    GroundednessReport,
    RefusalConsistencyRail,
    apply_verdict,
    measure_groundedness,
)
from voxrag.telemetry import Trace

_LOG = logging.getLogger(__name__)

#: Fallbacks used only if a key is absent from settings.stage_timeout_ms.
_INPUT_TIMEOUT_MS: Final = 25.0
_OUTPUT_TIMEOUT_MS: Final = 40.0

# Instantiated once at import. Every rail holds only compiled patterns and resolved thresholds,
# so construction is cheap -- but it is not free, and it has no business happening per request.
# Order is significant: `run_rails` reports the first *listed* blocking verdict, not the first
# to finish, so the list order is what makes the refusal reason deterministic.
INPUT_RAILS: Final[list[Guardrail]] = [
    EmptyQueryRail(),  # cheapest, and catches the most common real failure (a silent mic)
    LanguageRail(),
    SafetyRail(),
    InjectionRail(),
    # Last of the input rails: unanswerable-in-principle questions ("what is MY password").
    # Cannot be deferred to OffTopicRail -- MS MARCO contains banking passages, so those
    # queries retrieve strongly and every downstream check passes on a wrong answer.
    PersonalDataRail(),
]

#: Off-topic is only definable against a corpus, so these wait for the fused candidates.
#: OffTopicRail first: it owns the "retrieved nothing at all" case, and that is a better
#: refusal reason to report than "the passages omit your entities" when there are no passages.
#: EntityGroundingRail then catches what similarity cannot -- retrieval that is topically
#: adjacent but about different entities. See guardrails/grounding.py for why both exist.
POST_RETRIEVAL_RAILS: Final[list[Guardrail]] = [OffTopicRail(), EntityGroundingRail()]

#: The third post-retrieval rail cannot live in that constant: what it does depends on the
#: request's GenerationMode, which a module-level list cannot see. It is still built once per
#: mode rather than per request -- an instance holds a mode and a verifier reference and
#: nothing else -- so `post_retrieval_rails` stays a dict lookup on the hot path.
_CROSS_LANGUAGE_RAILS: Final[dict[GenerationMode, Guardrail]] = {
    mode: CrossLanguageRail(mode) for mode in GenerationMode
}


def post_retrieval_rails(mode: GenerationMode | None = None) -> list[Guardrail]:
    """`POST_RETRIEVAL_RAILS` plus the cross-language rail for this request's mode.

    Ordering is a contract, not a style choice: `run_rails` reports the first *listed*
    blocking verdict, so "we retrieved nothing" is still reported by `OffTopicRail` and a
    same-language pair by `EntityGroundingRail`. The cross-language rail is last and only ever
    names a refusal the other two structurally cannot reach -- it is the only one that fires
    when the query's language is absent from the evidence.
    """
    return [
        *POST_RETRIEVAL_RAILS,
        _CROSS_LANGUAGE_RAILS[mode if mode is not None else settings.default_mode],
    ]


#: Refusal consistency first: if the generator already declined, that is the verdict we want
#: reported, not a groundedness complaint about the refusal message.
OUTPUT_RAILS: Final[list[Guardrail]] = [
    RefusalConsistencyRail(),
    GroundednessRail(),
    CitationRail(),
]


async def _run_one(
    rail: Guardrail,
    timeout_s: float,
    *,
    query: str,
    answer: Answer | None,
    context: list[ScoredChunk] | None,
    lang: Lang | None,
) -> GuardrailVerdict:
    """Run one rail so that it can never raise, and never outlive its budget."""
    try:
        return await asyncio.wait_for(
            rail.check(query=query, answer=answer, context=context, lang=lang), timeout_s
        )
    except TimeoutError:
        _LOG.error("guardrail %s exceeded %.1fms", rail.name, timeout_s * 1000)
        return GuardrailVerdict(
            name=rail.name,
            allowed=True,
            detail=f"rail timed out after {timeout_s * 1000:.0f}ms and was skipped",
        )
    except Exception as exc:  # a broken rail must not break the request -- see module docstring
        _LOG.exception("guardrail %s failed", rail.name)
        return GuardrailVerdict(
            name=rail.name,
            allowed=True,
            detail=f"rail errored and was skipped: {type(exc).__name__}: {exc}",
        )


async def run_rails(
    rails: Sequence[Guardrail],
    *,
    query: str,
    answer: Answer | None = None,
    context: list[ScoredChunk] | None = None,
    lang: Lang | None = None,
    trace: Trace | None = None,
    stage: StageName | None = None,
    timeout_ms: float | None = None,
) -> tuple[list[GuardrailVerdict], GuardrailVerdict | None]:
    """Run a set of rails concurrently; return every verdict and the first blocking one.

    `rails` is a sequence of `contracts.Guardrail`. `trace` is a `telemetry.Trace` or None
    (the benchmark and the unit tests run rails without one). The returned blocking verdict is
    the first in *list* order, so the same query always refuses for the same reason.
    """
    if not rails:
        return [], None

    stage = stage or (
        StageName.OUTPUT_GUARDRAIL if rails[0].phase == "output" else StageName.INPUT_GUARDRAIL
    )

    if not settings.enable_guardrails:
        if trace is not None:
            trace.skip(stage, "guardrails disabled")
        return [], None

    default_ms = _OUTPUT_TIMEOUT_MS if stage is StageName.OUTPUT_GUARDRAIL else _INPUT_TIMEOUT_MS
    budget_ms = timeout_ms if timeout_ms is not None else settings.stage_timeout_ms.get(stage.value, default_ms)
    # The budget is for the set, and the set runs concurrently, so each rail gets all of it.
    timeout_s = max(0.001, budget_ms / 1000.0)

    detail: dict[str, Any] = {}
    cm = trace.stage(stage, rails=len(rails)) if trace is not None else nullcontext(detail)
    with cm as stage_detail:
        verdicts: list[GuardrailVerdict] = list(
            await asyncio.gather(
                *(
                    _run_one(rail, timeout_s, query=query, answer=answer, context=context, lang=lang)
                    for rail in rails
                )
            )
        )
        blocking = next((verdict for verdict in verdicts if not verdict.allowed), None)
        stage_detail["checked"] = [verdict.name for verdict in verdicts]
        stage_detail["blocked_by"] = blocking.name if blocking else None
        if blocking is not None and blocking.reason is not None:
            stage_detail["reason"] = blocking.reason.value

    if trace is not None:
        for verdict in verdicts:
            # Two ways a rail can stop protecting anything, and both have to reach the
            # analytics. "skipped" only ever appears in the detail of the fail-open paths
            # above, so it catches a rail that was timed out and allowed anyway. `degraded`
            # covers the opposite shape, which no string can: a fail-CLOSED rail that
            # REFUSED because its dependency was unreachable. That refusal looks identical
            # in a stored run to one the verifier actually decided -- see the field's comment
            # in contracts.py for the run where it was measured.
            if (verdict.allowed and "was skipped" in verdict.detail) or verdict.degraded:
                trace.degrade(f"guardrail {verdict.name}: {verdict.detail}")

    return verdicts, blocking


__all__ = [
    "CITATION_RAIL",
    "CROSS_LANGUAGE_RAIL",
    "EMPTY_QUERY_RAIL",
    "GROUNDEDNESS_RAIL",
    "GROUNDING_RAIL",
    "INJECTION_RAIL",
    "INPUT_RAILS",
    "LANGUAGE_RAIL",
    "OFF_TOPIC_RAIL",
    "OUTPUT_RAILS",
    "PERSONAL_DATA_RAIL",
    "POST_RETRIEVAL_RAILS",
    "REFUSAL_RAIL",
    "SAFETY_RAIL",
    "CitationRail",
    "CrossLanguageRail",
    "EmptyQueryRail",
    "EntityGroundingRail",
    "GroundednessRail",
    "GroundednessReport",
    "InjectionRail",
    "LangDetection",
    "LanguageRail",
    "OffTopicRail",
    "RefusalConsistencyRail",
    "SafetyRail",
    "apply_verdict",
    "detect_lang",
    "detected_lang",
    "distinctive_terms",
    "measure_groundedness",
    "post_retrieval_rails",
    "run_rails",
    "same_language_hits",
]
