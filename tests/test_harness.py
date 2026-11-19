"""What the orchestrator actually hands its rails -- as opposed to what the rails can do.

`tests/test_guardrails.py` proves `CrossLanguageRail` refuses a cross-language pair whose
verifier is broken, and proves `post_retrieval_rails(mode)` contains that rail. Neither says
anything about whether one line of `orchestrator.py` ever calls either. Four mutations
demonstrated the gap, each of them leaving the whole suite green at 553 passed:

  1. `_retrieval_phase` resolving the rail set as the literal "POST_RETRIEVAL_RAILS" -- the
     constant that by design excludes the cross-language rail. The rail then runs on no
     request at all and "what is the price of bitcoin right now" is answered from a Bengali
     passage again, which is the live failure the rail was written for.
  2. `post_retrieval_rails(GenerationMode.FAST)` instead of `req.mode`, so a QUALITY request
     gets the FAST rail: every English query refused, no verifier ever asked, and the demo's
     QUALITY toggle silently stops meaning anything.
  3. Deleting the `timeout_ms=` argument at the call site. The set then runs on
     `stage_timeout_ms["input_guardrail"]` = 25ms, `run_rails._run_one` converts the rail's
     fail-CLOSED refusal into a fail-OPEN "rail timed out after 25ms and was skipped", and
     the unverified answer is served -- verbatim the regression that
     `_CROSS_LANGUAGE_SLACK_MS` exists to prevent.
  4. Building the production rails with `timeout_ms=10_000.0`, so the rail takes far longer
     than the harness allots it and `run_rails` wins the race in the wrong direction.

So these tests assert the wiring: which rails, built for which mode, under which budget.
`run_rails` is replaced by a recorder, because what is being tested is the arguments, not
the rails -- their behaviour has its own file.

Hermetic: no retrieval, no model, no store. `_retrieval_phase` is driven directly with a
hand-built `RetrievalResult`.
"""

from __future__ import annotations

from typing import Any

import pytest

from voxrag.config import settings
from voxrag.contracts import (
    Chunk,
    GenerationMode,
    GuardrailVerdict,
    Lang,
    ScoredChunk,
)
from voxrag.guardrails import CROSS_LANGUAGE_RAIL, GROUNDING_RAIL, OFF_TOPIC_RAIL
from voxrag.harness import orchestrator
from voxrag.harness.orchestrator import Harness, _post_retrieval_timeout_ms, _Request
from voxrag.harness.tools import RetrievalResult
from voxrag.telemetry import Trace


def indic_context() -> list[ScoredChunk]:
    """What an English query retrieves against this corpus: not one English passage."""
    return [
        ScoredChunk(
            chunk=Chunk(
                chunk_id=f"c{n}",
                doc_id=f"d{n}",
                text=text,
                lang=lang,
                strategy=settings.active_strategy,
                ordinal=0,
            ),
            score=0.5,
            source="dense",
        )
        for n, (text, lang) in enumerate(
            [
                ("বাংলাদেশের রাজধানী ঢাকা, যা বুড়িগঙ্গা নদীর তীরে অবস্থিত।", Lang.BEN),
                ("तमिलनाडु की राजधानी चेन्नई है।", Lang.HIN),
            ]
        )
    ]


class RailRecorder:
    """Stands in for `voxrag.guardrails`, capturing what the orchestrator asks it to run.

    Only the three names `_rails` and `_retrieval_phase` touch are implemented. Anything
    else the orchestrator starts calling should fail loudly here rather than be absorbed.
    """

    def __init__(self, module: Any) -> None:
        self._module = module
        self.calls: list[dict[str, Any]] = []

    def post_retrieval_rails(self, mode: GenerationMode) -> list[Any]:
        return self._module.post_retrieval_rails(mode)

    def detected_lang(self, verdicts: Any) -> Lang | None:
        return self._module.detected_lang(verdicts)

    async def run_rails(
        self, rails: Any, **kwargs: Any
    ) -> tuple[list[GuardrailVerdict], GuardrailVerdict | None]:
        self.calls.append({"rails": list(rails), **kwargs})
        return [], None


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> RailRecorder:
    import voxrag.guardrails as guardrails_module

    rec = RailRecorder(guardrails_module)
    monkeypatch.setattr(orchestrator, "_guardrails", lambda: rec)
    return rec


def request_for(mode: GenerationMode) -> _Request:
    req = _Request(trace=Trace(request_id="harness-test"), request_id="harness-test", mode=mode)
    req.query_text = "what is the price of bitcoin right now"
    req.lang = Lang.ENG
    return req


async def drive(rec: RailRecorder, mode: GenerationMode) -> dict[str, Any]:
    """Run the post-retrieval phase once and return what `run_rails` was handed."""
    req = request_for(mode)
    retrieval = RetrievalResult(chunks=indic_context(), confidence=0.5, n_candidates=2)
    await Harness()._retrieval_phase(req, retrieval)
    assert len(rec.calls) == 1
    return rec.calls[0]


class TestPostRetrievalWiring:
    """The orchestrator's half of the cross-language contract."""

    @pytest.mark.parametrize("mode", list(GenerationMode))
    async def test_the_cross_language_rail_reaches_run_rails(
        self, recorder: RailRecorder, mode: GenerationMode
    ) -> None:
        # Mutation 1. Without this, the rail can be absent from every real request while
        # fifteen unit tests keep proving it works in isolation.
        call = await drive(recorder, mode)
        assert [rail.name for rail in call["rails"]] == [
            OFF_TOPIC_RAIL,
            GROUNDING_RAIL,
            CROSS_LANGUAGE_RAIL,
        ]

    async def test_the_rail_is_built_for_this_request_s_mode(
        self, recorder: RailRecorder
    ) -> None:
        # Mutation 2. `post_retrieval_rails` is mode-aware; the call site has to pass the
        # REQUEST's mode, or QUALITY silently serves the FAST policy (refuse every English
        # query without asking a verifier) and the demo's mode toggle stops meaning anything.
        quality = await drive(recorder, GenerationMode.QUALITY)
        recorder.calls.clear()
        fast = await drive(recorder, GenerationMode.FAST)
        assert quality["rails"][-1].mode is GenerationMode.QUALITY
        assert fast["rails"][-1].mode is GenerationMode.FAST
        assert quality["rails"][-1] is not fast["rails"][-1]

    @pytest.mark.parametrize("mode", list(GenerationMode))
    async def test_the_set_budget_reaches_run_rails(
        self, recorder: RailRecorder, mode: GenerationMode
    ) -> None:
        # Mutation 3. `_post_retrieval_timeout_ms` being correct is worth nothing if its
        # value never arrives: with the kwarg dropped the set runs on the 25ms stage default
        # and `_run_one` turns the rail's refusal into "was skipped" -- fail-open beating
        # fail-closed, which is the whole reason the argument exists.
        call = await drive(recorder, mode)
        assert call["timeout_ms"] == _post_retrieval_timeout_ms(mode)

    async def test_the_budget_leaves_the_rail_room_to_refuse(
        self, recorder: RailRecorder
    ) -> None:
        # Mutation 4. Asserted against the rail INSTANCE the orchestrator is about to run,
        # not against `settings`: the two diverged for exactly as long as the rail snapshotted
        # its ceiling in __init__, and a test reading settings cannot see that gap.
        call = await drive(recorder, GenerationMode.QUALITY)
        rail = call["rails"][-1]
        assert call["timeout_ms"] is not None
        assert call["timeout_ms"] - rail.timeout_ms >= 100.0, (
            "run_rails ALLOWS a rail that overruns and this rail REFUSES, so the rail's own "
            "ceiling has to fire first -- whoever times out first decides the policy"
        )

    async def test_fast_stays_on_the_cheap_stage_ceiling(self, recorder: RailRecorder) -> None:
        # FAST refuses cross-language pairs on policy without calling a verifier, so it must
        # not buy the QUALITY budget: that would put a 1750ms ceiling in front of the 200ms path.
        call = await drive(recorder, GenerationMode.FAST)
        assert call["timeout_ms"] is None

    async def test_the_budget_tracks_the_setting_at_call_time(
        self, recorder: RailRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Same shape as test_hybrid.py::test_fusion_reads_its_knobs_at_call_time_not_at_import,
        # and for the same reason: `scripts/validate_config.py --set` and `eval/sweep.py` both
        # setattr onto the live settings object, and the production rails are built once at
        # import. Measured before the fix -- with the setting at 200ms the rail still carried
        # 1500ms while the set ceiling moved to 450ms, so run_rails' fail-OPEN timeout fired
        # first and served an unverified answer.
        monkeypatch.setattr(settings, "cross_language_timeout_ms", 200.0)
        call = await drive(recorder, GenerationMode.QUALITY)
        rail = call["rails"][-1]
        assert rail.timeout_ms == 200.0
        assert call["timeout_ms"] == 450.0
