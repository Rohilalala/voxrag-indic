"""Cross-language answerability: the one case the lexical grounding rail cannot judge.

`EntityGroundingRail` compares the query's distinctive terms against the retrieved passages.
That is a lexical test, so it only means anything within a language, and it hands back every
pair whose query language is absent from the evidence (`grounding.same_language_hits`). On this
corpus that is the English path: `multilingual-e5-small` is cross-lingual, so an English query
retrieves Indic passages and shares no token with any of them. Live consequence before this
rail existed, against the running API:

    "what is the price of bitcoin right now"  -> ANSWERED, from an unrelated Bengali passage

Similarity cannot close it. That is measured twice and recorded in `config.min_dense_similarity`:
an off-topic query outscores a legitimate one, and 30% of
on-topic English queries score below the off-topic median. So this rail asks a different
question of a different instrument -- does this evidence ANSWER this question -- and gets it
from the generator that is already loaded, as a one-word entailment verdict.

FAIL-CLOSED, AND THAT INVERTS THIS PACKAGE'S USUAL POLICY. Everywhere else a dead dependency
degrades and serves: `run_rails` treats a rail that raises or times out as non-blocking
(`guardrails/__init__.py`), `generate_with_fallback` drops QUALITY to extractive, `_rails`
falls back to built-ins. Those are right because a broken *check* should not become an outage.
Here the check IS the answer: if nothing can confirm that an Odia passage answers an English
question, then the honest response is that we do not know, and serving a confident answer we
could not verify is precisely the failure this rail exists to stop. So a verifier that says no,
raises, times out, or returns anything unparseable all produce the same refusal.

The blast radius of that inversion is bounded by the trigger condition. Only pairs with NO
same-language evidence reach this rail, and 13 of the 14 indexed languages ask in their own
language, so an Ollama outage cannot refuse a single Hindi or Tamil query -- it can only stop
English ones from being answered unverified.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Final

from voxrag.config import settings
from voxrag.contracts import (
    Answer,
    GenerationMode,
    GuardrailVerdict,
    Lang,
    RefusalReason,
    ScoredChunk,
)
from voxrag.guardrails.grounding import same_language_hits

_LOG = logging.getLogger(__name__)

CROSS_LANGUAGE_RAIL: Final = "cross_language"

#: Anything that can answer "does this evidence answer this question", returning None for
#: "could not decide". `OllamaGenerator.entails` is the production one; the tests inject a
#: fake, because a guardrail test that needs a 9.6GB model running is a guardrail test that
#: will be deleted the first time it is inconvenient.
Verifier = Callable[[str, Sequence[ScoredChunk]], Awaitable[bool | None]]


async def _gemma_entails(query: str, context: Sequence[ScoredChunk]) -> bool | None:
    """Default verifier: the QUALITY generator's own client, imported lazily.

    Lazy because `voxrag.guardrails` is imported by the harness on the FAST path too, and
    importing the generator package there would pull in the extractive tokeniser and the httpx
    client for a rail that, in FAST mode, never calls a verifier at all.
    """
    from voxrag.generate import get_generator

    generator = get_generator(GenerationMode.QUALITY)
    entails = getattr(generator, "entails", None)
    if entails is None:  # a generator swapped in without the capability cannot verify
        return None
    result: bool | None = await entails(query, context)
    return result


class CrossLanguageRail:
    """Refuses cross-language pairs that nothing could confirm are answerable.

    Implements `contracts.Guardrail`. Runs after retrieval, after `EntityGroundingRail`, and
    fires only on the pairs that rail declines to judge.
    """

    name = CROSS_LANGUAGE_RAIL
    phase = "input"

    def __init__(
        self,
        mode: GenerationMode = GenerationMode.FAST,
        *,
        verifier: Verifier | None = None,
        timeout_ms: float | None = None,
    ) -> None:
        self.mode = mode
        self._verifier: Verifier = verifier or _gemma_entails
        self._timeout_ms = timeout_ms

    @property
    def timeout_ms(self) -> float:
        """Read at call time, never frozen at construction.

        `_CROSS_LANGUAGE_RAILS` builds one instance per mode at import, so a snapshot taken
        in `__init__` outlives every runtime override: `scripts/validate_config.py --set` and
        `eval/sweep.py` both `setattr(settings, ...)`, and --set is the standard held-out
        A/B tool. Measured with the snapshot in place -- after
        `settings.cross_language_timeout_ms = 200.0`, this rail still carried 1500.0 while
        `orchestrator._post_retrieval_timeout_ms` (which reads live) returned 450.0, so
        `run_rails`' fail-OPEN ceiling fired first: elapsed 451ms, blocking=None, detail
        "rail timed out after 450ms and was skipped", and an unverified cross-language answer
        was served with the trace reporting that the rail had run. Same class of defect as
        the frozen fusion knobs, which have their own regression test.
        """
        return settings.cross_language_timeout_ms if self._timeout_ms is None else self._timeout_ms

    async def check(
        self,
        *,
        query: str,
        answer: Answer | None = None,
        context: list[ScoredChunk] | None = None,
        lang: Lang | None = None,
    ) -> GuardrailVerdict:
        if not context:
            return GuardrailVerdict(
                name=self.name, allowed=True, detail="no context; OffTopicRail owns this case"
            )

        # The same expression EntityGroundingRail uses to decide it must skip. One definition,
        # so exactly one of the two rails owns any given pair.
        if same_language_hits(context, lang):
            return GuardrailVerdict(
                name=self.name,
                allowed=True,
                detail=(
                    f"evidence includes {lang.value if lang else 'the query'} passages; "
                    "EntityGroundingRail judges this pair lexically"
                ),
            )

        found = ", ".join(sorted({hit.chunk.lang.value for hit in context}))
        asked = lang.value if lang else "unknown"

        if self.mode is not GenerationMode.QUALITY:
            # FAST is extractive: it copies sentences out of the passages, so an English
            # question answered from Odia evidence would return Odia text the asker cannot
            # read, and there is no model in the FAST path to verify it with either.
            return self._refuse(
                f"query is {asked}, evidence is [{found}]; FAST mode has no verifier",
                decided=True,
            )

        if not settings.cross_language_verification:
            return self._refuse(
                f"query is {asked}, evidence is [{found}]; "
                "cross-language verification is disabled by configuration",
                decided=True,
            )

        try:
            verdict = await asyncio.wait_for(
                self._verifier(query, context), self.timeout_ms / 1000.0
            )
        except TimeoutError:
            # Deliberately NOT the fail-open behaviour of `run_rails`' own timeout: an
            # unverified cross-language pair is exactly what this rail refuses.
            _LOG.warning("cross-language verifier exceeded %.0fms", self.timeout_ms)
            return self._refuse(
                f"query is {asked}, evidence is [{found}]; "
                f"the verifier did not answer within {self.timeout_ms:.0f}ms",
                decided=False,
            )
        except Exception as exc:
            _LOG.exception("cross-language verifier failed")
            return self._refuse(
                f"query is {asked}, evidence is [{found}]; "
                f"the verifier failed ({type(exc).__name__}: {exc})",
                decided=False,
            )

        if verdict is True:
            return GuardrailVerdict(
                name=self.name,
                allowed=True,
                detail=(
                    f"query is {asked}, evidence is [{found}]; "
                    "the verifier confirms the passages answer it"
                ),
            )
        if verdict is False:
            return self._refuse(
                f"query is {asked}, evidence is [{found}]; "
                "the verifier says these passages do not answer it",
                decided=True,
            )
        return self._refuse(
            f"query is {asked}, evidence is [{found}]; the verifier could not decide",
            decided=False,
        )

    def _refuse(self, detail: str, *, decided: bool) -> GuardrailVerdict:
        """Every refusal this rail issues, split by whether anything actually judged the pair.

        `decided=True` means something reached a conclusion: the verifier said NO, or the
        mode/configuration cannot serve the pair at all. Those get `score=1.0` -- confidence
        that the BLOCK is correct -- and are not degradations.

        `decided=False` means nothing could check: the verifier timed out, raised, or replied
        with something unreadable. Those refuse just as hard, because that is the whole point
        of a fail-closed rail, but they carry `score=0.0` and `degraded=True`. Reporting them
        as 1.0 would launder "we do not know" into "we are sure" in every dashboard reading
        that field, and omitting the degradation would make a run in which the verifier never
        executed once look identical to one in which it judged every pair -- which is what
        `data/eval_crosslang.json` and every bench artifact would otherwise have recorded.

        One flag, not two, because `score=1.0, degraded=True` is a contradiction and there is
        no reason to let a future call site express it.
        """
        return GuardrailVerdict(
            name=self.name,
            allowed=False,
            reason=RefusalReason.UNVERIFIED_CROSS_LANGUAGE,
            score=1.0 if decided else 0.0,
            detail=detail,
            degraded=not decided,
        )


__all__ = ["CROSS_LANGUAGE_RAIL", "CrossLanguageRail", "Verifier"]
