"""`OllamaGenerator.entails` -- the verifier the cross-language rail actually calls.

`TestCrossLanguageRail` injects a `FakeVerifier`, which is right for testing a rail and
leaves the thing the rail calls completely unguarded. Three mutations to
`generate/ollama_gemma.py` each survived the full suite at 553 passed:

  a. Replacing the verdict-parsing tail with `return True`. An unparseable reply and an
     explicit NO both become an ALLOW, so every cross-language query is answered unverified
     -- a fail-CLOSED rail turned inside out by four lines in another module.
  b. Inverting the polarity to `== "no"`. YES and NO swap, so the rail refuses precisely the
     answerable pairs and answers the rest. Every guardrail metric still looks plausible.
  c. `"think": True` with `num_predict: 4096` -- the reasoning-model landmine that
     `settings.ollama_think` warns about. gemma4 spends the whole
     budget on hidden tokens, `_extract_content` yields "", `_VERDICT_RE` misses, `entails`
     returns None, and the rail refuses 100% of English queries at full model cost.

None of these needs a network or a model: the transport is one method, so it is stubbed and
everything above it is exercised for real.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from voxrag.config import settings
from voxrag.contracts import Chunk, Lang, ScoredChunk
from voxrag.generate.ollama_gemma import (
    _ENTAILMENT_MAX_PASSAGE_CHARS,
    _ENTAILMENT_MAX_TOKENS,
    _VERDICT_RE,
    GenerationError,
    OllamaGenerator,
    _entailment_messages,
)


def chunk(text: str, *, lang: Lang = Lang.BEN, n: int = 0) -> ScoredChunk:
    return ScoredChunk(
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


def context(n: int = 6) -> list[ScoredChunk]:
    return [chunk(f"passage {i} " + "বাংলা " * 200, n=i) for i in range(n)]


class StubTransport:
    """Replaces `_request_json`, recording the payload and returning a canned reply."""

    def __init__(self, content: str | None = None, raises: Exception | None = None) -> None:
        self.content = content
        self.raises = raises
        self.payloads: list[dict[str, Any]] = []

    async def __call__(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append({"path": path, **payload})
        if self.raises is not None:
            raise self.raises
        return {"message": {"content": self.content}}


def generator(stub: StubTransport, *, healthy: bool = True) -> OllamaGenerator:
    gen = OllamaGenerator()
    gen._request_json = stub  # type: ignore[method-assign]

    async def health() -> bool:
        return healthy

    gen.health = health  # type: ignore[method-assign]
    return gen


class TestVerdictParsing:
    """A verdict is the FIRST word. Anything else is a model that ignored the instruction,
    and the rail refuses on None -- so strictness here fails closed."""

    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            ("YES", "yes"),
            ("no", "no"),
            (" Yes.", "yes"),
            ("**NO**", "no"),
            ("yes, the evidence covers it", "yes"),
        ],
    )
    def test_reads_a_leading_verdict(self, reply: str, expected: str) -> None:
        match = _VERDICT_RE.match(reply.strip())
        assert match is not None
        assert match.group(1).casefold() == expected

    @pytest.mark.parametrize("reply", ["the answer is yes", "", "maybe", "I cannot tell", "1"])
    def test_refuses_to_guess_at_anything_else(self, reply: str) -> None:
        # "the answer is yes" is the important one: a substring search would read it as a
        # verdict, and a model that is explaining itself is a model that ignored the prompt.
        assert _VERDICT_RE.match(reply.strip()) is None


class TestEntailmentPrompt:
    def test_asks_for_one_word_and_not_the_answer_prompt(self) -> None:
        # `prompts.build_messages` asks for a cited answer in the query's language and tells
        # the model to emit settings.refusal_message when it cannot -- so a model that judged
        # the evidence insufficient would reply with a refusal SENTENCE, which parses as
        # unparseable rather than as NO. Two different questions need two different prompts.
        messages = _entailment_messages("what is a corporation", context(2))
        system, user = messages[0]["content"], messages[1]["content"]
        assert "YES" in system and "NO" in system
        assert settings.refusal_message not in system + user
        assert "what is a corporation" in user

    def test_caps_each_passage(self) -> None:
        # The cap bounds prompt size across context_top_k passages, which is what keeps the
        # verifier cheap enough to sit in front of an answer.
        user = _entailment_messages("q", [chunk("x" * 5_000)])[1]["content"]
        assert "x" * _ENTAILMENT_MAX_PASSAGE_CHARS in user
        assert "x" * (_ENTAILMENT_MAX_PASSAGE_CHARS + 1) not in user


class TestEntails:
    async def test_yes_is_true_and_no_is_false(self) -> None:
        # Mutations (a) and (b). Both directions asserted in one place: a rail wired to a
        # verifier with inverted polarity refuses exactly the answerable pairs.
        assert await generator(StubTransport("YES")).entails("q", context(1)) is True
        assert await generator(StubTransport("NO")).entails("q", context(1)) is False

    @pytest.mark.parametrize("reply", ["", "maybe", "the answer is yes"])
    async def test_an_unreadable_reply_is_not_a_yes(self, reply: str) -> None:
        # Mutation (a). None reaches CrossLanguageRail as a REFUSAL, which is the whole
        # point: "we could not read the verdict" must never become "the evidence answers it".
        assert await generator(StubTransport(reply)).entails("q", context(1)) is None

    async def test_a_dead_daemon_cannot_decide(self) -> None:
        stub = StubTransport(raises=GenerationError("connection refused"))
        assert await generator(stub).entails("q", context(1)) is None

    async def test_an_unhealthy_generator_never_pays_for_a_request(self) -> None:
        # Health is cached for _HEALTH_TTL_S, so once the daemon is known to be down this
        # converts a connect timeout per request into an immediate "cannot decide".
        stub = StubTransport("YES")
        assert await generator(stub, healthy=False).entails("q", context(1)) is None
        assert stub.payloads == []

    @pytest.mark.parametrize("query", ["", "   "])
    async def test_no_query_and_no_context_are_not_asked_about(self, query: str) -> None:
        stub = StubTransport("YES")
        assert await generator(stub).entails(query, context(1)) is None
        assert await generator(stub).entails("q", []) is None
        assert stub.payloads == []

    async def test_thinking_stays_off_and_the_budget_stays_one_word(self) -> None:
        # Mutation (c), the documented landmine: with thinking on, a reasoning model spends
        # the whole num_predict budget on hidden tokens and returns empty content, so the
        # verifier can only ever answer None and the rail refuses every English query at full
        # model cost. settings.ollama_think is False for this reason.
        stub = StubTransport("YES")
        await generator(stub).entails("q", context(1))
        payload = stub.payloads[0]
        assert payload["think"] is settings.ollama_think is False
        assert payload["options"]["temperature"] == 0.0
        assert payload["options"]["num_predict"] == _ENTAILMENT_MAX_TOKENS
        assert payload["stream"] is False

    async def test_verifies_the_passages_the_generator_will_answer_from(self) -> None:
        # Same slice and the same k as `orchestrator._generate` uses. If this ever exceeds
        # context_top_k the verifier is judging evidence the answer will not be built from.
        stub = StubTransport("YES")
        await generator(stub).entails("q", context(10))
        user = stub.payloads[0]["messages"][1]["content"]
        assert user.count("passage ") == settings.context_top_k

    async def test_a_cancelled_load_is_finished_in_the_background(self) -> None:
        # The rail's ceiling (1500ms) sits below a cold model load (6089.7ms measured), and
        # cancelling the request aborts the load with it -- so a cold verifier could never
        # become warm and every cross-language query refused forever. Measured before this:
        # three consecutive QUALITY queries refused at ~1.53s each with `ollama ps` still
        # empty afterwards. The cancellation now leaves a load running so the NEXT query is
        # the warm one, which is what config.cross_language_timeout_ms claims it costs.
        stub = StubTransport(raises=asyncio.CancelledError())
        gen = generator(stub)
        scheduled: list[bool] = []
        gen._schedule_preload = lambda: scheduled.append(True)  # type: ignore[method-assign]
        with pytest.raises(asyncio.CancelledError):
            await gen.entails("q", context(1))
        assert scheduled == [True], "an aborted load that nothing restarts is a dead verifier"
