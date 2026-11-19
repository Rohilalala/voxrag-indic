"""AGENTIC mode end to end: the real harness, rails and graph over a fake index and model.

Only two things are fake. Retrieval runs on `FakeEmbedder` / `FakeVectorStore` /
`FakeLexicalIndex` behind the real `HybridRetriever`, and the model is a real
`OllamaGenerator` whose `_request_json` transport is scripted, so prompt building, the strict
YES/NO parse, citation parsing and groundedness are all the production code. Every input,
post-retrieval and output rail is the real one.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from typing import Any

import pytest
import respx
from conftest import HI_TEXT, FakeEmbedder, FakeLexicalIndex, FakeVectorStore

import voxrag.generate as generate_module
from voxrag.config import settings
from voxrag.contracts import (
    Answer,
    Chunk,
    GenerationMode,
    Lang,
    PipelineResult,
    RefusalReason,
    StageName,
)
from voxrag.eval import answer as answer_eval
from voxrag.eval.qrels import EvalQuery
from voxrag.generate.ollama_gemma import LLM_CALLS, GenerationError, OllamaGenerator
from voxrag.harness import agentic, orchestrator, retry, tools
from voxrag.harness.orchestrator import Harness
from voxrag.store import hybrid
from voxrag.telemetry import TRACES, report

QUESTION = "भारत की राजधानी क्या है"
PARLIAMENT = (
    "नई दिल्ली भारत की राजधानी है और यह केंद्र सरकार का मुख्यालय है। "
    "संसद भवन और राष्ट्रपति भवन इसी शहर में हैं।"
)
MUMBAI = "मुंबई महाराष्ट्र की राजधानी है और यह देश का आर्थिक केंद्र मानी जाती है।"
GROUNDED = "भारत की राजधानी नई दिल्ली है [1]।"
#: "The moon has cheese mines": nothing in the corpus carries a word of it.
UNGROUNDED = "चंद्रमा पर पनीर की खदानें हैं [1]।"
#: Ranks HI_TEXT (hi-0) first; the question itself ranks PARLIAMENT (hi-1) first.
YAMUNA_QUERY = "यमुना नदी के किनारे कौन सा शहर है"


def yamuna_only(evidence: str) -> str:
    return "YES" if "यमुना" in evidence else "NO"


class ScriptedModel:
    """Stands in for the model server behind `OllamaGenerator._request_json`.

    Tells the three call kinds apart by their system prompt, records each one, and answers
    from a script: `grade(evidence) -> reply`, then `rewrites` and `answers` in order.
    """

    def __init__(
        self,
        grade: Callable[[str], str] = lambda evidence: "YES",
        rewrites: list[str] | None = None,
        answers: list[str] | None = None,
    ) -> None:
        self.grade = grade
        self.rewrites = list(rewrites or [])
        self.answers = list(answers or [GROUNDED])
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        system, user = payload["messages"][0]["content"], payload["messages"][-1]["content"]
        if system.startswith("You verify"):
            kind, reply = "grade", self.grade(user)
        elif system.startswith("Rewrite"):
            kind, reply = "rewrite", self.rewrites.pop(0) if self.rewrites else ""
        else:
            reply = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
            kind = "generate"
        self.calls.append((kind, user))
        return {"message": {"content": reply}}

    def count(self, kind: str) -> int:
        return sum(1 for k, _ in self.calls if k == kind)


def install(monkeypatch: pytest.MonkeyPatch, model: Any, *, healthy: bool = True) -> None:
    llm = OllamaGenerator()
    monkeypatch.setattr(llm, "_request_json", model)

    async def health() -> bool:
        return healthy

    monkeypatch.setattr(llm, "health", health)
    monkeypatch.setattr(generate_module, "_OLLAMA", llm)


@pytest.fixture(autouse=True)
async def stack(monkeypatch: pytest.MonkeyPatch, patch_settings: Callable[..., None]) -> None:
    """Three Hindi passages behind the real hybrid retriever; fresh tool and breaker caches."""
    chunks = [
        Chunk(
            chunk_id=f"hi-{n}",
            doc_id=f"hi-{n}",
            text=text,
            lang=Lang.HIN,
            strategy=settings.active_strategy,
        )
        for n, text in enumerate([HI_TEXT, PARLIAMENT, MUMBAI])
    ]
    embedder, store, lexical = FakeEmbedder(), FakeVectorStore(), FakeLexicalIndex()
    await store.upsert(chunks, embedder.encode([c.text for c in chunks]))
    lexical.index(chunks)
    retriever = hybrid.HybridRetriever(store, lexical, enable_lexical=True)
    monkeypatch.setattr(hybrid, "_INSTANCE", retriever)
    monkeypatch.setattr(tools, "_EMBEDDER", embedder)
    monkeypatch.setattr(tools, "_GENERATORS", {})
    monkeypatch.setattr(retry, "_BREAKERS", {})
    # Tuned for the e5 encoder; a bag-of-words fake's cosines are not on that scale.
    patch_settings(min_dense_similarity=0.0)


async def ask(mode: str = "agentic", events: list[tuple[str, Any]] | None = None) -> PipelineResult:
    def record(kind: str, name: Any, value: Any) -> None:
        if events is not None:
            events.append((kind, name))

    return await Harness().run(text=QUESTION, lang="hi", mode=mode, on_event=record)


def stages(result: PipelineResult, stage: StageName) -> int:
    return sum(1 for t in result.timings if t.stage is stage and not t.skipped)


async def test_relevant_passages_are_answered_without_a_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ScriptedModel()
    install(monkeypatch, model)

    result = await ask()

    assert not result.answer.refused
    assert result.answer.text == GROUNDED
    assert result.answer.citations
    assert model.count("rewrite") == 0
    assert model.count("grade") == len(result.retrieved) == 3, "one call per passage"
    assert stages(result, StageName.FUSE) == 1
    assert stages(result, StageName.REWRITE) == 0


async def test_weak_grades_rewrite_and_retrieve_again(
    monkeypatch: pytest.MonkeyPatch, patch_settings: Callable[..., None]
) -> None:
    # The grader passes only the Yamuna passage. The question's own top hit is the
    # parliament passage, so only the rewrite's retrieval can find the one it accepts.
    patch_settings(context_top_k=1)
    model = ScriptedModel(grade=yamuna_only, rewrites=[YAMUNA_QUERY])
    install(monkeypatch, model)

    result = await ask()

    assert model.count("rewrite") == 1
    assert stages(result, StageName.FUSE) == 2, "the rewrite has to be retrieved, not just made"
    assert result.retrieved[0].chunk.chunk_id == "hi-0"
    assert not result.answer.refused
    assert [c.chunk_id for c in result.answer.citations] == ["hi-0"]


async def test_rewrites_stop_at_the_maximum(monkeypatch: pytest.MonkeyPatch) -> None:
    model = ScriptedModel(
        grade=lambda evidence: "NO",
        rewrites=["दिल्ली शहर", "लाल किला", "यमुना नदी", "मुंबई शहर"],
    )
    install(monkeypatch, model)

    result = await ask()

    assert model.count("rewrite") == agentic.MAX_REWRITES
    assert stages(result, StageName.FUSE) == agentic.MAX_REWRITES + 1
    # Out of rewrites with the rails satisfied: answer from what was retrieved.
    assert model.count("generate") == 1


async def test_retrieval_the_rails_refuse_is_rewritten_then_refused(
    monkeypatch: pytest.MonkeyPatch, patch_settings: Callable[..., None]
) -> None:
    patch_settings(min_dense_similarity=0.99)  # OffTopicRail refuses every retrieval
    model = ScriptedModel(rewrites=["दिल्ली शहर", "लाल किला"])
    install(monkeypatch, model)

    result = await ask()

    assert model.count("rewrite") == agentic.MAX_REWRITES
    assert model.count("grade") == model.count("generate") == 0
    refusals = [v for v in result.guardrails if not v.allowed]
    assert len(refusals) == agentic.MAX_REWRITES + 1, "one per retrieval"
    assert {v.name for v in refusals} == {"off_topic"}
    assert result.answer.refused
    assert result.answer.refusal_reason is refusals[-1].reason


async def test_an_ungrounded_answer_is_regenerated_with_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ScriptedModel(answers=[UNGROUNDED, GROUNDED])
    install(monkeypatch, model)

    result = await ask()

    prompts = [user for kind, user in model.calls if kind == "generate"]
    assert len(prompts) == 2
    assert "rejected" not in prompts[0]
    assert "rejected" in prompts[1] and "groundedness" in prompts[1]
    assert not result.answer.refused
    assert result.answer.text == GROUNDED


async def test_ungrounded_twice_is_refused_and_never_streamed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ScriptedModel(answers=[UNGROUNDED])
    install(monkeypatch, model)
    events: list[tuple[str, Any]] = []

    result = await ask(events=events)

    assert model.count("generate") == agentic.MAX_GENERATIONS
    assert result.answer.refused
    assert result.answer.refusal_reason is RefusalReason.UNGROUNDED_ANSWER
    assert result.answer.text == settings.refusal_message
    assert all(UNGROUNDED not in str(name) for _, name in events)


async def test_a_dead_model_degrades_to_the_extractive_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def down(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise GenerationError("connection refused")

    install(monkeypatch, down, healthy=False)

    result = await ask()

    assert not result.answer.refused
    assert result.answer.mode is GenerationMode.FAST
    assert result.answer.model == generate_module._EXTRACTIVE.name
    assert any("grader" in note for note in result.degradations)
    assert any("generation failed" in note for note in result.degradations)


async def test_every_node_is_timed_in_the_trace_and_the_analytics(
    monkeypatch: pytest.MonkeyPatch, patch_settings: Callable[..., None]
) -> None:
    patch_settings(context_top_k=1)
    install(
        monkeypatch,
        ScriptedModel(grade=yamuna_only, rewrites=[YAMUNA_QUERY], answers=[UNGROUNDED, GROUNDED]),
    )

    result = await ask()

    for stage in (
        StageName.QUERY_EMBED,
        StageName.RETRIEVE_DENSE,
        StageName.RETRIEVE_LEXICAL,
        StageName.FUSE,
        StageName.INPUT_GUARDRAIL,
        StageName.GRADE,
        StageName.REWRITE,
        StageName.GENERATE,
        StageName.OUTPUT_GUARDRAIL,
    ):
        assert stages(result, stage) >= 1, f"{stage.value} missing from the trace"
    assert stages(result, StageName.GENERATE) == 2
    analytics = report(TRACES.all())["stages"]
    assert {"grade", "rewrite", "generate"} <= analytics.keys()


async def test_the_graph_terminates_when_nothing_ever_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every grade NO, every rewrite new, every answer ungrounded: the worst case is bounded.
    model = ScriptedModel(
        grade=lambda evidence: "NO",
        rewrites=[f"दिल्ली {n}" for n in range(100)],
        answers=[UNGROUNDED],
    )
    install(monkeypatch, model)

    result = await ask()

    assert model.count("rewrite") == agentic.MAX_REWRITES
    assert model.count("generate") == agentic.MAX_GENERATIONS
    assert result.answer.refused


async def test_fast_mode_never_enters_the_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    model = ScriptedModel()
    install(monkeypatch, model)
    events: list[tuple[str, Any]] = []

    result = await ask("fast", events)

    assert model.calls == []
    assert stages(result, StageName.GRADE) == stages(result, StageName.REWRITE) == 0
    kinds = [kind for kind, _ in events]
    assert kinds.count("token") == 1
    assert kinds.index("token") < events.index(("stage", "generate"))


async def test_without_langgraph_agentic_serves_quality(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, ScriptedModel())
    monkeypatch.setitem(sys.modules, "voxrag.harness.agentic", None)

    result = await ask()

    assert not result.answer.refused
    assert any("`agent` extra" in note for note in result.degradations)
    assert stages(result, StageName.GRADE) == 0


@respx.mock
async def test_rewrite_counts_its_call_and_keeps_one_line() -> None:
    llm = OllamaGenerator()

    async def healthy() -> bool:
        return True

    llm.health = healthy  # type: ignore[method-assign]
    respx.post(f"{settings.ollama_base_url.rstrip('/')}/api/chat").respond(
        json={"message": {"content": '"दिल्ली संसद भवन"\nThis query keeps the key terms.'}}
    )
    calls = [0]
    LLM_CALLS.set(calls)

    assert await llm.rewrite(QUESTION) == "दिल्ली संसद भवन"
    assert calls == [1]
    await llm.aclose()


async def test_the_eval_counts_each_query_s_llm_calls_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Four queries in flight at once, making 1, 2, 3 and 4 calls with every call interleaved:
    # a process-wide counter would hand each query some of the others' calls.
    class CountingHarness:
        async def warmup(self) -> dict[str, Any]:
            return {}

        async def run(self, *, text: str, lang: Lang, mode: GenerationMode) -> PipelineResult:
            for _ in text:
                calls = LLM_CALLS.get()
                assert calls is not None
                calls[0] += 1
                await asyncio.sleep(0)
            answer = Answer(text="x", mode=mode)
            return PipelineResult(request_id=text, query_text=text, answer=answer)

    monkeypatch.setattr(orchestrator, "get_harness", CountingHarness)
    queries = [
        EvalQuery(
            query="q" * n,
            lang=Lang.HIN,
            relevant=frozenset({"d"}),
            negatives=frozenset(),
            gold_answer="x",
        )
        for n in (1, 2, 3, 4)
    ]

    result = await answer_eval.evaluate(queries, mode=GenerationMode.AGENTIC, concurrency=4)

    assert result["overall"]["llm_calls_per_query"] == 2.5
    assert {"p50_latency_ms", "p95_latency_ms"} <= result["overall"].keys()
