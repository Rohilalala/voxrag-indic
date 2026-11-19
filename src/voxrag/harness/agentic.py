"""AGENTIC mode: corrective, self-checking RAG as a LangGraph state machine.

    retrieve --> grade ------------> generate --> check --> END
       ^           |                    ^           |
       |           v                    +-----------+  rejected as ungrounded: regenerate
       +------- rewrite                                once, with the rails' reason
                (no passage graded relevant: at most two rewrites)

Every node is a harness stage rather than a second copy of one: `retrieve` is `_retrieve`
plus the post-retrieval rails, `generate` is `_generate` over the QUALITY generator, `check`
is `_output_phase`. Retries, timeouts, breakers, degradations and the per-stage trace are
therefore the harness's own. The two model calls the harness did not already make get their
own stages: GRADE asks `entails` once per passage, with the strict first-word YES/NO parse the
cross-language rail relies on, and REWRITE asks for one search query.

It degrades the way the harness does: a grader that judges no passage leaves them all
ungraded, a rewrite that fails ends the rewriting, and a dead model falls back to the
extractive answer inside `_generate`. Every loop is bounded by a counter in the state, so the
graph always terminates.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Final, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from voxrag.config import settings
from voxrag.contracts import (
    Answer,
    GenerationMode,
    GuardrailVerdict,
    RefusalReason,
    ScoredChunk,
    StageName,
)
from voxrag.generate import OllamaGenerator, get_generator
from voxrag.harness.orchestrator import Harness, _Request
from voxrag.harness.retry import stage_budget_ms, with_timeout

#: Rewrites after the first retrieval, so at most three retrievals per request.
MAX_REWRITES: Final = 2
#: The first answer plus one regeneration with the output rails' feedback.
MAX_GENERATIONS: Final = 2

#: Put after the question on a regeneration. Instruction first, so a long question cuts the
#: rails' reason and never the instruction or the question itself.
_FEEDBACK: Final = (
    "Your previous answer was rejected. Use only facts stated in the CONTEXT and cite every "
    "sentence. Reason: "
)


class AgentState(TypedDict):
    #: What `retrieve` searches with: the question, then each rewrite of it.
    query: str
    #: The latest fused retrieval.
    passages: list[ScoredChunk]
    #: The post-retrieval rails' refusal of `passages`, if they refused them.
    blocked: GuardrailVerdict | None
    #: What `generate` answers from: the passages graded relevant.
    relevant: list[ScoredChunk]
    rewrites: int
    generations: int
    #: Why the output rails rejected the previous answer; empty before the first.
    feedback: str
    answer: Answer | None


@dataclass
class _Run:
    harness: Harness
    req: _Request


async def _ask_model(work: Awaitable[Any], stage: StageName, detail: dict[str, Any]) -> Any:
    """One model call under the generator's own ceiling. None when it fails or overruns."""
    try:
        return await with_timeout(work, stage_budget_ms(StageName.GENERATE), stage)
    except Exception as exc:
        detail["error"] = f"{type(exc).__name__}: {exc}"
        return None


def _model() -> OllamaGenerator:
    """The QUALITY generator's client, which grades and rewrites as well as answers."""
    return cast(OllamaGenerator, get_generator(GenerationMode.QUALITY))


def _give_up(state: AgentState) -> Command[str]:
    """Stop rewriting: refuse if the rails refused this retrieval, else answer from it."""
    if state["blocked"] is not None:
        return Command(goto=END)
    return Command(
        goto="generate", update={"relevant": state["passages"][: settings.context_top_k]}
    )


async def retrieve(state: AgentState, runtime: Runtime[_Run]) -> dict[str, Any]:
    harness, req = runtime.context.harness, runtime.context.req
    retrieval = await harness._retrieve(req, state["query"])
    req.retrieved = list(retrieval.chunks)
    await harness._flush(req)
    await harness._emit(req, "retrieval", len(retrieval.chunks), round(retrieval.confidence, 4))
    blocked = await harness._retrieval_phase(req, retrieval)
    return {"passages": req.retrieved, "blocked": blocked}


async def grade(state: AgentState, runtime: Runtime[_Run]) -> Command[str]:
    harness, req = runtime.context.harness, runtime.context.req
    top = state["passages"][: settings.context_top_k]
    if not top:  # nothing retrieved: a rewrite cannot fix a dead index
        return _give_up(state)
    relevant: list[ScoredChunk] = []
    if state["blocked"] is None:  # passages the rails refused are not worth grading
        llm = _model()
        with req.trace.stage(StageName.GRADE, graded=len(top)) as detail:
            work = asyncio.gather(*(llm.entails(req.query_text, [hit]) for hit in top))
            verdicts = await _ask_model(work, StageName.GRADE, detail) or [None] * len(top)
            relevant = [hit for hit, ok in zip(top, verdicts, strict=True) if ok is True]
            detail.update(relevant=len(relevant), undecided=verdicts.count(None))
        await harness._flush(req)
        if verdicts.count(None) == len(top):
            req.trace.degrade("grader judged no passage; answering from them ungraded")
            relevant = top
    if relevant:
        return Command(goto="generate", update={"relevant": relevant})
    if state["rewrites"] < MAX_REWRITES:
        return Command(goto="rewrite")
    return _give_up(state)


async def rewrite(state: AgentState, runtime: Runtime[_Run]) -> Command[str]:
    harness, req = runtime.context.harness, runtime.context.req
    llm = _model()
    with req.trace.stage(StageName.REWRITE, rewrite=state["rewrites"] + 1) as detail:
        query = await _ask_model(llm.rewrite(state["query"]), StageName.REWRITE, detail)
        detail["query"] = query
    await harness._flush(req)
    if not query or query == state["query"]:
        req.trace.degrade("query rewrite gave no new query; rewriting stopped")
        return _give_up(state)
    return Command(goto="retrieve", update={"query": query, "rewrites": state["rewrites"] + 1})


async def generate(state: AgentState, runtime: Runtime[_Run]) -> dict[str, Any]:
    harness, req = runtime.context.harness, runtime.context.req
    query = req.query_text
    room = settings.max_query_chars - len(query) - 2
    if state["feedback"] and room > 0:
        query = f"{query}\n\n{(_FEEDBACK + state['feedback'])[:room]}"
    answer = await harness._generate(req, query=query, context=state["relevant"])
    await harness._flush(req)
    return {"answer": answer, "generations": state["generations"] + 1}


async def check(state: AgentState, runtime: Runtime[_Run]) -> Command[str]:
    harness, req = runtime.context.harness, runtime.context.req
    answer = state["answer"]
    assert answer is not None  # set by `generate`, the only way into this node
    seen = len(req.verdicts)
    answer = await harness._output_phase(req, answer, context=state["relevant"])
    await harness._flush(req)
    blocking = next((v for v in req.verdicts[seen:] if not v.allowed), None)
    if (
        blocking is not None
        and blocking.reason is RefusalReason.UNGROUNDED_ANSWER
        and state["generations"] < MAX_GENERATIONS
    ):
        return Command(goto="generate", update={"feedback": blocking.detail})
    return Command(goto=END, update={"answer": answer})


def _build() -> Any:
    graph = StateGraph(AgentState, context_schema=_Run)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade, destinations=("generate", "rewrite", END))
    graph.add_node("rewrite", rewrite, destinations=("retrieve", "generate", END))
    graph.add_node("generate", generate)
    graph.add_node("check", check, destinations=("generate", END))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("generate", "check")
    return graph.compile()


GRAPH: Final = _build()


async def run_graph(harness: Harness, req: _Request) -> Answer | GuardrailVerdict:
    """The checked answer, or the post-retrieval rails' refusal if no retrieval passed them."""
    state = await GRAPH.ainvoke(
        AgentState(
            query=req.query_text,
            passages=[],
            blocked=None,
            relevant=[],
            rewrites=0,
            generations=0,
            feedback="",
            answer=None,
        ),
        context=_Run(harness, req),
    )
    answer: Answer | None = state["answer"]
    return answer if answer is not None else state["blocked"]


__all__ = ["GRAPH", "MAX_GENERATIONS", "MAX_REWRITES", "AgentState", "run_graph"]
