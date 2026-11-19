"""The latency benchmark: drives the harness over a real query set and reports percentiles.

Four decisions define what these numbers mean, and all four are printed with the results
rather than buried here:

  1. **Text input, not audio.** STT is a round trip to a third-party vendor; including it
     would mean reporting Sarvam's network latency as the system's. `Trace.finish` already
     separates CORE (= TOTAL - STT) for exactly this reason, and CORE is what the 200ms
     budget is graded against. Driving the harness with text makes STT a genuine zero
     instead of a subtracted estimate.
  2. **Warmup runs are discarded, and the report says so and shows them anyway.** The first
     query through the process pays for the ONNX session, the tokeniser, the HNSW graph or
     the Qdrant connection. Folding that into P50 misreports model-load time as query
     latency; hiding the discard is worse. Both blocks are printed.
  3. **Every query is timed twice.** The harness reports its own per-stage timings; the
     benchmark independently wall-clocks each call. If the two disagree the harness is
     under-instrumented, and the report shows both so nobody has to take the instrumented
     number on faith.
  4. **Correctness is reported next to latency.** A pipeline can hit any latency target by
     refusing everything, so refusal rate, groundedness, retrieval score and per-category
     adversarial accuracy sit in the same output as the percentiles.

The benchmark loop itself is kept out of the measurement: the query set is materialised up
front, rich renders nothing while queries are in flight (the progress bar updates only after
each timer stops), and the JSON artefact is written after the last result lands.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import json
import os
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from voxrag.bench.queries import (
    DEFAULT_SEED,
    BenchQuery,
    ExpectedBehaviour,
    corpus_source,
    describe_set,
    load_queries,
)
from voxrag.config import DATA_DIR, settings
from voxrag.contracts import GenerationMode, Lang, PipelineResult, StageName
from voxrag.telemetry import Trace, report, summarize

# --------------------------------------------------------------------------------------
# Module-level constants. Benchmark-harness facts, not runtime tunables.
# --------------------------------------------------------------------------------------

#: Per-query wall-clock ceiling. A wedged dependency must fail one query, not the whole run.
#: FAST is generously above its own 200ms budget so that a slow-but-working machine still
#: produces a report; QUALITY tracks the Ollama timeout plus the rest of the pipeline.
_QUERY_TIMEOUT_S: dict[GenerationMode, float] = {
    GenerationMode.FAST: 15.0,
    GenerationMode.QUALITY: settings.ollama_timeout_s + 20.0,
}

#: Names a harness module might expose, most specific first. Resolution is empirical -- each
#: candidate is actually called once during warmup and kept only if it returns a
#: PipelineResult -- so a wrong guess costs one probe, not a wrong benchmark.
_FACTORY_NAMES = (
    "get_harness",
    "build_harness",
    "create_harness",
    "default_harness",
    "Harness",
    "VoxRagHarness",
    "Pipeline",
    "RagHarness",
)
_METHOD_NAMES = ("run_text", "run_query", "run", "arun", "handle", "ask", "query", "answer")
_FUNC_NAMES = ("run_text", "run_query", "run", "handle_text", "answer_text", "ask", "pipeline")
_TEXT_PARAMS = ("text", "query", "query_text", "transcript", "utterance", "input_text", "q")
_LANG_PARAMS = ("lang", "lang_hint", "language", "detected_lang")
_MODE_PARAMS = ("mode", "generation_mode", "gen_mode")
_WARMUP_NAMES = ("warmup", "start", "prepare", "aopen")

#: Query text is echoed into the JSON artefact for auditability; the over-length adversarial
#: query is 2KB on its own, so long texts are stored truncated with their true length beside.
_JSON_TEXT_CHARS = 400


class BenchError(RuntimeError):
    """The benchmark could not be run at all (no usable harness). Distinct from a query
    failing, which is data."""


# --------------------------------------------------------------------------------------
# Harness adapter
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Candidate:
    """One concrete way of asking the harness to answer a text query."""

    label: str
    fn: Callable[..., Any]
    text_kw: str | None
    lang_kw: str | None
    mode_kw: str | None
    is_async: bool


def _bind(label: str, fn: Callable[..., Any]) -> _Candidate | None:
    """Work out how to pass (text, lang, mode) to `fn`, or None if it cannot take a query."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins and C callables have no signature
        return None

    names = set(params)
    text_kw = next((p for p in _TEXT_PARAMS if p in names), None)
    if text_kw is None:
        positional = [
            p
            for p in params.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name not in ("self",)
        ]
        if not positional:
            return None
    required_unfilled = [
        p
        for p in params.values()
        if p.default is p.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.name not in (text_kw, "self")
    ]
    # More than one mandatory argument means this is not "answer a question" -- it is some
    # other entry point that happens to share a name. Skip rather than call it blindly.
    if len(required_unfilled) > (0 if text_kw else 1):
        return None

    return _Candidate(
        label=label,
        fn=fn,
        text_kw=text_kw,
        lang_kw=next((p for p in _LANG_PARAMS if p in names), None),
        mode_kw=next((p for p in _MODE_PARAMS if p in names), None),
        is_async=inspect.iscoroutinefunction(fn),
    )


class HarnessRunner:
    """Calls the harness once per query and hands back a `PipelineResult`.

    Written as a resolver rather than a hard-coded call because `voxrag.harness` is developed
    alongside this module: the benchmark accepts a module-level coroutine, a factory function
    or a class with a `run`-shaped method, verifies the choice by making a real call, and then
    caches the winning binding so the request path costs one attribute lookup per query.
    """

    def __init__(
        self,
        candidates: Sequence[_Candidate],
        warmup_hooks: Sequence[Callable[[], Any]] = (),
        *,
        pass_lang: bool = True,
        diagnostics: Sequence[str] = (),
    ) -> None:
        if not candidates:
            detail = ("\n  " + "\n  ".join(diagnostics)) if diagnostics else ""
            raise BenchError(
                "voxrag.harness exposes no callable that takes a text query. Expected one of "
                f"{_FUNC_NAMES} at module level, or one of {_FACTORY_NAMES} returning an "
                f"object with one of {_METHOD_NAMES}.{detail}"
            )
        self._candidates = list(candidates)
        self._warmup_hooks = list(warmup_hooks)
        self._chosen: _Candidate | None = None
        self.pass_lang = pass_lang

    @property
    def label(self) -> str:
        return self._chosen.label if self._chosen else "unresolved"

    async def warmup(self) -> list[str]:
        """Run whatever start-up hook the harness offers. Failures are reported, never fatal:
        a missing Ollama or a cold Qdrant should show up as degradations in the run, not as a
        benchmark that refuses to start."""
        notes: list[str] = []
        for hook in self._warmup_hooks:
            try:
                res = hook()
                if inspect.isawaitable(res):
                    res = await res
                notes.append(f"{getattr(hook, '__qualname__', hook)}: ok")
            except Exception as exc:
                notes.append(f"{getattr(hook, '__qualname__', hook)}: {type(exc).__name__}: {exc}")
        return notes

    async def __call__(self, query: BenchQuery, mode: GenerationMode) -> PipelineResult:
        if self._chosen is not None:
            return await self._invoke(self._chosen, query, mode)
        failures: list[str] = []
        for cand in self._candidates:
            try:
                result = await self._invoke(cand, query, mode)
            except Exception as exc:
                failures.append(f"{cand.label}: {type(exc).__name__}: {exc}")
                continue
            self._chosen = cand
            return result
        raise BenchError(
            "no voxrag.harness entry point accepted a text query:\n  " + "\n  ".join(failures)
        )

    async def _invoke(
        self, cand: _Candidate, query: BenchQuery, mode: GenerationMode
    ) -> PipelineResult:
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        if cand.text_kw:
            kwargs[cand.text_kw] = query.text
        else:
            args.append(query.text)
        if cand.lang_kw and self.pass_lang:
            kwargs[cand.lang_kw] = query.lang
        if cand.mode_kw:
            kwargs[cand.mode_kw] = mode

        if cand.is_async:
            result = await cand.fn(*args, **kwargs)
        else:
            # A synchronous harness would otherwise stall the event loop and make every
            # concurrency>1 measurement a lie.
            result = await asyncio.to_thread(cand.fn, *args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, PipelineResult):
            raise TypeError(f"returned {type(result).__name__}, expected PipelineResult")
        return result


def resolve_runner(*, pass_lang: bool = True) -> HarnessRunner:
    """Discover a way to drive `voxrag.harness`."""
    try:
        module = importlib.import_module("voxrag.harness")
    except ImportError as exc:
        raise BenchError(
            f"cannot import voxrag.harness ({exc}); the benchmark drives the harness, "
            "it does not reimplement the pipeline"
        ) from exc

    candidates: list[_Candidate] = []
    warmup_hooks: list[Callable[[], Any]] = []
    diagnostics: list[str] = []

    def _take_hook(owner: Any, label: str) -> None:
        for hook_name in _WARMUP_NAMES:
            hook = getattr(owner, hook_name, None)
            if not callable(hook):
                continue
            try:
                if inspect.signature(hook).parameters:
                    continue
            except (TypeError, ValueError):
                continue
            warmup_hooks.append(hook)
            diagnostics.append(f"warmup hook: {label}.{hook_name}")
            return

    # Factories/classes first: a harness that owns preloaded models is the expected shape, and
    # a module-level convenience function usually just wraps one of these anyway.
    for name in _FACTORY_NAMES:
        factory = getattr(module, name, None)
        if factory is None or not callable(factory):
            continue
        try:
            obj = factory()
        except Exception as exc:  # needs constructor arguments we cannot invent, or it threw
            diagnostics.append(f"{name}(): {type(exc).__name__}: {exc}")
            continue
        if inspect.isawaitable(obj):
            # A factory that must be awaited cannot be built here (there is no guaranteed
            # running loop); close it rather than leaking an un-awaited coroutine.
            with contextlib.suppress(Exception):
                obj.close()  # type: ignore[union-attr]
            diagnostics.append(f"{name}(): awaitable factory, not usable synchronously")
            continue
        for method_name in _METHOD_NAMES:
            method = getattr(obj, method_name, None)
            if method is None or not callable(method):
                continue
            bound = _bind(f"{name}().{method_name}", method)
            if bound is not None:
                candidates.append(bound)
        _take_hook(obj, f"{name}()")

    for name in _FUNC_NAMES:
        fn = getattr(module, name, None)
        if fn is None or not callable(fn) or inspect.isclass(fn):
            continue
        bound = _bind(name, fn)
        if bound is not None:
            candidates.append(bound)

    _take_hook(module, "voxrag.harness")

    return HarnessRunner(candidates, warmup_hooks, pass_lang=pass_lang, diagnostics=diagnostics)


#: Signature of anything that can stand in for the harness (used by tests and by --compare,
#: which reuses one resolved runner across both modes).
Runner = Callable[[BenchQuery, GenerationMode], Awaitable[PipelineResult]]


# --------------------------------------------------------------------------------------
# Per-query record
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class QueryRecord:
    """Everything the report needs about one query, latency and correctness alike."""

    index: int
    text: str
    lang: str
    expected: str
    category: str
    source: str
    #: Wall clock measured by the benchmark, independent of the harness's instrumentation.
    wall_ms: float
    ok: bool
    error: str | None = None
    request_id: str = ""
    refused: bool = False
    refusal_reason: str | None = None
    groundedness: float = 0.0
    answer_chars: int = 0
    n_retrieved: int = 0
    top_score: float | None = None
    retries: int = 0
    degradations: list[str] = field(default_factory=list)
    stages: dict[str, float] = field(default_factory=dict)
    #: Rebuilt from the PipelineResult so telemetry.report can be reused verbatim.
    trace: Trace | None = field(default=None, repr=False, compare=False)

    @property
    def correct(self) -> bool:
        """Did the system do what the query's tag says it should?"""
        if not self.ok:
            return False
        expect_refuse = self.expected == ExpectedBehaviour.REFUSE.value
        return self.refused is expect_refuse

    def to_dict(self) -> dict[str, Any]:
        text = self.text if len(self.text) <= _JSON_TEXT_CHARS else self.text[:_JSON_TEXT_CHARS]
        return {
            "index": self.index,
            "text": text,
            "text_chars": len(self.text),
            "text_truncated": len(self.text) > _JSON_TEXT_CHARS,
            "lang": self.lang,
            "expected": self.expected,
            "category": self.category,
            "source": self.source,
            "wall_ms": round(self.wall_ms, 3),
            "ok": self.ok,
            "error": self.error,
            "request_id": self.request_id,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "correct": self.correct,
            "groundedness": round(self.groundedness, 4),
            "answer_chars": self.answer_chars,
            "n_retrieved": self.n_retrieved,
            "top_score": None if self.top_score is None else round(self.top_score, 5),
            "retries": self.retries,
            "degradations": self.degradations,
            "stages": {k: round(v, 3) for k, v in self.stages.items()},
        }


def _record_from_result(
    index: int, query: BenchQuery, result: PipelineResult, wall_ms: float
) -> QueryRecord:
    return QueryRecord(
        index=index,
        text=query.text,
        lang=query.lang.value,
        expected=query.expected.value,
        category=query.category.value,
        source=query.source,
        wall_ms=wall_ms,
        ok=True,
        request_id=result.request_id,
        refused=result.answer.refused,
        refusal_reason=(
            result.answer.refusal_reason.value if result.answer.refusal_reason else None
        ),
        groundedness=result.answer.groundedness,
        answer_chars=len(result.answer.text),
        n_retrieved=len(result.retrieved),
        # `retrieved` is returned ranked, but taking the max costs nothing and survives a
        # store that ever returns unsorted candidates.
        top_score=max((sc.score for sc in result.retrieved), default=None),
        retries=result.retries,
        degradations=list(result.degradations),
        stages={t.stage.value: t.ms for t in result.timings if not t.skipped},
        # Rebuilt rather than fetched from telemetry.TRACES: at concurrency > 1 the ring
        # cannot be attributed back to a query, and a Trace is only these four fields.
        trace=Trace(
            request_id=result.request_id,
            timings=list(result.timings),
            degradations=list(result.degradations),
            retries=result.retries,
        ),
    )


# --------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------


@dataclass
class BenchRun:
    """One complete benchmark of one generation mode."""

    mode: GenerationMode
    concurrency: int
    warmup_n: int
    records: list[QueryRecord]
    warmup_records: list[QueryRecord]
    wall_s: float
    runner_label: str
    corpus: str
    composition: dict[str, Any]
    started_at: str
    warmup_notes: list[str] = field(default_factory=list)
    #: Set when the query pool was smaller than the requested run and had to be cycled.
    repeat_factor: float = 1.0
    notes: list[str] = field(default_factory=list)

    @property
    def ok_records(self) -> list[QueryRecord]:
        return [r for r in self.records if r.ok]

    @property
    def throughput_qps(self) -> float:
        return len(self.records) / self.wall_s if self.wall_s > 0 else 0.0

    def stage_report(self) -> dict[str, Any]:
        """Per-stage percentiles via telemetry.report, plus the benchmark's own wall clock."""
        rep = report([r.trace for r in self.ok_records if r.trace is not None])
        rep["stages"]["wall_bench"] = summarize([r.wall_ms for r in self.ok_records])
        return rep

    def core_values(self) -> list[float]:
        """CORE latencies, falling back to bench wall clock if the harness never sealed a
        trace. Stated in `notes` when the fallback is used -- an unmeasured CORE must not
        silently become a passing budget."""
        key = StageName.CORE.value
        core = [r.stages[key] for r in self.ok_records if key in r.stages]
        return core or [r.wall_ms for r in self.ok_records]

    def quality(self) -> dict[str, Any]:
        """Correctness-adjacent signals, so a fast-because-it-refuses pipeline is visible."""
        ok = self.ok_records
        answered = [r for r in ok if not r.refused]
        refused = [r for r in ok if r.refused]
        expect_refuse = [r for r in ok if r.expected == ExpectedBehaviour.REFUSE.value]
        expect_answer = [r for r in ok if r.expected == ExpectedBehaviour.ANSWER.value]
        scored = [r.top_score for r in ok if r.top_score is not None]

        by_category: dict[str, dict[str, Any]] = {}
        for r in ok:
            slot = by_category.setdefault(
                r.category, {"n": 0, "refused": 0, "expected": r.expected, "correct": 0}
            )
            slot["n"] += 1
            slot["refused"] += int(r.refused)
            slot["correct"] += int(r.correct)
        for slot in by_category.values():
            slot["accuracy"] = slot["correct"] / slot["n"] if slot["n"] else 0.0

        reasons: dict[str, int] = {}
        for r in refused:
            key = r.refusal_reason or "unspecified"
            reasons[key] = reasons.get(key, 0) + 1

        # Aggregated, not listed: 300 identical connection errors are one fact, and printing
        # them one per line buries the rest of the report.
        errors: dict[str, int] = {}
        for r in self.records:
            if not r.ok and r.error:
                errors[r.error] = errors.get(r.error, 0) + 1

        return {
            "n_ok": len(ok),
            "n_errors": len(self.records) - len(ok),
            "refusal_rate": len(refused) / len(ok) if ok else 0.0,
            "mean_groundedness": (
                sum(r.groundedness for r in answered) / len(answered) if answered else 0.0
            ),
            "mean_top_score": sum(scored) / len(scored) if scored else 0.0,
            "mean_retrieved": sum(r.n_retrieved for r in ok) / len(ok) if ok else 0.0,
            "adversarial_refusal_accuracy": (
                sum(r.refused for r in expect_refuse) / len(expect_refuse) if expect_refuse else 0.0
            ),
            "adversarial_n": len(expect_refuse),
            "false_refusal_rate": (
                sum(r.refused for r in expect_answer) / len(expect_answer) if expect_answer else 0.0
            ),
            "answerable_n": len(expect_answer),
            "degradations": sum(len(r.degradations) for r in ok),
            "retries": sum(r.retries for r in ok),
            "by_category": by_category,
            "refusal_reasons": reasons,
            "errors": [
                {"error": msg, "count": n}
                for msg, n in sorted(errors.items(), key=lambda kv: -kv[1])[:10]
            ],
        }

    def budget(self) -> dict[str, Any]:
        core = self.core_values()
        stats = summarize(core)
        target = settings.latency_budget_ms
        instrumented = any(StageName.CORE.value in r.stages for r in self.ok_records)
        return {
            "target_ms": target,
            "measured": "core" if instrumented else "wall_bench",
            "n": len(core),
            "within_budget_pct": (
                100.0 * sum(1 for v in core if v <= target) / len(core) if core else 0.0
            ),
            "p50": stats["p50"],
            "p70": stats["p70"],
            "p100": stats["p100"],
            "worst_ms": stats["max"],
            # P70, because P70 is the tightest percentile the task actually asks to report.
            # P100 is a single worst observation -- one GC pause or one scheduler hiccup in
            # 500 queries would flip a verdict that is supposed to describe the system, and
            # P100 is printed next to this anyway so nothing is hidden by the choice.
            "pass": bool(core) and stats["p70"] <= target,
        }

    def to_dict(self, *, include_queries: bool = True) -> dict[str, Any]:
        warm = [r.wall_ms for r in self.warmup_records if r.ok]
        payload: dict[str, Any] = {
            "mode": self.mode.value,
            "concurrency": self.concurrency,
            "started_at": self.started_at,
            "runner": self.runner_label,
            "corpus": self.corpus,
            "n_measured": len(self.records),
            "n_warmup_discarded": len(self.warmup_records),
            "warmup_discarded_from_statistics": True,
            "warmup_wall_ms": summarize(warm),
            "warmup_notes": self.warmup_notes,
            "wall_s": round(self.wall_s, 4),
            "throughput_qps": round(self.throughput_qps, 3),
            "repeat_factor": round(self.repeat_factor, 3),
            "composition": self.composition,
            "notes": self.notes,
            "budget": self.budget(),
            "latency": self.stage_report(),
            "quality": self.quality(),
        }
        if include_queries:
            payload["queries"] = [r.to_dict() for r in self.records]
            payload["warmup_queries"] = [r.to_dict() for r in self.warmup_records]
        return payload


def _coerce_mode(mode: GenerationMode | str | None) -> GenerationMode:
    if mode is None:
        return settings.default_mode
    if isinstance(mode, GenerationMode):
        return mode
    try:
        return GenerationMode(str(mode).strip().lower())
    except ValueError as exc:
        known = ", ".join(m.value for m in GenerationMode)
        raise ValueError(f"unknown generation mode {mode!r}; known modes: {known}") from exc


def _fill(queries: Sequence[BenchQuery], total: int) -> tuple[list[BenchQuery], float]:
    """Cycle the pool up to `total` queries, reporting how many times it had to repeat.

    Repeats matter to honesty, not just to completeness: an identical query hits the
    embedding LRU (`settings.embed_cache_size`) and skips the encoder entirely, so a run with
    repeat_factor 5 has a flattered P50. The factor is printed and stored.
    """
    pool = list(queries)
    if not pool or len(pool) >= total:
        return pool[:total], 1.0
    out: list[BenchQuery] = []
    while len(out) < total:
        out.extend(pool)
    return out[:total], total / len(pool)


def _split_warmup(
    queries: Sequence[BenchQuery], warmup_n: int
) -> tuple[list[BenchQuery], list[BenchQuery]]:
    """Take warmup queries that are expected to be ANSWERED.

    Warming up on refusals would be worthless: an input rail short-circuits before the
    encoder, the store or the generator are ever touched, so the models would still be cold
    when the first measured query arrives.
    """
    if warmup_n <= 0:
        return [], list(queries)
    pool = list(queries)
    picked: list[int] = []
    for i, q in enumerate(pool):
        if len(picked) >= warmup_n:
            break
        if q.expected is ExpectedBehaviour.ANSWER:
            picked.append(i)
    for i in range(len(pool)):  # pool had too few answerable queries; top up in order
        if len(picked) >= warmup_n:
            break
        if i not in picked:
            picked.append(i)
    chosen = set(picked)
    warm = [pool[i] for i in picked]
    rest = [q for i, q in enumerate(pool) if i not in chosen]
    return warm, rest


async def _run_one(
    index: int,
    query: BenchQuery,
    runner: Runner,
    mode: GenerationMode,
    sem: asyncio.Semaphore,
    timeout_s: float,
) -> QueryRecord:
    """One measured query. Never raises: a failure is a data point, not the end of the run."""
    async with sem:
        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(runner(query, mode), timeout=timeout_s)
        except TimeoutError:
            wall = (time.perf_counter() - t0) * 1000.0
            error = f"timeout after {timeout_s:.1f}s"
            result = None
        except Exception as exc:
            wall = (time.perf_counter() - t0) * 1000.0
            error = f"{type(exc).__name__}: {exc}"
            result = None
        else:
            wall = (time.perf_counter() - t0) * 1000.0
            error = None
    if result is None:
        return QueryRecord(
            index=index,
            text=query.text,
            lang=query.lang.value,
            expected=query.expected.value,
            category=query.category.value,
            source=query.source,
            wall_ms=wall,
            ok=False,
            error=error,
        )
    return _record_from_result(index, query, result, wall)


async def run_benchmark(
    n_queries: int = 500,
    concurrency: int = 1,
    mode: GenerationMode | str | None = None,
    warmup_n: int = 10,
    include_adversarial: bool = True,
    *,
    langs: Sequence[Lang | str] | None = None,
    seed: int = DEFAULT_SEED,
    docs_path: Path | str | None = None,
    runner: Runner | None = None,
    queries: Sequence[BenchQuery] | None = None,
    pass_lang: bool = True,
    console: Console | None = None,
    show_progress: bool = True,
) -> BenchRun:
    """Drive the harness over `n_queries` text queries and collect timings.

    `runner` is an injection point: it lets a test (or a future comparison against a second
    pipeline configuration) drive the same measurement machinery without a live harness.
    Left at None, the harness is discovered and probed.
    """
    resolved_mode = _coerce_mode(mode)
    n_queries = max(1, int(n_queries))
    warmup_n = max(0, int(warmup_n))
    concurrency = max(1, int(concurrency))
    err_console = console or Console(stderr=True)
    notes: list[str] = []

    pool = (
        list(queries)
        if queries is not None
        else load_queries(
            n_queries + warmup_n,
            langs,
            include_adversarial,
            docs_path=docs_path,
            seed=seed,
        )
    )
    if not pool:
        raise BenchError("query set is empty; nothing to measure")
    warm_queries, measured_pool = _split_warmup(pool, warmup_n)
    measured, repeat_factor = _fill(measured_pool, n_queries)
    if repeat_factor > 1.0:
        notes.append(
            f"query pool held only {len(measured_pool)} distinct queries for {n_queries} "
            f"slots; cycled {repeat_factor:.1f}x -- repeats hit the embedding cache and "
            "flatter P50"
        )

    the_runner: Runner
    warmup_notes: list[str] = []
    if runner is not None:
        the_runner = runner
        runner_label = getattr(runner, "label", None) or getattr(
            runner, "__qualname__", type(runner).__name__
        )
    else:
        harness_runner = resolve_runner(pass_lang=pass_lang)
        warmup_notes = await harness_runner.warmup()
        the_runner = harness_runner
        runner_label = harness_runner.label

    timeout_s = _QUERY_TIMEOUT_S.get(resolved_mode, 30.0)

    # -- warmup: sequential on purpose. Firing N cold requests concurrently would have each
    # of them pay for a model load, which measures lock contention instead of warming caches.
    serial = asyncio.Semaphore(1)
    warmup_records: list[QueryRecord] = []
    for i, q in enumerate(warm_queries):
        warmup_records.append(
            await _run_one(i, q, the_runner, resolved_mode, serial, timeout_s)
        )
    if runner is None and isinstance(the_runner, HarnessRunner):
        runner_label = the_runner.label  # now resolved by the probe call
        if warmup_records and runner_label == "unresolved":
            # Every candidate entry point rejected every warmup query. Running the other 500
            # would just collect the same error 500 times and print a table of zeros.
            raise BenchError(
                "no voxrag.harness entry point answered a warmup query:\n  "
                + "\n  ".join(dict.fromkeys(r.error or "" for r in warmup_records))
            )

    # -- measured phase
    sem = asyncio.Semaphore(concurrency)
    progress_ctx: Any = contextlib.nullcontext(None)
    if show_progress:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]bench {resolved_mode.value} c={concurrency}"),
            BarColumn(bar_width=32),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=err_console,
            transient=True,
            # 4Hz: the render thread must not compete with the loop it is measuring.
            refresh_per_second=4,
        )

    with progress_ctx as progress:
        task_id = progress.add_task("", total=len(measured)) if progress is not None else None

        async def one(index: int, q: BenchQuery) -> QueryRecord:
            rec = await _run_one(index, q, the_runner, resolved_mode, sem, timeout_s)
            if progress is not None and task_id is not None:
                progress.advance(task_id)  # after the timer stops; never inside a measurement
            return rec

        started = time.perf_counter()
        records = list(
            await asyncio.gather(*(one(i, q) for i, q in enumerate(measured)))
        )
        wall_s = time.perf_counter() - started

    run = BenchRun(
        mode=resolved_mode,
        concurrency=concurrency,
        warmup_n=len(warmup_records),
        records=records,
        warmup_records=warmup_records,
        wall_s=wall_s,
        runner_label=str(runner_label),
        corpus=corpus_source(langs, docs_path=docs_path) if queries is None else "supplied",
        composition=describe_set(measured),
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        warmup_notes=warmup_notes,
        repeat_factor=repeat_factor,
        notes=notes,
    )
    if not run.ok_records:
        run.notes.append(
            f"every one of {len(run.records)} queries failed; the latency table is empty and "
            "the budget verdict below means nothing -- fix the harness, then re-run"
        )
    elif not any(StageName.CORE.value in r.stages for r in run.ok_records):
        run.notes.append(
            "harness reported no CORE stage; the budget below is measured against the "
            "benchmark's own wall clock, which INCLUDES anything the harness does not time"
        )
    return run


async def run_compare(
    n_queries: int = 500,
    concurrency: int = 1,
    warmup_n: int = 10,
    include_adversarial: bool = True,
    **kwargs: Any,
) -> list[BenchRun]:
    """Benchmark FAST and QUALITY over the identical query set.

    The set is materialised once and passed to both runs: comparing modes over two different
    random draws would attribute sampling noise to the generator.
    """
    shared = kwargs.pop("queries", None)
    if shared is None:
        shared = load_queries(
            n_queries + warmup_n,
            kwargs.get("langs"),
            include_adversarial,
            docs_path=kwargs.get("docs_path"),
            seed=kwargs.get("seed", DEFAULT_SEED),
        )
    runs: list[BenchRun] = []
    for mode in (GenerationMode.FAST, GenerationMode.QUALITY):
        runs.append(
            await run_benchmark(
                n_queries,
                concurrency,
                mode,
                warmup_n,
                include_adversarial,
                queries=shared,
                **kwargs,
            )
        )
    return runs


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

#: Canonical stage order, plus the benchmark's own wall clock last.
_STAGE_ORDER = [s.value for s in StageName] + ["wall_bench"]
_PCT_KEYS = [f"p{p:g}" for p in settings.report_percentiles]


def _fmt_ms(v: float) -> str:
    return f"{v:,.1f}" if v >= 10 else f"{v:.2f}"


def print_run(run: BenchRun, console: Console | None = None) -> None:
    """Render one BenchRun as terminal tables."""
    out = console or Console()
    rep = run.stage_report()
    budget = run.budget()
    qual = run.quality()

    out.print()
    out.rule(f"[bold]voxrag bench — mode={run.mode.value} concurrency={run.concurrency}")
    out.print(
        f"harness=[cyan]{run.runner_label}[/]  corpus=[cyan]{run.corpus}[/]  "
        f"queries=[cyan]{len(run.records):,}[/]  unique=[cyan]"
        f"{run.composition.get('unique_texts', 0):,}[/]  "
        f"langs=[cyan]{','.join(f'{k}:{v}' for k, v in run.composition.get('by_lang', {}).items())}[/]  "
        f"started=[dim]{run.started_at}[/]"
    )
    # Both facts, every run: the discard happened, and here is what was discarded.
    warm = summarize([r.wall_ms for r in run.warmup_records if r.ok])
    out.print(
        f"[yellow]warmup:[/] first [bold]{len(run.warmup_records)}[/] queries were run "
        f"sequentially and are [bold]EXCLUDED[/] from every statistic below "
        f"(mean {_fmt_ms(warm['mean'])}ms, worst {_fmt_ms(warm['max'])}ms) — they pay for "
        f"model load, tokeniser init and index attach. "
        f"Statistics are computed over the remaining {len(run.records):,} queries."
    )

    table = Table(
        title=f"latency by stage (ms) — {len(run.ok_records):,} successful queries",
        title_style="bold",
        header_style="bold",
    )
    table.add_column("stage")
    table.add_column("n", justify="right")
    table.add_column("mean", justify="right")
    for key in _PCT_KEYS:
        table.add_column(key.upper(), justify="right")
    for name in _STAGE_ORDER:
        stats = rep["stages"].get(name)
        if not stats or not stats["n"]:
            continue
        style = "bold green" if name == StageName.CORE.value else ""
        row = [name, f"{stats['n']:,}", _fmt_ms(stats["mean"])]
        row += [_fmt_ms(stats.get(k, 0.0)) for k in _PCT_KEYS]
        table.add_row(*row, style=style)
    out.print(table)

    verdict = "[bold green]WITHIN BUDGET" if budget["pass"] else "[bold red]OVER BUDGET"
    if run.mode is GenerationMode.QUALITY and not budget["pass"]:
        verdict += " [dim](expected: a 4B abstractive model cannot answer in 200ms on CPU)"
    out.print(
        f"\n[bold]budget[/] {budget['measured'].upper()} vs {budget['target_ms']:.0f}ms  "
        f"P50 {_fmt_ms(budget['p50'])}  P70 {_fmt_ms(budget['p70'])}  "
        f"P100 {_fmt_ms(budget['p100'])}  "
        f"within budget {budget['within_budget_pct']:.1f}% of queries  {verdict}"
    )
    out.print(
        f"[bold]throughput[/] {run.throughput_qps:,.1f} queries/sec at concurrency "
        f"{run.concurrency} ({len(run.records):,} queries in {run.wall_s:.2f}s)"
    )

    signals = Table(
        title="correctness signals (latency without these is gameable)",
        title_style="bold",
        header_style="bold",
    )
    signals.add_column("signal")
    signals.add_column("value", justify="right")
    signals.add_row("refusal rate (all queries)", f"{qual['refusal_rate']:.1%}")
    signals.add_row(
        f"adversarial refusal accuracy ({qual['adversarial_n']} refuse-expected)",
        f"{qual['adversarial_refusal_accuracy']:.1%}",
    )
    signals.add_row(
        f"false refusal rate ({qual['answerable_n']} answer-expected)",
        f"{qual['false_refusal_rate']:.1%}",
    )
    signals.add_row("mean groundedness (answered)", f"{qual['mean_groundedness']:.3f}")
    signals.add_row("mean top retrieval score", f"{qual['mean_top_score']:.3f}")
    signals.add_row("mean chunks retrieved", f"{qual['mean_retrieved']:.1f}")
    signals.add_row("degradations", f"{qual['degradations']:,}")
    signals.add_row("retries", f"{qual['retries']:,}")
    signals.add_row(
        "errors", f"[red]{qual['n_errors']:,}" if qual["n_errors"] else f"{qual['n_errors']:,}"
    )
    out.print(signals)

    adversarial = {k: v for k, v in qual["by_category"].items() if k != "corpus"}
    if adversarial:
        cat = Table(title="adversarial set by category", title_style="bold", header_style="bold")
        cat.add_column("category")
        cat.add_column("expected")
        cat.add_column("n", justify="right")
        cat.add_column("refused", justify="right")
        cat.add_column("accuracy", justify="right")
        for name, slot in sorted(adversarial.items()):
            good = slot["accuracy"] >= 1.0
            cat.add_row(
                name,
                slot["expected"],
                f"{slot['n']:,}",
                f"{slot['refused']:,}",
                f"{slot['accuracy']:.0%}",
                style="green" if good else "red",
            )
        out.print(cat)
    if qual["refusal_reasons"]:
        out.print(
            "refusal reasons: "
            + ", ".join(f"{k}={v}" for k, v in sorted(qual["refusal_reasons"].items()))
        )
    for note in run.notes:
        out.print(f"[yellow]note:[/] {note}")
    for err in qual["errors"]:
        out.print(f"[red]error x{err['count']}:[/] {err['error']}")


def print_comparison(runs: Sequence[BenchRun], console: Console | None = None) -> None:
    """FAST vs QUALITY side by side over the same query set."""
    out = console or Console()
    if len(runs) < 2:
        return
    table = Table(
        title="mode comparison (same query set, same order)",
        title_style="bold",
        header_style="bold",
    )
    table.add_column("metric")
    for run in runs:
        table.add_column(run.mode.value, justify="right")

    budgets = [r.budget() for r in runs]
    quals = [r.quality() for r in runs]
    reps = [r.stage_report() for r in runs]

    def row(label: str, values: Iterable[str], style: str = "") -> None:
        table.add_row(label, *values, style=style)

    row("CORE P50 (ms)", [_fmt_ms(b["p50"]) for b in budgets], "bold")
    row("CORE P70 (ms)", [_fmt_ms(b["p70"]) for b in budgets], "bold")
    row("CORE P100 (ms)", [_fmt_ms(b["p100"]) for b in budgets], "bold")
    row("within 200ms", [f"{b['within_budget_pct']:.1f}%" for b in budgets])
    row(
        "generate P50 (ms)",
        [_fmt_ms(rep["stages"].get("generate", {}).get("p50", 0.0)) for rep in reps],
    )
    row("throughput (q/s)", [f"{r.throughput_qps:,.1f}" for r in runs])
    row("refusal rate", [f"{q['refusal_rate']:.1%}" for q in quals])
    row("adversarial accuracy", [f"{q['adversarial_refusal_accuracy']:.1%}" for q in quals])
    row("mean groundedness", [f"{q['mean_groundedness']:.3f}" for q in quals])
    row("degradations", [f"{q['degradations']:,}" for q in quals])
    row("errors", [f"{q['n_errors']:,}" for q in quals])
    out.print()
    out.print(table)


def write_report(
    runs: Sequence[BenchRun], path: Path | str | None = None, *, include_queries: bool = True
) -> Path:
    """Write the full raw benchmark to `data/bench_<timestamp>.json`.

    Written to a `.tmp` sibling and renamed, so an interrupted run cannot leave a truncated
    JSON that looks like a real result.
    """
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(path) if path is not None else DATA_DIR / f"bench_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "voxrag.bench/1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "budget_ms": settings.latency_budget_ms,
        "percentiles": list(settings.report_percentiles),
        "input": "text (STT excluded by design: vendor network time is not the retrieval path)",
        "config": {
            "vector_store": settings.vector_store,
            "active_strategy": settings.active_strategy.value,
            "embed_model": settings.embed_model_repo,
            "embed_dim": settings.embed_dim,
            "retrieve_top_k": settings.retrieve_top_k,
            "lexical_top_k": settings.lexical_top_k,
            "fuse_top_k": settings.fuse_top_k,
            "context_top_k": settings.context_top_k,
            "enable_lexical": settings.enable_lexical,
            "enable_rerank": settings.enable_rerank,
            "enable_guardrails": settings.enable_guardrails,
            "hnsw_ef_search": settings.hnsw_ef_search,
            # Guardrail thresholds belong in the artefact for the same reason the retrieval
            # settings do: refusal rate and adversarial accuracy are reported right beside the
            # percentiles, and a stored run that does not say which thresholds produced them
            # cannot be compared with another. Two runs were once read as contradicting each
            # other when they had simply been measured at different grounding thresholds.
            "min_retrieval_score": settings.min_retrieval_score,
            "min_groundedness": settings.min_groundedness,
            "min_entity_grounding": settings.min_entity_grounding,
            "min_dense_similarity": settings.min_dense_similarity,
        },
        "runs": [r.to_dict(include_queries=include_queries) for r in runs],
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return out


__all__ = [
    "BenchError",
    "BenchRun",
    "HarnessRunner",
    "QueryRecord",
    "Runner",
    "print_comparison",
    "print_run",
    "resolve_runner",
    "run_benchmark",
    "run_compare",
    "write_report",
]
