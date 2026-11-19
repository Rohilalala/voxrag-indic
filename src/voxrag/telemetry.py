"""Stage timing, trace collection, and latency percentiles.

Timing uses `time.perf_counter`, a monotonic clock, so wall-clock adjustments cannot
produce negative durations. Every stage of every request is recorded; the percentile
report the task asks for is computed from this ring buffer.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from voxrag.config import settings
from voxrag.contracts import StageName, StageTiming


@dataclass
class Trace:
    """Accumulates StageTimings for one request. Not thread-safe by design: one per request."""

    request_id: str
    timings: list[StageTiming] = field(default_factory=list)
    degradations: list[str] = field(default_factory=list)
    retries: int = 0
    _t0: float = field(default_factory=time.perf_counter)
    _stt_ms: float = 0.0

    @contextmanager
    def stage(self, name: StageName, **detail: Any) -> Iterator[dict[str, Any]]:
        """Time a block and record it.

        The yielded dict is merged into the timing's `detail`, so a stage can report
        candidate counts or cache hits it only learns about while running:

            with trace.stage(StageName.RETRIEVE_DENSE) as d:
                hits = store.search(v)
                d["candidates"] = len(hits)
        """
        extra: dict[str, Any] = dict(detail)
        start = time.perf_counter()
        try:
            yield extra
        finally:
            ms = (time.perf_counter() - start) * 1000.0
            self.timings.append(StageTiming(stage=name, ms=ms, detail=extra))
            if name is StageName.STT:
                self._stt_ms = ms

    def skip(self, name: StageName, why: str = "") -> None:
        """Record a stage that did not run, so the trace shape stays constant."""
        self.timings.append(
            StageTiming(stage=name, ms=0.0, skipped=True, detail={"why": why} if why else {})
        )

    def degrade(self, note: str) -> None:
        """Record that the harness recovered from a failure rather than erroring out."""
        self.degradations.append(note)

    def finish(self) -> list[StageTiming]:
        """Seal the trace, appending the synthetic TOTAL and CORE stages.

        CORE is TOTAL minus STT: it is the number the 200ms budget is measured against,
        because network transit to a third-party STT vendor is not "chunking + retrieval
        + everything through to final output".
        """
        total_ms = (time.perf_counter() - self._t0) * 1000.0
        self.timings.append(StageTiming(stage=StageName.TOTAL, ms=total_ms))
        self.timings.append(
            StageTiming(
                stage=StageName.CORE,
                ms=max(0.0, total_ms - self._stt_ms),
                detail={"budget_ms": settings.latency_budget_ms},
            )
        )
        return self.timings

    def ms(self, name: StageName) -> float:
        for t in self.timings:
            if t.stage is name:
                return t.ms
        return 0.0


class TraceStore:
    """Bounded, thread-safe ring of finished traces backing the /analytics endpoint."""

    def __init__(self, maxlen: int | None = None) -> None:
        self._traces: deque[Trace] = deque(maxlen=maxlen or settings.trace_ring_size)
        self._lock = threading.Lock()

    def add(self, trace: Trace) -> None:
        with self._lock:
            self._traces.append(trace)

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()

    def all(self) -> list[Trace]:
        with self._lock:
            return list(self._traces)

    def __len__(self) -> int:
        with self._lock:
            return len(self._traces)


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile.

    Nearest-rank, not linear interpolation: P100 must be an observed worst case, and an
    interpolated P100 on a short sample can understate the tail. The task asks for P100
    specifically to see the worst case, so we do not smooth it away.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if p <= 0:
        return ordered[0]
    if p >= 100:
        return ordered[-1]
    rank = max(1, min(len(ordered), int(-(-p / 100.0 * len(ordered) // 1))))
    return ordered[rank - 1]


def summarize(values: list[float], percentiles: tuple[float, ...] | None = None) -> dict[str, Any]:
    """Percentile summary for one series of millisecond measurements."""
    ps = percentiles or settings.report_percentiles
    if not values:
        return {"n": 0, "mean": 0.0, "min": 0.0, "max": 0.0, **{f"p{p:g}": 0.0 for p in ps}}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        **{f"p{p:g}": percentile(values, p) for p in ps},
    }


def report(traces: list[Trace], percentiles: tuple[float, ...] | None = None) -> dict[str, Any]:
    """Per-stage latency report across many requests.

    Returns a dict keyed by stage name, each holding n/mean/min/max/percentiles, plus a
    top-level `budget` block stating whether CORE met the target at each percentile.
    """
    by_stage: dict[str, list[float]] = {}
    for tr in traces:
        for t in tr.timings:
            if t.skipped:
                continue
            by_stage.setdefault(t.stage.value, []).append(t.ms)

    stages = {name: summarize(vals, percentiles) for name, vals in by_stage.items()}
    core = by_stage.get(StageName.CORE.value, [])
    budget = settings.latency_budget_ms
    return {
        "stages": stages,
        "budget": {
            "target_ms": budget,
            "n": len(core),
            "within_budget_pct": (
                100.0 * sum(1 for v in core if v <= budget) / len(core) if core else 0.0
            ),
            "worst_ms": max(core) if core else 0.0,
        },
        "degradations": sum(len(t.degradations) for t in traces),
        "retries": sum(t.retries for t in traces),
    }


#: Process-wide trace ring. The API and the benchmark both read from this.
TRACES = TraceStore()
