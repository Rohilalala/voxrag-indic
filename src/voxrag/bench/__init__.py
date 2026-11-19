"""Latency analytics: P50 / P70 / P100 over a real query set, plus the signals that keep
those numbers honest.

    python -m voxrag.bench --n 500 --concurrency 1 --mode fast
    python -m voxrag.bench --n 500 --compare

`latency.run_benchmark` drives the harness with TEXT input and collects one `Trace` per
query, then reports through `voxrag.telemetry.summarize`/`report`, so the benchmark and the
live `/api/analytics` endpoint compute their percentiles with the same code.

Re-exports are lazy (PEP 562): importing `voxrag.bench` from the API for a type or a helper
must not drag rich, the harness and the whole model stack into a serving process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from voxrag.bench.latency import (
        BenchError,
        BenchRun,
        QueryRecord,
        print_comparison,
        print_run,
        run_benchmark,
        run_compare,
        write_report,
    )
    from voxrag.bench.queries import (
        BenchQuery,
        Category,
        ExpectedBehaviour,
        adversarial_queries,
        describe_set,
        load_queries,
    )

__all__ = [
    "BenchError",
    "BenchQuery",
    "BenchRun",
    "Category",
    "ExpectedBehaviour",
    "QueryRecord",
    "adversarial_queries",
    "describe_set",
    "load_queries",
    "print_comparison",
    "print_run",
    "run_benchmark",
    "run_compare",
    "write_report",
]

#: Which submodule owns each name, so the lazy import pulls in only the half that is needed.
_QUERIES = frozenset(
    {
        "BenchQuery",
        "Category",
        "ExpectedBehaviour",
        "adversarial_queries",
        "describe_set",
        "load_queries",
    }
)
_LATENCY = frozenset(__all__) - _QUERIES


def __getattr__(name: str) -> Any:
    if name in _QUERIES:
        from voxrag.bench import queries

        return getattr(queries, name)
    if name in _LATENCY:
        from voxrag.bench import latency

        return getattr(latency, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
