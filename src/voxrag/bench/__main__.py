"""CLI entry point for the latency benchmark.

    python -m voxrag.bench --n 500 --concurrency 1 --mode fast
    python -m voxrag.bench --n 500 --compare
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from voxrag.bench.latency import (
    BenchError,
    BenchRun,
    print_comparison,
    print_run,
    run_benchmark,
    run_compare,
    write_report,
)
from voxrag.bench.queries import DEFAULT_SEED
from voxrag.config import settings
from voxrag.contracts import GenerationMode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m voxrag.bench",
        description=(
            "Drive the harness over a real MSMARCO-XI query set and report P50/P70/P100 "
            "per stage, the 200ms CORE budget verdict, throughput, and the correctness "
            "signals that stop latency from being gamed."
        ),
    )
    parser.add_argument("--n", type=int, default=500, help="measured queries (default 500)")
    parser.add_argument(
        "--concurrency", type=int, default=1, help="in-flight queries; >1 measures under load"
    )
    parser.add_argument(
        "--mode",
        default=settings.default_mode.value,
        choices=[m.value for m in GenerationMode],
        help="generation mode to benchmark",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="benchmark fast and quality over the identical query set and print them side by side",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="queries run first and DISCARDED from the statistics (model load, index attach)",
    )
    parser.add_argument(
        "--langs",
        default=None,
        help=f"comma-separated codes, default {','.join(x.value for x in settings.ingest_langs)}",
    )
    parser.add_argument(
        "--no-adversarial",
        action="store_true",
        help="drop the off-topic/unsafe/injection/degenerate set (not recommended: it is the "
        "only check that the latency was not bought with refusals)",
    )
    parser.add_argument(
        "--no-lang-hint",
        action="store_true",
        help="do not pass the known language to the harness; forces its own detection path",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="query sampling seed")
    parser.add_argument(
        "--docs", type=Path, default=None, help="docs JSONL to sample queries from"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="JSON report path (default data/bench_<ts>.json)"
    )
    parser.add_argument("--no-json", action="store_true", help="skip the JSON artefact")
    parser.add_argument("--quiet", action="store_true", help="no progress bar")
    parser.add_argument(
        "--fail-over-budget",
        action="store_true",
        help="exit non-zero when the fast path misses the 200ms budget (for CI)",
    )
    return parser


async def _amain(args: argparse.Namespace) -> int:
    console = Console()
    langs = [x for x in (args.langs.split(",") if args.langs else []) if x.strip()] or None
    common = {
        "langs": langs,
        "seed": args.seed,
        "docs_path": args.docs,
        "pass_lang": not args.no_lang_hint,
        "show_progress": not args.quiet,
    }

    runs: list[BenchRun]
    if args.compare:
        runs = await run_compare(
            args.n, args.concurrency, args.warmup, not args.no_adversarial, **common
        )
    else:
        runs = [
            await run_benchmark(
                args.n,
                args.concurrency,
                args.mode,
                args.warmup,
                not args.no_adversarial,
                **common,
            )
        ]

    for run in runs:
        print_run(run, console)
    if len(runs) > 1:
        print_comparison(runs, console)

    if not args.no_json:
        path = write_report(runs, args.out)
        console.print(f"\nraw per-query timings written to [green]{path}[/]")

    if args.fail_over_budget:
        fast = [r for r in runs if r.mode is GenerationMode.FAST]
        # Only the fast path is graded against 200ms; a 4B abstractive model is expected to
        # miss it, and failing the command for that would be failing it for a stated tradeoff.
        if fast and not fast[0].budget()["pass"]:
            return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return asyncio.run(_amain(args))
    except BenchError as exc:
        Console(stderr=True).print(f"[red]benchmark could not run:[/] {exc}")
        return 2
    except KeyboardInterrupt:
        Console(stderr=True).print("[yellow]interrupted; no report written")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
