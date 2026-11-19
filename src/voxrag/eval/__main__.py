"""Run every evaluation and print one consolidated report.

    python -m voxrag.eval                      # everything
    python -m voxrag.eval --skip answer        # guardrails + retrieval only
    python -m voxrag.eval --per-lang 40

Guardrails run first and need no index, so a broken or still-building index still produces
that half of the report rather than nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console

from voxrag.contracts import GenerationMode, Lang


async def _run(args: argparse.Namespace) -> int:
    console = Console()
    report: dict[str, Any] = {}
    skip = set(args.skip.split(",")) if args.skip else set()

    if "guardrail" not in skip:
        from voxrag.eval.guardrail import _report as guardrail_report
        from voxrag.eval.guardrail import run as run_guardrails

        console.rule("[bold]guardrails")
        report["guardrail"] = await run_guardrails()
        guardrail_report(report["guardrail"])

    if {"retrieval", "answer"} - skip:
        from voxrag.eval.qrels import balanced_sample, load_qrels

        langs = [Lang(x) for x in args.langs.split(",")] if args.langs else None
        queries, stats = load_qrels(args.docs, langs=langs)
        sample = balanced_sample(queries, args.per_lang)
        console.print(
            f"\n[dim]{stats.total_queries:,} graded queries available; evaluating "
            f"{len(sample):,} ({args.per_lang}/language)[/dim]"
        )

        if "retrieval" not in skip:
            from voxrag.eval.retrieval import _print_table
            from voxrag.eval.retrieval import evaluate as eval_retrieval

            console.rule("[bold]retrieval")
            rows = [
                await eval_retrieval(sample, mode=mode, concurrency=args.concurrency)
                for mode in args.modes.split(",")
            ]
            report["retrieval"] = rows
            _print_table(rows)

        if "answer" not in skip:
            from voxrag.eval.answer import _report as answer_report
            from voxrag.eval.answer import evaluate as eval_answer

            console.rule("[bold]answers")
            report["answer"] = await eval_answer(
                sample, mode=GenerationMode(args.mode), concurrency=args.concurrency
            )
            answer_report(report["answer"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"\nwrote {out}")

    guard = report.get("guardrail")
    return 1 if guard and (guard["false_positives"] or guard["missed"]) else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="data/docs14.jsonl")
    parser.add_argument("--langs", default=None)
    parser.add_argument("--per-lang", type=int, default=25)
    parser.add_argument("--modes", default="dense,lexical,hybrid")
    parser.add_argument("--mode", default="fast", help="generation mode for the answer eval")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--skip", default="", help="comma-separated: guardrail,retrieval,answer")
    parser.add_argument("--out", default="data/eval_report.json")
    return asyncio.run(_run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
