"""Hyperparameter sweep driven by the retrieval evaluation.

Tunes the knobs that actually trade quality against latency -- fusion weighting, RRF's
constant, HNSW's search effort, and how many candidates survive each stage -- and picks the
best configuration by measured nDCG@10 SUBJECT TO the latency budget, rather than by nDCG
alone. A configuration that wins on quality and blows the 200ms target has not won anything.

Coordinate descent, not a full grid. The grid over five parameters would be thousands of
evaluation runs; coordinate descent sweeps one parameter at a time, keeps the winner, and
moves on. It can settle in a local optimum, which is an acceptable trade for finishing, and
the per-parameter curves it prints are more informative than a single best-of-grid number --
they show which knobs matter at all.

Every configuration is scored on the SAME query sample (balanced_sample is deterministic), so
differences are attributable to the parameter rather than to the sample.

CLI: python -m voxrag.eval.sweep --per-lang 30 --mode hybrid
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voxrag.config import settings
from voxrag.contracts import Lang
from voxrag.eval.qrels import EvalQuery, balanced_sample, load_qrels
from voxrag.eval.retrieval import evaluate

#: (setting name, candidate values). Ordered cheapest-to-riskiest so an interrupted sweep has
#: still tuned something useful.
GRID: tuple[tuple[str, tuple[Any, ...]], ...] = (
    # Fusion balance. 0.0 = pure lexical, 1.0 = pure dense. The default 0.65 is a guess that
    # this sweep exists to replace with a measurement.
    ("dense_weight", (0.35, 0.5, 0.65, 0.8, 1.0)),
    # RRF's rank-smoothing constant. 60 is the value from the original paper; smaller values
    # weight the very top ranks far more aggressively.
    ("rrf_k", (10, 30, 60, 100)),
    # HNSW search effort: the main quality/latency dial on the dense arm.
    ("hnsw_ef_search", (48, 96, 192, 384)),
    # How many candidates each arm contributes before fusion.
    ("retrieve_top_k", (10, 20, 40)),
    # How many chunks the generator finally sees.
    ("context_top_k", (2, 4, 8)),
)

#: Objective. nDCG@10 rewards putting a relevant document high, not merely somewhere in the
#: top 20, which is what a user actually experiences.
OBJECTIVE = "ndcg@10"


@dataclass
class Trial:
    setting: str
    value: Any
    score: float
    recall_at_10: float
    latency_ms: float
    within_budget: bool


async def _score(queries: Sequence[EvalQuery], mode: str, concurrency: int) -> tuple[float, float, float]:
    result = await evaluate(queries, mode=mode, concurrency=concurrency)
    overall = result["overall"]
    if not overall.get("n"):
        return (0.0, 0.0, float("inf"))
    return (overall[OBJECTIVE], overall["recall@10"], overall["mean_latency_ms"])


async def sweep(
    queries: Sequence[EvalQuery],
    *,
    mode: str = "hybrid",
    concurrency: int = 4,
    budget_ms: float | None = None,
) -> dict[str, Any]:
    """Coordinate descent over GRID. Returns the chosen configuration and every trial."""
    budget = budget_ms if budget_ms is not None else settings.latency_budget_ms
    trials: list[Trial] = []
    chosen: dict[str, Any] = {}

    base_score, base_recall, base_latency = await _score(queries, mode, concurrency)
    print(
        f"baseline: {OBJECTIVE}={base_score:.4f} recall@10={base_recall:.4f} "
        f"{base_latency:.1f}ms\n"
    )
    best_score = base_score

    for name, values in GRID:
        original = getattr(settings, name)
        best_value, best_local = original, best_score
        print(f"  {name} (current {original}):")

        for value in values:
            setattr(settings, name, value)
            try:
                score, recall, latency = await _score(queries, mode, concurrency)
            finally:
                # Restore immediately: an exception mid-trial must not leave the process
                # running with a half-applied experimental configuration.
                setattr(settings, name, original)

            within = latency <= budget
            trials.append(Trial(name, value, score, recall, latency, within))
            flag = "" if within else "  OVER BUDGET"
            marker = " *" if within and score > best_local else ""
            print(f"    {value!s:>6} -> {score:.4f}  r@10={recall:.4f}  {latency:6.1f}ms{flag}{marker}")

            # Quality only counts if it fits the budget. Without this guard the sweep happily
            # selects ef_search=384 for a hundredth of a point and quietly breaks the headline.
            if within and score > best_local:
                best_local, best_value = score, value

        setattr(settings, name, best_value)
        chosen[name] = best_value
        if best_local > best_score:
            print(f"    -> {name}={best_value} (+{best_local - best_score:.4f})\n")
            best_score = best_local
        else:
            print(f"    -> keeping {best_value} (nothing beat it)\n")

    final_score, final_recall, final_latency = await _score(queries, mode, concurrency)
    return {
        "mode": mode,
        "objective": OBJECTIVE,
        "budget_ms": budget,
        "baseline": {OBJECTIVE: base_score, "recall@10": base_recall, "latency_ms": base_latency},
        "final": {OBJECTIVE: final_score, "recall@10": final_recall, "latency_ms": final_latency},
        "improvement": final_score - base_score,
        "chosen": chosen,
        "trials": [t.__dict__ for t in trials],
    }


async def _main_async(args: argparse.Namespace) -> int:
    langs = [Lang(x) for x in args.langs.split(",")] if args.langs else None
    queries, _stats = load_qrels(args.docs, langs=langs)
    sample = balanced_sample(queries, args.per_lang)
    print(f"sweeping on {len(sample):,} queries ({args.per_lang}/language), mode={args.mode}\n")

    result = await sweep(
        sample, mode=args.mode, concurrency=args.concurrency, budget_ms=args.budget_ms
    )

    print("chosen configuration:")
    for key, value in result["chosen"].items():
        print(f"  VOXRAG_{key.upper()}={value}")
    print(
        f"\n{OBJECTIVE}: {result['baseline'][OBJECTIVE]:.4f} -> {result['final'][OBJECTIVE]:.4f} "
        f"({result['improvement']:+.4f}), latency {result['final']['latency_ms']:.1f}ms"
    )
    if result["improvement"] <= 0:
        print("no configuration beat the defaults -- report that, do not ship a null result")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="data/docs14.jsonl")
    parser.add_argument("--langs", default=None)
    parser.add_argument("--per-lang", type=int, default=30)
    parser.add_argument("--mode", default="hybrid", choices=("dense", "lexical", "hybrid"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--budget-ms", type=float, default=None)
    parser.add_argument("--out", default="data/eval_sweep.json")
    return asyncio.run(_main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
