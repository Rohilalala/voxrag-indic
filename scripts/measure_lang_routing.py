"""Measure what the pipeline actually does with each language when nobody tells it the language.

The voice path has no reliable language tag: Sarvam returns one, but a user can switch the UI
language, and the benchmark's `--no-lang-hint` mode exists precisely because a hint hides
detection errors. This script runs real corpus queries through the harness with NO hint and
reports, per language:

  detected      share of queries whose detected language is the true one
  retrieved     share of retrieved chunks that are in the TRUE language of the query
  refused       share of queries the pipeline declined to answer
  top_score     mean fused confidence of the best hit

A language that detects wrong retrieves from the wrong partition, which shows up here as a
low `retrieved` and a high `refused` -- the two numbers together are what separates "the
guardrail is too strict" from "retrieval was handed the wrong index".
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from voxrag.config import DATA_DIR
from voxrag.contracts import GenerationMode, Lang

DEFAULT_POOL = DATA_DIR / "queries_by_lang.json"


async def measure(
    pools: dict[str, list[str]], per_lang: int, seed: int, mode: GenerationMode
) -> dict[str, dict[str, Any]]:
    from voxrag.harness import get_harness

    harness = get_harness()
    await harness.warmup()

    rng = random.Random(seed)
    out: dict[str, dict[str, Any]] = {}
    for code in sorted(pools):
        try:
            true_lang = Lang(code)
        except ValueError:
            continue
        queries = rng.sample(pools[code], k=min(per_lang, len(pools[code])))
        detected_ok = 0
        refused = 0
        chunk_hits = 0
        chunk_total = 0
        top_scores: list[float] = []
        detected_as: Counter = Counter()
        reasons: Counter = Counter()
        for query in queries:
            result = await harness.run(text=query, mode=mode)
            detected_as[result.detected_lang.value if result.detected_lang else "none"] += 1
            detected_ok += int(result.detected_lang is true_lang)
            refused += int(result.answer.refused)
            if result.answer.refused and result.answer.refusal_reason:
                reasons[result.answer.refusal_reason.value] += 1
            for scored in result.retrieved:
                chunk_total += 1
                chunk_hits += int(scored.chunk.lang is true_lang)
            if result.retrieved:
                top_scores.append(max(sc.score for sc in result.retrieved))
        n = len(queries)
        out[code] = {
            "n": n,
            "detected_accuracy": detected_ok / n if n else 0.0,
            "retrieved_lang_purity": chunk_hits / chunk_total if chunk_total else 0.0,
            "refusal_rate": refused / n if n else 0.0,
            "mean_top_score": sum(top_scores) / len(top_scores) if top_scores else 0.0,
            "detected_as": dict(detected_as.most_common()),
            "refusal_reasons": dict(reasons.most_common()),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--per-lang", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--mode", default=GenerationMode.FAST.value)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    pools = json.loads(args.pool.read_text(encoding="utf-8"))
    report = asyncio.run(measure(pools, args.per_lang, args.seed, GenerationMode(args.mode)))

    print(f"{'lang':5s} {'n':>4s} {'detected':>9s} {'retrieved':>10s} {'refused':>8s} {'top':>6s}  top detection")
    for code, row in report.items():
        top_detect = ", ".join(f"{k}:{v}" for k, v in list(row["detected_as"].items())[:3])
        print(
            f"{code:5s} {row['n']:4d} {row['detected_accuracy']:9.1%} "
            f"{row['retrieved_lang_purity']:10.1%} {row['refusal_rate']:8.1%} "
            f"{row['mean_top_score']:6.3f}  {top_detect}"
        )
    macro = {
        key: sum(row[key] for row in report.values()) / len(report)
        for key in ("detected_accuracy", "retrieved_lang_purity", "refusal_rate")
    }
    print(
        f"\nmacro-average over {len(report)} languages: detected {macro['detected_accuracy']:.1%}, "
        f"retrieved-in-language {macro['retrieved_lang_purity']:.1%}, refused {macro['refusal_rate']:.1%}"
    )
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
