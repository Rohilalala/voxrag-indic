"""Paired held-out confirm (committed because config comments cite its numbers): one retrieval cache, several candidate configs, bootstrap CI.

Reuses scripts/tune_extractive.py's own caching and scoring path so the numbers are
comparable with its --confirm output; adds per-query F1 so the baseline/candidate
difference can be tested as a PAIRED difference instead of two independent means.
"""

from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tune_extractive as tune

from voxrag.config import settings
from voxrag.contracts import Lang
from voxrag.eval.answer import token_f1
from voxrag.eval.qrels import balanced_sample, load_qrels
from voxrag.generate import extractive as ex

VARIANTS: dict[str, dict[str, object]] = {
    "baseline (current tree)": {},
    "full package (8 knobs)": {
        "settings.extractive_semantic_weight": 0.45,
        "settings.extractive_semantic_candidates": 8,
        "ex._MAX_ANSWER_SENTENCES": 3,
        "ex._NEIGHBOUR_BONUS": 0.35,
        "ex._NEIGHBOUR_RELATIVE": 0.55,
        "ex._RETRIEVAL_PRIOR_FLOOR": 0.3,
        "ex._BM25_B": 0.3,
        "ex._MIN_LEX_COVERAGE": 0.2,
    },
    "_RETRIEVAL_PRIOR_FLOOR 0.6->0.3": {"ex._RETRIEVAL_PRIOR_FLOOR": 0.3},
    "_BM25_B 0.6->0.3": {"ex._BM25_B": 0.3},
}


def per_query(cached) -> tuple[list[float], int]:
    generator = ex.ExtractiveGenerator()
    f1s, refused = [], 0
    for query, hits in cached:
        answer = generator.answer(query.query, hits, lang=query.lang)
        gold = query.gold_answer or ""
        if not gold:
            continue
        f1, _exact = token_f1(answer.text, gold)
        f1s.append(0.0 if answer.refused else f1)
        refused += 1 if answer.refused else 0
    return f1s, refused


def bootstrap(deltas: list[float], reps: int = 10000, seed: int = 7) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(reps):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


def randomization_p(deltas: list[float], reps: int = 10000, seed: int = 11) -> float:
    rng = random.Random(seed)
    observed = abs(sum(deltas) / len(deltas))
    hits = 0
    for _ in range(reps):
        flipped = sum(d if rng.random() < 0.5 else -d for d in deltas) / len(deltas)
        hits += 1 if abs(flipped) >= observed - 1e-12 else 0
    return (hits + 1) / (reps + 1)


async def main() -> None:
    queries, _stats = load_qrels("data/docs14.jsonl", langs=[Lang(x) for x in Lang])
    sample = [q for q in balanced_sample(queries, 40, seed=1) if q.gold_answer]
    print(f"{len(sample)} queries with gold answers (seed 1); caching retrieval...", flush=True)
    cached = await tune.cache_contexts(sample, 4)

    originals = {
        name: tune._get(name)
        for spec in VARIANTS.values()
        for name in spec
    }
    results: dict[str, list[float]] = {}
    for label, spec in VARIANTS.items():
        for name, value in originals.items():
            tune._set(name, value)
        for name, value in spec.items():
            tune._set(name, value)
        f1s, refused = per_query(cached)
        results[label] = f1s
        print(f"{label:34s} f1={sum(f1s)/len(f1s):.4f}  refused={refused}/{len(f1s)}", flush=True)
    for name, value in originals.items():
        tune._set(name, value)

    base = results["baseline (current tree)"]
    print(f"\ncontext_top_k={settings.context_top_k}  n={len(base)}")
    for label, f1s in results.items():
        if label == "baseline (current tree)":
            continue
        deltas = [c - b for c, b in zip(f1s, base, strict=True)]
        mean = sum(deltas) / len(deltas)
        lo, hi = bootstrap(deltas)
        changed = sum(1 for d in deltas if abs(d) > 1e-12)
        print(
            f"{label:34s} delta {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
            f"p={randomization_p(deltas):.4f}  queries changed {changed}/{len(deltas)}"
        )


asyncio.run(main())
