"""Does a tuned configuration actually beat the current one? Held-out, paired, repeated.

`voxrag.eval.sweep` picks a configuration by scoring candidates on ONE sample -- the same
sample it tunes on -- and its winning margin is routinely smaller than the run-to-run spread
of an identical configuration (0.0328 nDCG@10, measured; see eval/retrieval.py). A number
produced that way is a hypothesis, not a result. This script tests it:

  * on a DIFFERENT sample from the one the sweep tuned on (a different `balanced_sample`
    seed), so the configuration cannot win by having been fitted to these queries;
  * REPEATED and INTERLEAVED (A, B, A, B, ...), so a cold cache or a warming index cannot be
    attributed to whichever configuration happened to run second;
  * PAIRED per query, then run through both the bootstrap and the randomization test in
    `voxrag.eval.significance`, which agree before anything counts as significant.

    .venv/bin/python scripts/validate_config.py --per-lang 40 --repeats 3 \
        --set dense_weight=0.5 --set rrf_k=10 --set hnsw_ef_search=192
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from voxrag.config import settings
from voxrag.contracts import Lang
from voxrag.eval.qrels import balanced_sample, load_qrels
from voxrag.eval.retrieval import _retrieve, score_ranking
from voxrag.eval.significance import compare


def _coerce(current: Any, raw: str) -> Any:
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


async def _score_all(queries: list, mode: str, top_k: int, concurrency: int) -> list[float]:
    """Per-query nDCG@10, in query order."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(query) -> float:
        async with semaphore:
            try:
                hits = await _retrieve(mode, query, strategy=None, top_k=top_k)
            except Exception:
                return 0.0
        ranked = [hit.chunk.doc_id for hit in hits]
        _recall, ndcg, _first = score_ranking(ranked, query.relevant)
        return ndcg

    return list(await asyncio.gather(*(one(q) for q in queries)))


async def _run(args: argparse.Namespace) -> int:
    langs = [Lang(x) for x in args.langs.split(",")] if args.langs else None
    queries, _stats = load_qrels(args.docs, langs=langs)
    sample = balanced_sample(queries, args.per_lang, seed=args.sample_seed)
    print(f"{len(sample)} held-out queries (balanced_sample seed={args.sample_seed})")

    candidate = {}
    for item in args.set:
        name, _, raw = item.partition("=")
        name = name.strip()
        if not hasattr(settings, name):
            raise SystemExit(f"unknown setting {name!r}")
        candidate[name] = _coerce(getattr(settings, name), raw)
    baseline = {name: getattr(settings, name) for name in candidate}
    print(f"baseline  {baseline}")
    print(f"candidate {candidate}\n")

    def apply(config: dict[str, Any]) -> None:
        for name, value in config.items():
            setattr(settings, name, value)

    # Interleaved repeats: A, B, A, B... Order effects hit both configurations equally.
    runs: dict[str, list[list[float]]] = {"baseline": [], "candidate": []}
    for repeat in range(args.repeats):
        for label, config in (("baseline", baseline), ("candidate", candidate)):
            apply(config)
            scores = await _score_all(sample, args.mode, args.top_k, args.concurrency)
            runs[label].append(scores)
            mean = sum(scores) / len(scores)
            print(f"  repeat {repeat + 1} {label:9s} ndcg@10 {mean:.4f}")
    apply(baseline)

    # Average each query across repeats before pairing: one query's score under one
    # configuration, with run-order noise averaged out rather than treated as a sample.
    def averaged(label: str) -> list[float]:
        rows = runs[label]
        return [sum(r[i] for r in rows) / len(rows) for i in range(len(sample))]

    result = compare("baseline", averaged("baseline"), "candidate", averaged("candidate"))
    print(f"\n{result.verdict}")
    print(
        f"\nmean nDCG@10 baseline {result.mean_a:.4f} -> candidate {result.mean_b:.4f} "
        f"(delta {result.mean_delta:+.4f}, 95% CI [{result.ci_low:+.4f}, {result.ci_high:+.4f}], "
        f"bootstrap p={result.bootstrap_p:.4f}, randomization p={result.randomization_p:.4f}, "
        f"significant={result.significant})"
    )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "baseline": {k: str(v) for k, v in baseline.items()},
                    "candidate": {k: str(v) for k, v in candidate.items()},
                    "n_queries": len(sample),
                    "repeats": args.repeats,
                    "per_repeat_means": {
                        label: [sum(r) / len(r) for r in rows] for label, rows in runs.items()
                    },
                    "mean_baseline": result.mean_a,
                    "mean_candidate": result.mean_b,
                    "delta": result.mean_delta,
                    "ci": [result.ci_low, result.ci_high],
                    "bootstrap_p": result.bootstrap_p,
                    "randomization_p": result.randomization_p,
                    "significant": result.significant,
                    "verdict": result.verdict,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwritten to {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="data/docs14.jsonl")
    parser.add_argument("--langs", default=None)
    parser.add_argument("--per-lang", type=int, default=40)
    parser.add_argument("--sample-seed", type=int, default=1, help="0 is the sweep's own sample")
    parser.add_argument("--mode", default="hybrid", choices=("dense", "lexical", "hybrid"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--set", action="append", default=[], help="name=value, repeatable")
    parser.add_argument("--out", default=None)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
