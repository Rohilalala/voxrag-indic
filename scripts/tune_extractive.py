"""Tune the extractive answerer against gold answers, with retrieval held fixed.

The generator is the binding constraint on answer quality: token F1 is 0.208 against a
measured extractive ceiling of 0.684, so two thirds of what the retrieved evidence supports is
being left on the table by span SELECTION, not by retrieval.

Retrieval is therefore run ONCE per query and cached, and every configuration is scored over
the identical evidence. That makes a sweep cheap (retrieval is ~95% of the cost of a query)
and, more importantly, makes it a measurement of the generator alone: two configurations
cannot differ because one of them got luckier candidates.

    .venv/bin/python scripts/tune_extractive.py --per-lang 25
    .venv/bin/python scripts/tune_extractive.py --per-lang 40 --sample-seed 1 --confirm

`--confirm` scores only the current defaults against one named candidate on a different
sample, which is the check that matters before shipping anything the sweep likes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from voxrag.config import settings
from voxrag.contracts import Lang, ScoredChunk
from voxrag.eval.answer import context_ceiling, token_f1
from voxrag.eval.qrels import balanced_sample, load_qrels
from voxrag.generate import extractive as ex

#: Knobs and the values swept for each. Module constants and settings fields are both here --
#: the split between them is a code-organisation decision, not a tuning one.
GRID: dict[str, tuple[Any, ...]] = {
    "settings.context_top_k": (2, 4, 6, 8),
    "settings.extractive_semantic_weight": (0.0, 0.25, 0.45, 0.65, 0.85),
    "settings.extractive_semantic_candidates": (4, 8, 16),
    "ex._MAX_ANSWER_SENTENCES": (1, 2, 3, 4),
    "ex._NEIGHBOUR_BONUS": (0.0, 0.35, 0.7),
    "ex._NEIGHBOUR_RELATIVE": (0.35, 0.55, 0.75),
    "ex._RETRIEVAL_PRIOR_FLOOR": (0.3, 0.6, 0.9),
    "ex._BM25_B": (0.3, 0.6, 0.9),
    "ex._MIN_LEX_COVERAGE": (0.05, 0.20, 0.35),
}


def _get(name: str) -> Any:
    owner, attr = name.split(".", 1)
    return getattr(settings if owner == "settings" else ex, attr)


def _set(name: str, value: Any) -> None:
    owner, attr = name.split(".", 1)
    setattr(settings if owner == "settings" else ex, attr, value)


async def cache_contexts(queries: list, concurrency: int) -> list[tuple[Any, list[ScoredChunk]]]:
    """Retrieve once per query. Every configuration is then scored over identical evidence."""
    from voxrag.embed import get_embedder
    from voxrag.store.hybrid import get_hybrid_retriever
    from voxrag.telemetry import Trace

    embedder = get_embedder()
    retriever = get_hybrid_retriever()
    await retriever.warmup()
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(query) -> tuple[Any, list[ScoredChunk]]:
        async with semaphore:
            vector = embedder.encode([query.query], is_query=True)[0]
            hits = await retriever.retrieve(
                query.query, vector, trace=Trace(request_id="tune"), lang=query.lang
            )
        return query, hits

    return list(await asyncio.gather(*(one(q) for q in queries)))


def score(cached: list[tuple[Any, list[ScoredChunk]]]) -> dict[str, float]:
    """Mean token F1 of the generator over cached evidence, plus what explains it."""
    generator = ex.ExtractiveGenerator()
    f1s, answered, ceilings = [], 0, []
    for query, hits in cached:
        answer = generator.answer(query.query, hits, lang=query.lang)
        gold = query.gold_answer or ""
        if not gold:
            continue
        f1, _exact = token_f1(answer.text, gold)
        f1s.append(0.0 if answer.refused else f1)
        answered += 0 if answer.refused else 1
        ceilings.append(
            context_ceiling(gold, [h.chunk.generation_text for h in hits[: settings.context_top_k]])
        )
    n = max(1, len(f1s))
    return {
        "f1": sum(f1s) / n,
        "answer_rate": answered / n,
        "ceiling": sum(ceilings) / n,
        "n": float(n),
    }


def sweep(cached: list, keys: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Coordinate descent: one knob at a time, keep what wins, move on."""
    baseline = score(cached)
    print(f"baseline f1={baseline['f1']:.4f} answered={baseline['answer_rate']:.1%} "
          f"ceiling={baseline['ceiling']:.4f} over {int(baseline['n'])} queries\n")
    chosen: dict[str, Any] = {}
    best_f1 = baseline["f1"]
    trials: list[dict[str, Any]] = []

    for name in keys:
        original = _get(name)
        best_value = original
        print(f"  {name} (current {original}):")
        for value in GRID[name]:
            _set(name, value)
            row = score(cached)
            trials.append({"setting": name, "value": value, **row})
            mark = ""
            if row["f1"] > best_f1 + 1e-9:
                best_f1, best_value, mark = row["f1"], value, " *"
            print(f"      {value!s:>6s} -> f1={row['f1']:.4f}  answered={row['answer_rate']:.1%}{mark}")
        _set(name, best_value)
        chosen[name] = best_value
        if best_value != original:
            print(f"    -> {name}={best_value}")
        else:
            print("    -> unchanged")
    return chosen, trials


async def _run(args: argparse.Namespace) -> int:
    langs = [Lang(x) for x in args.langs.split(",")] if args.langs else None
    queries, _stats = load_qrels(args.docs, langs=langs)
    sample = [q for q in balanced_sample(queries, args.per_lang, seed=args.sample_seed) if q.gold_answer]
    print(f"{len(sample)} queries with gold answers (seed {args.sample_seed}); caching retrieval...")
    cached = await cache_contexts(sample, args.concurrency)

    if args.confirm:
        candidate = {}
        for item in args.set:
            name, _, raw = item.partition("=")
            current = _get(name)
            candidate[name] = type(current)(raw) if not isinstance(current, bool) else raw == "true"
        base = score(cached)
        for name, value in candidate.items():
            _set(name, value)
        cand = score(cached)
        print(f"\nbaseline  f1={base['f1']:.4f} answered={base['answer_rate']:.1%}")
        print(f"candidate f1={cand['f1']:.4f} answered={cand['answer_rate']:.1%}  "
              f"delta {cand['f1'] - base['f1']:+.4f}")
        return 0

    keys = [k for k in GRID if not args.only or k in args.only.split(",")]
    chosen, trials = sweep(cached, keys)
    final = score(cached)
    print(f"\nchosen: {json.dumps({k: str(v) for k, v in chosen.items()}, indent=2)}")
    print(f"f1 -> {final['f1']:.4f} (answered {final['answer_rate']:.1%})")
    if args.out:
        Path(args.out).write_text(
            json.dumps({"chosen": {k: str(v) for k, v in chosen.items()}, "final": final,
                        "trials": trials}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written to {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="data/docs14.jsonl")
    parser.add_argument("--langs", default=None)
    parser.add_argument("--per-lang", type=int, default=25)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--only", default=None, help="comma-separated subset of the grid")
    parser.add_argument("--confirm", action="store_true", help="score one candidate, no sweep")
    parser.add_argument("--set", action="append", default=[], help="name=value for --confirm")
    parser.add_argument("--out", default=None)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
