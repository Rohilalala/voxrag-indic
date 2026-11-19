"""Retrieval quality evaluation: Recall@k, MRR@10 and nDCG@10 against MSMARCO-XI qrels.

Runs the same retrieval stack the API serves, in three modes -- dense only, lexical only, and
the hybrid RRF fusion -- so the report shows what fusion is actually worth rather than
asserting it. Can be pointed at any indexed chunking strategy, which is how the eight
strategies are compared on identical queries.

CLI:
    python -m voxrag.eval.retrieval --docs data/docs14.jsonl --per-lang 40
    python -m voxrag.eval.retrieval --strategies all --modes hybrid --per-lang 25
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from voxrag.contracts import ChunkStrategyName, Lang, ScoredChunk
from voxrag.eval.qrels import EvalQuery, balanced_sample, load_qrels
from voxrag.telemetry import Trace

#: Cutoffs reported for Recall. 1 and 5 are what a user feels; 10 and 20 show headroom a
#: reranker could exploit.
RECALL_AT = (1, 5, 10, 20)
#: MRR and nDCG cutoff. 10 is the MS MARCO convention, kept so numbers are comparable to
#: published baselines rather than only to themselves.
RANK_CUTOFF = 10


@dataclass
class QueryResult:
    """Per-query outcome, retained so a bad aggregate can be traced to specific queries."""

    query: str
    lang: str
    n_relevant: int
    #: 1-based rank of the first relevant document, or 0 if none was retrieved.
    first_rank: int
    recall: dict[int, float] = field(default_factory=dict)
    ndcg: float = 0.0
    latency_ms: float = 0.0


def _rank_documents(hits: Sequence[ScoredChunk]) -> list[str]:
    """Collapse a chunk ranking into a document ranking, keeping first appearance.

    This is the single most important line in the module for a FAIR strategy comparison.
    Strategies emit wildly different chunk counts per document -- SENTENCE_WINDOW produced
    3.82 chunks/doc against PASSAGE_NATIVE's 1.00 -- so a strategy that shatters a document
    into many chunks gets many more chances to place one in the top k. Scoring at chunk level
    would reward fragmentation rather than relevance. Deduplicating to documents first makes
    every strategy answer the same question: did you surface the right DOCUMENT, and how high.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for hit in hits:
        doc_id = hit.chunk.doc_id
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


def score_ranking(ranked: Sequence[str], relevant: frozenset[str]) -> tuple[dict[int, float], float, int]:
    """Recall@k, nDCG@RANK_CUTOFF and the first relevant rank for one ranked document list."""
    recall: dict[int, float] = {}
    for k in RECALL_AT:
        hit_count = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
        recall[k] = hit_count / len(relevant) if relevant else 0.0

    # Binary-gain nDCG with the standard log2(rank+1) discount.
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:RANK_CUTOFF], start=1)
        if doc_id in relevant
    )
    # IDCG over min(|relevant|, cutoff): a query with 12 relevant documents cannot place more
    # than 10 of them in a top-10, and dividing by an unreachable ideal would cap its nDCG
    # below 1.0 no matter how perfect the ranking is.
    ideal_n = min(len(relevant), RANK_CUTOFF)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_n + 1))
    ndcg = dcg / idcg if idcg else 0.0

    first_rank = 0
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            first_rank = rank
            break

    return recall, ndcg, first_rank


async def _retrieve(
    mode: str,
    query: EvalQuery,
    *,
    strategy: ChunkStrategyName | None,
    top_k: int,
) -> list[ScoredChunk]:
    """One query through one retrieval mode. `mode` is dense | lexical | hybrid."""
    from voxrag.embed import get_embedder
    from voxrag.store import get_vector_store
    from voxrag.store.bm25 import get_bm25_index
    from voxrag.store.hybrid import HybridRetriever

    if mode == "lexical":
        index = get_bm25_index()
        return await asyncio.to_thread(
            index.search, query.query, top_k=top_k, lang=query.lang
        )

    embedder = get_embedder()
    vector = (await embedder.encode_async([query.query], is_query=True))[0]

    if mode == "dense":
        store = get_vector_store()
        return await store.search(vector, top_k=top_k, lang=query.lang, strategy=strategy)

    if mode == "hybrid":
        # A throwaway Trace: the retriever times its arms into one, and the eval reports its
        # own wall clock rather than these stage times.
        return await HybridRetriever().retrieve(
            query.query,
            vector,
            trace=Trace(request_id="eval"),
            lang=query.lang,
            strategy=strategy,
            top_k=top_k,
        )

    raise ValueError(f"unknown retrieval mode {mode!r}")


#: Queries run and DISCARDED before measurement begins.
#:
#: Without this the first run of any configuration is an outlier and every comparison built
#: on it is noise. Measured with an identical config over identical queries, four repeats:
#:     run 1  ndcg 0.2885  680.9ms   <- cold page cache, dense arm timing out
#:     run 2  ndcg 0.3166  219.5ms
#:     run 3  ndcg 0.3213  222.2ms
#:     run 4  ndcg 0.2998  251.1ms
#: a spread of 0.0328 -- larger than the entire effect a parameter sweep was reporting as a
#: win (+0.0274) and four times the largest per-parameter difference it measured (~0.008).
#: The sweep was tuning the warm-up.
_WARMUP_QUERIES = 15


async def evaluate(
    queries: Sequence[EvalQuery],
    *,
    mode: str = "hybrid",
    strategy: ChunkStrategyName | None = None,
    top_k: int = max(RECALL_AT),
    concurrency: int = 4,
    warmup: int | None = None,
) -> dict[str, Any]:
    """Score one (mode, strategy) configuration over the query set.

    The first `warmup` queries are run and their results thrown away, so a cold page cache
    is not charged to the configuration under test.
    """
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list[QueryResult] = []

    async def run_one(query: EvalQuery) -> QueryResult:
        async with semaphore:
            started = time.perf_counter()
            try:
                hits = await _retrieve(mode, query, strategy=strategy, top_k=top_k)
            except Exception:
                # A configuration that cannot serve a query scores zero for it rather than
                # aborting the sweep; the empty-result count in the summary surfaces it.
                hits = []
            elapsed = (time.perf_counter() - started) * 1000.0

        ranked = _rank_documents(hits)
        recall, ndcg, first_rank = score_ranking(ranked, query.relevant)
        return QueryResult(
            query=query.query,
            lang=query.lang.value,
            n_relevant=len(query.relevant),
            first_rank=first_rank,
            recall=recall,
            ndcg=ndcg,
            latency_ms=elapsed,
        )

    n_warmup = _WARMUP_QUERIES if warmup is None else max(0, warmup)
    if n_warmup and len(queries) > n_warmup:
        # Reuse the head of the SAME sample rather than separate queries: the point is to
        # page in the segments this configuration will actually touch.
        await asyncio.gather(*(run_one(q) for q in queries[:n_warmup]))

    results = list(await asyncio.gather(*(run_one(q) for q in queries)))
    return summarise(results, mode=mode, strategy=strategy)


def summarise(
    results: Sequence[QueryResult], *, mode: str, strategy: ChunkStrategyName | None
) -> dict[str, Any]:
    """Aggregate per-query results overall and per language."""

    def agg(rows: Sequence[QueryResult]) -> dict[str, Any]:
        if not rows:
            return {"n": 0}
        n = len(rows)
        return {
            "n": n,
            **{f"recall@{k}": sum(r.recall.get(k, 0.0) for r in rows) / n for k in RECALL_AT},
            f"mrr@{RANK_CUTOFF}": sum(
                (1.0 / r.first_rank if 0 < r.first_rank <= RANK_CUTOFF else 0.0) for r in rows
            )
            / n,
            f"ndcg@{RANK_CUTOFF}": sum(r.ndcg for r in rows) / n,
            "found_any": sum(1 for r in rows if r.first_rank > 0) / n,
            "mean_latency_ms": sum(r.latency_ms for r in rows) / n,
        }

    langs = sorted({r.lang for r in results})
    return {
        "mode": mode,
        "strategy": strategy.value if strategy else "(all indexed)",
        "overall": agg(results),
        "per_lang": {lang: agg([r for r in results if r.lang == lang]) for lang in langs},
        "worst": [
            {"query": r.query, "lang": r.lang, "first_rank": r.first_rank}
            for r in sorted(results, key=lambda r: (r.first_rank == 0, -r.first_rank))[:10]
            if r.first_rank == 0
        ],
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title="retrieval quality (document-level, MSMARCO-XI is_selected qrels)")
    for column in ("strategy", "mode", "n", "R@1", "R@5", "R@10", "R@20", "MRR@10", "nDCG@10", "ms"):
        table.add_column(column, justify="right" if column not in ("strategy", "mode") else "left")
    for row in rows:
        o = row["overall"]
        if not o.get("n"):
            continue
        table.add_row(
            row["strategy"],
            row["mode"],
            str(o["n"]),
            *(f"{o[f'recall@{k}']:.3f}" for k in RECALL_AT),
            f"{o['mrr@10']:.3f}",
            f"{o['ndcg@10']:.3f}",
            f"{o['mean_latency_ms']:.1f}",
        )
    Console().print(table)


async def _main_async(args: argparse.Namespace) -> int:
    langs = [Lang(x) for x in args.langs.split(",")] if args.langs else None
    queries, stats = load_qrels(args.docs, langs=langs)
    sample = balanced_sample(queries, args.per_lang)

    print(
        f"qrels: {stats.total_queries:,} graded queries "
        f"(mean {stats.mean_relevant:.2f} relevant of {stats.mean_candidates:.1f} candidates); "
        f"dropped {stats.dropped_no_relevant:,} with no relevant passage and "
        f"{stats.dropped_all_relevant:,} with no negatives"
    )
    print(f"evaluating {len(sample):,} queries: {args.per_lang} per language\n")

    if args.strategies == "all":
        strategies: list[ChunkStrategyName | None] = list(ChunkStrategyName)
    elif args.strategies == "served":
        strategies = [None]
    else:
        strategies = [ChunkStrategyName(s) for s in args.strategies.split(",")]

    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        for mode in args.modes.split(","):
            row = await evaluate(
                sample, mode=mode, strategy=strategy, concurrency=args.concurrency
            )
            rows.append(row)
            label = row["strategy"]
            o = row["overall"]
            if o.get("n"):
                print(
                    f"  {label:>16s} {mode:<8s} R@10={o['recall@10']:.3f} "
                    f"nDCG@10={o['ndcg@10']:.3f} found={o['found_any']:.2f}"
                )

    print()
    _print_table(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"qrels": asdict(stats), "n_evaluated": len(sample), "results": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="data/docs14.jsonl", help="JSONL of ingested Documents")
    parser.add_argument("--langs", default=None, help="comma-separated ISO codes, default all")
    parser.add_argument("--per-lang", type=int, default=40, help="graded queries per language")
    parser.add_argument(
        "--strategies",
        default="served",
        help="'served' (whatever is indexed), 'all', or a comma-separated list",
    )
    parser.add_argument("--modes", default="dense,lexical,hybrid")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", default="data/eval_retrieval.json")
    args = parser.parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
