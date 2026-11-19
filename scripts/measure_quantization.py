"""Measure what Qdrant quantization actually costs on THIS index, instead of assuming.

Scalar (int8) quantization is the one unexplored lever on both problems this system has left:
dense search is the dominant latency stage (measured p50 68.3ms of a 108.3ms CORE) and the
2,268,974 x 384 float32 vectors are ~3.5GB that must stay RAM-resident, which is what makes
the index expensive to deploy. Compression addresses both. It is also a RECALL change, and the
standard reassurance -- "rescoring recovers almost all of it" -- is a measurement somebody else
made on a different corpus with a different embedder.

So this script measures it here, and it measures it on a COPY. The serving collection is never
written: switching quantization re-encodes every segment in the background, which competes for
exactly the CPU the latency numbers describe, and a bad result would already be shipped.

METHOD, and what the resulting numbers do and do not support
------------------------------------------------------------
* The copy holds every point of a subset of LANGUAGES, not a random sample of points. That
  matters: MSMARCO-XI qrels grade a query against documents drawn from the same corpus, so a
  random point sample would delete relevant documents and drive recall toward zero for reasons
  that have nothing to do with quantization. A language slice keeps every candidate document
  of every retained query intact, so Recall@10 inside those languages stays meaningful.
* The copy is therefore SMALLER than the serving index, so its absolute latency is lower than
  the published 68.3ms p50 and is not comparable to it. Searches also run one at a time here,
  where the published numbers were measured with four in flight. Compare rows to each other,
  never a row to the README.
* Timing wraps `QdrantStore.search`, the same call the harness times as `retrieve_dense` --
  including payload hydration -- rather than a bare `query_points`. That is the cost the
  system actually pays.
* Every kind is scored on the SAME queries with the SAME pre-computed query vectors, so a
  difference between rows is a difference in the collection.
* The copy is forced to be INDEXED, and the script aborts if it is not. Qdrant only builds
  quantized vectors for segments it has indexed; a collection below `indexing_threshold`
  (20000 on the serving index) keeps every segment plain and answers by exact search over the
  original float32 vectors, so the quantization config is accepted, reported back by
  `get_collection` -- and completely inert. Verified here on a 3000-point throwaway
  collection: unindexed, binary quantization returned the identical top-10 as unquantized;
  with `indexing_threshold=100` the same query returned 6 of those 10. Without this guard the
  script's own output is the perfect fake result: three identical rows reading as
  "quantization costs nothing".

READ THE NOISE FLOOR BEFORE READING THE TABLE. Repeated runs of an identical configuration
over identical queries on this system produced an nDCG@10 spread of 0.0328 (stdev 0.0152). A
quality difference smaller than that is not evidence of anything -- a parameter sweep already
produced a fake +0.0274 "win" this way. The script reports the spread it observes across its
own repeats so the reader can check that assumption instead of trusting it.

Usage:
    python scripts/measure_quantization.py --source msmarco_xi14 --langs hi,ta
    python scripts/measure_quantization.py --langs all --per-lang 25 --keep
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import AsyncQdrantClient, models

from voxrag.config import settings
from voxrag.contracts import Lang
from voxrag.eval.qrels import EvalQuery, balanced_sample, load_qrels

# Imported rather than reimplemented on purpose. `_rank_documents` collapses a chunk ranking
# into a document ranking, and `score_ranking` computes Recall@k / nDCG@10 with this project's
# IDCG convention. A local copy of either would drift and silently make these numbers
# incomparable with `python -m voxrag.eval.retrieval`, which is the whole point of running this.
from voxrag.eval.retrieval import RANK_CUTOFF, _rank_documents, score_ranking
from voxrag.store.quantization import KINDS, apply_to_collection, kind_of
from voxrag.telemetry import percentile

#: nDCG@10 spread measured on this system across four repeats of an identical configuration
#: over identical queries (0.0328; stdev 0.0152). Any quality delta below this is noise.
NOISE_FLOOR_NDCG = 0.0328

#: Bytes each kind spends per stored 384-dim vector. This is arithmetic from the encoding
#: (float32 -> int8 -> 1 bit per dimension), NOT a measurement of process memory. Qdrant keeps
#: the original vectors alongside the quantized ones so rescoring can use them, so these
#: numbers describe the size of the SEARCHED copy, not the collection's total footprint.
_BYTES_PER_VECTOR = {"none": 384 * 4, "scalar": 384, "binary": 384 // 8}

#: Points per scroll page / upsert batch while copying. Large enough that a 300k-point copy is
#: a few hundred round trips, small enough that one request body with 384-dim float vectors
#: plus full chunk payloads stays a few MB.
_COPY_BATCH = 512

#: Seconds to wait for a collection to finish re-encoding after a quantization change. The
#: rebuild is asynchronous; measuring at status=yellow measures a mixture of two configs.
_GREEN_TIMEOUT_S = 900.0

#: Columns the result table needs before rich starts truncating the numbers in it.
_MIN_TABLE_WIDTH = 104


async def _wait_green(client: AsyncQdrantClient, collection: str, *, label: str) -> None:
    """Block until the optimiser has finished, or raise. Never measure a yellow collection."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _GREEN_TIMEOUT_S
    waited = 0.0
    while loop.time() < deadline:
        info = await client.get_collection(collection)
        if info.status == models.CollectionStatus.GREEN:
            if waited:
                print(f"    {label}: green after {waited:.0f}s")
            return
        if info.status == models.CollectionStatus.RED:
            raise RuntimeError(f"collection {collection} went red while {label}")
        await asyncio.sleep(2.0)
        waited += 2.0
    raise RuntimeError(f"{collection} still not green after {_GREEN_TIMEOUT_S:.0f}s ({label})")


async def _build_copy(
    client: AsyncQdrantClient,
    source: str,
    copy: str,
    langs: list[Lang] | None,
    *,
    indexing_threshold: int | None,
) -> int:
    """Create `copy` with the SAME vector/HNSW config as `source` and stream its points in.

    Vector params and HNSW params are read off the source rather than rebuilt from settings:
    if the serving index was built with different values than the config currently holds, a
    copy built from settings would be measuring that difference instead of quantization.

    `indexing_threshold` is carried over for the same reason, but it is the one knob worth
    overriding: quantized vectors only exist inside INDEXED segments, so a copy that lands
    below the threshold measures unquantized exact search under every kind's label.
    """
    info = await client.get_collection(source)
    vectors = info.config.params.vectors
    if not isinstance(vectors, models.VectorParams):
        raise RuntimeError(f"{source} uses named vectors; this script handles a single unnamed vector")

    hnsw = info.config.hnsw_config
    if await client.collection_exists(copy):
        print(f"dropping stale copy {copy}")
        await client.delete_collection(copy)

    threshold = (
        indexing_threshold
        if indexing_threshold is not None
        else info.config.optimizer_config.indexing_threshold
    )
    await client.create_collection(
        copy,
        vectors_config=models.VectorParams(size=vectors.size, distance=vectors.distance),
        hnsw_config=models.HnswConfigDiff(m=hnsw.m, ef_construct=hnsw.ef_construct),
        on_disk_payload=info.config.params.on_disk_payload,
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=threshold),
        # Always starts unquantized: the "none" row is the baseline every other row is read
        # against, so it must be measured on a collection that was never quantized.
        quantization_config=None,
    )
    for field in ("lang", "strategy"):
        await client.create_payload_index(
            copy, field, field_schema=models.PayloadSchemaType.KEYWORD, wait=True
        )

    scroll_filter = (
        None
        if langs is None
        else models.Filter(
            must=[
                models.FieldCondition(
                    key="lang", match=models.MatchAny(any=[lang.value for lang in langs])
                )
            ]
        )
    )

    copied = 0
    offset: Any = None
    while True:
        points, offset = await client.scroll(
            source,
            scroll_filter=scroll_filter,
            limit=_COPY_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break
        await client.upsert(
            copy,
            points=[
                models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                for p in points
            ],
            wait=False,
        )
        copied += len(points)
        if copied % (_COPY_BATCH * 40) == 0:
            print(f"    copied {copied:,} points")
        if offset is None:
            break

    print(f"    copied {copied:,} points; waiting for the index to build")
    await _wait_green(client, copy, label="building the copy")
    print(f"    indexing_threshold {threshold:,} (source: {info.config.optimizer_config.indexing_threshold:,})")
    return copied


async def _indexed_vectors(client: AsyncQdrantClient, collection: str) -> int:
    """Vectors sitting in indexed segments. Zero means quantization cannot be in play."""
    info = await client.get_collection(collection)
    return int(info.indexed_vectors_count or 0)


async def _measure_kind(
    store: Any,
    client: AsyncQdrantClient,
    copy: str,
    kind: str,
    pairs: Sequence[tuple[EvalQuery, np.ndarray]],
    *,
    repeats: int,
    warmup: int,
    top_k: int,
) -> dict[str, Any]:
    """Apply `kind` to the copy, then score every query `repeats` times over it."""
    await apply_to_collection(client, copy, kind)
    await _wait_green(client, copy, label=f"re-encoding as {kind}")
    info = await client.get_collection(copy)
    on_server = kind_of(info.config.quantization_config)
    if on_server != kind:
        raise RuntimeError(f"asked for {kind}, collection reports {on_server}")
    # Re-checked per kind, not once: a re-encode rebuilds the index, and a collection that
    # came back green with plain segments would report this kind while searching float32.
    indexed = int(info.indexed_vectors_count or 0)
    if kind != "none" and indexed == 0:
        raise RuntimeError(
            f"{copy} reports quantization={kind} but has 0 indexed vectors, so the quantized "
            "vectors do not exist and every search is exact float32 -- the row would be a lie"
        )

    # The store reads this per search, so the measured path is the serving path.
    store.quantization = kind

    # Discarded: the first queries against a freshly re-encoded collection pay for paging in
    # segments the previous configuration never touched, and charging that to the kind under
    # test is how a warm-up gets mistaken for a result.
    for query, vector in pairs[:warmup]:
        await store.search(vector, top_k=top_k, lang=query.lang)

    latencies: list[float] = []
    recalls: list[dict[int, float]] = []
    first_ranks: list[int] = []
    ndcg_per_repeat: list[float] = []
    empty = 0

    for _ in range(repeats):
        run_ndcg: list[float] = []
        for query, vector in pairs:
            started = time.perf_counter()
            hits = await store.search(vector, top_k=top_k, lang=query.lang)
            latencies.append((time.perf_counter() - started) * 1000.0)
            if not hits:
                empty += 1
            recall, ndcg, first_rank = score_ranking(_rank_documents(hits), query.relevant)
            recalls.append(recall)
            first_ranks.append(first_rank)
            run_ndcg.append(ndcg)
        ndcg_per_repeat.append(sum(run_ndcg) / len(run_ndcg))

    n = len(recalls)
    return {
        "kind": kind,
        "indexed_vectors": indexed,
        "n_queries": len(pairs),
        "n_measurements": n,
        "repeats": repeats,
        "empty_results": empty,
        "latency_ms": {
            "p50": percentile(latencies, 50.0),
            "p90": percentile(latencies, 90.0),
            "p100": percentile(latencies, 100.0),
            "mean": sum(latencies) / len(latencies),
        },
        "recall@1": sum(r.get(1, 0.0) for r in recalls) / n,
        "recall@10": sum(r.get(10, 0.0) for r in recalls) / n,
        f"mrr@{RANK_CUTOFF}": sum(
            (1.0 / fr if 0 < fr <= RANK_CUTOFF else 0.0) for fr in first_ranks
        )
        / n,
        f"ndcg@{RANK_CUTOFF}": sum(ndcg_per_repeat) / len(ndcg_per_repeat),
        "ndcg_spread_across_repeats": max(ndcg_per_repeat) - min(ndcg_per_repeat),
        "ndcg_per_repeat": ndcg_per_repeat,
    }


def _print_table(rows: list[dict[str, Any]], *, points: int) -> None:
    from rich.console import Console
    from rich.table import Table

    # Rich assumes 80 columns whenever stdout is not a tty -- which is the normal case here,
    # since a measurement run gets piped into a log. Eleven numeric columns do not fit in 80,
    # and rich's response is to truncate the digits ("2.5..."), turning the one artefact this
    # script exists to produce into decoration. Widen instead.
    console = Console()
    if console.width < _MIN_TABLE_WIDTH:
        console = Console(width=_MIN_TABLE_WIDTH)
    baseline = next((r for r in rows if r["kind"] == "none"), None)
    table = Table(
        title="Qdrant quantization on a copy of the collection "
        "(sequential searches, one language slice -- rows compare only to each other)"
    )
    for column in (
        "kind",
        "vec bytes",
        "search p50",
        "p90",
        "p100",
        "R@1",
        "R@10",
        "MRR@10",
        "nDCG@10",
        "d nDCG",
        "repeat spread",
    ):
        table.add_column(column, justify="left" if column == "kind" else "right")

    for row in rows:
        lat = row["latency_ms"]
        delta = (
            "--"
            if baseline is None or row["kind"] == "none"
            else f"{row['ndcg@10'] - baseline['ndcg@10']:+.4f}"
        )
        table.add_row(
            row["kind"],
            f"{_BYTES_PER_VECTOR[row['kind']] * points / 1e9:.2f} GB",
            f"{lat['p50']:.1f}ms",
            f"{lat['p90']:.1f}ms",
            f"{lat['p100']:.1f}ms",
            f"{row['recall@1']:.3f}",
            f"{row['recall@10']:.3f}",
            f"{row['mrr@10']:.3f}",
            f"{row['ndcg@10']:.4f}",
            delta,
            f"{row['ndcg_spread_across_repeats']:.4f}",
        )
    console.print(table)

    console.print(
        "\n[bold]How to read the d nDCG column.[/bold] This system's measured noise floor is "
        f"an nDCG@10 spread of {NOISE_FLOOR_NDCG:.4f} across repeats of an IDENTICAL "
        "configuration (stdev 0.0152, four repeats). A delta smaller than that -- in either "
        "direction -- is not evidence that quantization helped or hurt. It is the same size as "
        "the measurement itself, and treating it as a result is how a parameter sweep on this "
        "project once produced a fake +0.0274 win. The 'repeat spread' column is this run's "
        "own noise; if it is comparable to the deltas, the run has not separated the kinds and "
        "needs more repeats or more queries, not an interpretation."
    )
    console.print(
        "[bold]How to read the vec bytes column.[/bold] It is arithmetic from the encoding "
        "(384 float32 = 1536B, 384 int8 = 384B, 384 bits = 48B) times the copied point count, "
        "not a measurement of process memory. Qdrant retains the original float32 vectors so "
        "rescoring can use them, so this is the size of the vectors SEARCHED, not the size of "
        "the collection on the box."
    )
    indexed = ", ".join(f"{row['kind']} {row['indexed_vectors']:,}" for row in rows)
    console.print(
        "[bold]Indexed vectors per row.[/bold] " + indexed + ". Quantized vectors are only "
        "built for INDEXED segments: a row measured with 0 would be exact float32 search "
        "wearing the kind's label, which is why the script refuses to produce one."
    )
    console.print(
        "[bold]What these latencies are not.[/bold] They come from a language-sliced copy "
        "searched one query at a time. The published 68.3ms dense p50 was measured on the full "
        "2.27M-point index with four requests in flight. Do not carry a number from this table "
        "into the README."
    )


async def _main_async(args: argparse.Namespace) -> int:
    if not settings.qdrant_url:
        print("VOXRAG_QDRANT_URL is unset: embedded Qdrant is brute-force and ignores "
              "quantization entirely. Start the server (`docker compose up -d`) first.")
        return 2

    source = args.source or settings.qdrant_collection
    copy = args.copy or f"{source}_qtest"
    # The single most important guard in the file. Everything below writes to `copy`, and one
    # of those writes re-encodes every vector in it.
    if copy in (source, settings.qdrant_collection):
        print(f"refusing to run: --copy {copy!r} is a serving collection")
        return 2

    kinds = [k.strip().lower() for k in args.kinds.split(",") if k.strip()]
    unknown = [k for k in kinds if k not in KINDS]
    if unknown:
        print(f"unknown quantization kinds {unknown}; expected from {list(KINDS)}")
        return 2

    try:
        langs = (
            None
            if args.langs.strip().lower() == "all"
            else [Lang(x.strip()) for x in args.langs.split(",") if x.strip()]
        )
    except ValueError as exc:
        # `Lang` holds two-letter codes ("hi"), which is easy to get wrong from the outside.
        print(f"{exc}; --langs takes {[x.value for x in Lang]} or 'all'")
        return 2
    queries, stats = load_qrels(args.docs, langs=langs)
    sample = balanced_sample(queries, args.per_lang)
    if not sample:
        print(f"no graded queries for langs={args.langs} in {args.docs}")
        return 2

    settings.quantization_rescore = not args.no_rescore
    settings.quantization_oversampling = args.oversampling
    # Clamped so the header states what will actually happen: `pairs[:warmup]` silently
    # stops at the end of a short query set, and printing 15 next to a 4-query sample
    # describes a run nobody performed.
    warmup = min(args.warmup, len(sample))

    print(f"source     {source}")
    print(f"copy       {copy}  (dropped at the end unless --keep)")
    print(f"languages  {'all' if langs is None else ','.join(x.value for x in langs)}")
    print(
        f"queries    {len(sample):,} graded ({args.per_lang}/language) x {args.repeats} repeats, "
        f"sequential; {warmup} discarded as warm-up per kind"
    )
    print(
        f"rescore    {settings.quantization_rescore}   oversampling {settings.quantization_oversampling}"
    )
    print(f"qrels      {stats.total_queries:,} graded queries, mean {stats.mean_relevant:.2f} relevant\n")

    from voxrag.embed import get_embedder
    from voxrag.store.qdrant_store import QdrantStore

    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        # Copying streams whole segments; the store's 10s request timeout is sized for
        # per-query reads and would abort a scroll page under load.
        timeout=300,
    )
    store = QdrantStore(url=settings.qdrant_url, collection=copy)
    rows: list[dict[str, Any]] = []
    points = 0

    try:
        print(f"building {copy} from {source}")
        points = await _build_copy(
            client, source, copy, langs, indexing_threshold=args.indexing_threshold
        )
        if points == 0:
            print("the copy is empty -- does the source hold these languages?")
            return 2

        indexed = await _indexed_vectors(client, copy)
        print(f"    indexed vectors {indexed:,} of {points:,}")
        if indexed == 0:
            print(
                f"\nrefusing to measure: {copy} has {points:,} points but 0 of them are in an\n"
                "indexed segment, which means Qdrant is answering by exact search over the\n"
                "original float32 vectors. Quantization is only built for indexed segments, so\n"
                "every kind would return identical results and the table would read as\n"
                "'quantization is free'. Copy more languages (--langs), or lower the threshold\n"
                "with --indexing-threshold (the serving index uses "
                f"{(await client.get_collection(source)).config.optimizer_config.indexing_threshold:,})."
            )
            return 2

        # Embedded once, up front, and reused for every kind: the comparison must not include
        # the encoder's own run-to-run variation, and re-embedding per kind would add it.
        embedder = get_embedder()
        vectors = await embedder.encode_async([q.query for q in sample], is_query=True)
        pairs = list(zip(sample, list(vectors), strict=True))

        for kind in kinds:
            print(f"\n  {kind}")
            row = await _measure_kind(
                store,
                client,
                copy,
                kind,
                pairs,
                repeats=args.repeats,
                warmup=warmup,
                top_k=args.top_k,
            )
            rows.append(row)
            lat = row["latency_ms"]
            print(
                f"    p50 {lat['p50']:.1f}ms  p90 {lat['p90']:.1f}ms  p100 {lat['p100']:.1f}ms  "
                f"R@10 {row['recall@10']:.3f}  nDCG@10 {row['ndcg@10']:.4f}"
            )
    finally:
        await store.close()
        if args.keep:
            print(f"\nkeeping {copy} (--keep); delete it when done, it is a full second index")
        else:
            try:
                await client.delete_collection(copy)
                print(f"\ndropped {copy}")
            except Exception as exc:
                print(f"\ncould not drop {copy}: {exc} -- delete it manually")
        await client.close()

    if not rows:
        return 1

    print()
    _print_table(rows, points=points)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "source": source,
                "copy_points": points,
                "langs": "all" if langs is None else [x.value for x in langs],
                "n_queries": len(sample),
                "repeats": args.repeats,
                "top_k": args.top_k,
                "indexing_threshold": args.indexing_threshold,
                "rescore": settings.quantization_rescore,
                "oversampling": settings.quantization_oversampling,
                "hnsw_ef_search": settings.hnsw_ef_search,
                "noise_floor_ndcg": NOISE_FLOOR_NDCG,
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", default=None, help="collection to copy (default: the configured one)")
    parser.add_argument("--copy", default=None, help="throwaway collection name (default: <source>_qtest)")
    parser.add_argument(
        "--langs",
        # Two-letter `Lang` values, not ISO 639-3: "hi,ta", never "hin,tam".
        default="hi,ta",
        help="languages to copy WHOLE, or 'all'. A language slice keeps every candidate "
        "document of its queries; a random point sample would not.",
    )
    parser.add_argument("--docs", default="data/docs14.jsonl", help="JSONL of ingested Documents")
    parser.add_argument("--per-lang", type=int, default=40, help="graded queries per language")
    parser.add_argument("--kinds", default="none,scalar,binary")
    parser.add_argument(
        "--indexing-threshold",
        type=int,
        default=None,
        help="points per segment before Qdrant indexes (and therefore quantizes) it. Default: "
        "whatever the source collection uses. Lower it when the copy is small, or the copy "
        "stays plain and quantization is inert.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="passes over the query set per kind")
    parser.add_argument("--warmup", type=int, default=15, help="queries discarded after each re-encode")
    parser.add_argument("--top-k", type=int, default=settings.retrieve_top_k)
    parser.add_argument("--oversampling", type=float, default=settings.quantization_oversampling)
    parser.add_argument(
        "--no-rescore",
        action="store_true",
        help="skip full-precision re-ranking of the quantized shortlist (cheap and lossy)",
    )
    parser.add_argument("--keep", action="store_true", help="do not drop the copy afterwards")
    parser.add_argument("--out", default="data/quantization.json")
    args = parser.parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
