"""Build the whole BM25 index in ONE pass, after the vector index is done.

Why this is separate from `voxrag.index`:

`BM25Index` opens every language shard that already exists on disk. Building lexically once
per language therefore re-reads a growing index on every run -- quadratic I/O that reached
738MB of memory-mapped .npy files per language and pushed the indexer into uninterruptible
disk wait (0.6% CPU, ~1M pageouts) partway through the first of fourteen languages.

Chunking is cheap (~0.18ms/doc) and BM25 needs no embeddings, so rebuilding every chunk here
from scratch costs a few minutes and touches the index exactly once.

Usage: python scripts/build_bm25.py --docs data/docs14.jsonl --strategy parent_child
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from voxrag.chunking import get_chunker
from voxrag.contracts import Chunk, ChunkStrategyName
from voxrag.ingest.msmarco import load_jsonl
from voxrag.store.bm25 import get_bm25_index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="data/docs14.jsonl")
    parser.add_argument("--strategy", default="parent_child")
    parser.add_argument("--flush-every", type=int, default=250_000, help="chunks per index() call")
    args = parser.parse_args(argv)

    console = Console()
    strategy = ChunkStrategyName(args.strategy)
    chunker = get_chunker(strategy)
    index = get_bm25_index()

    started = time.perf_counter()
    per_lang: Counter[str] = Counter()
    batch: list[Chunk] = []
    total = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed:,} chunks"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"bm25 <- {strategy.value}", total=None)
        for doc in load_jsonl(Path(args.docs)):
            for chunk in chunker.chunk(doc):
                batch.append(chunk)
                per_lang[chunk.lang.value] += 1
                total += 1
            if len(batch) >= args.flush_every:
                index.index(batch)
                batch.clear()
                progress.update(task, completed=total)
        if batch:
            index.index(batch)
        progress.update(task, completed=total)

    elapsed = time.perf_counter() - started
    console.print(
        f"indexed {total:,} chunks across {len(per_lang)} languages in {elapsed:.0f}s "
        f"({total / max(elapsed, 1e-9):,.0f} chunks/s)"
    )
    console.print("  " + "  ".join(f"{k}={v:,}" for k, v in sorted(per_lang.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
