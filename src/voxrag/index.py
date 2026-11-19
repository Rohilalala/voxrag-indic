"""End-to-end index builder: docs.jsonl -> chunks -> vectors -> vector store + BM25.

    python -m voxrag.index --docs data/docs.jsonl --strategy parent_child
    python -m voxrag.index --strategy all          # every strategy into one collection

`--strategy all` is not a convenience: `Chunk.strategy` is a filterable payload field in
both stores, so one collection can hold all eight strategies and the benchmark selects
between them at query time with a filter instead of needing eight collections and eight
rebuilds.

Two properties the build has to keep:

* **Streaming.** Documents are read, chunked, embedded and upserted in fixed-size batches.
  A 20k-doc-per-language corpus expands to millions of chunks under PARENT_CHILD; holding
  them (or their vectors -- 384 float32 each) in memory before the first write would need
  tens of GB.
* **A stable batch shape.** onnxruntime re-plans its memory arena for every distinct input
  shape it sees, so flushes are cut to exactly `--batch-size` rows and only the final
  remainder is a different size. Ragged batches make the build measurably slower.

The run writes `data/index_manifest.json`, which is what `GET /api/strategies` reads to
report which strategies are actually in the index and how many chunks each contributed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voxrag.config import DATA_DIR, settings
from voxrag.contracts import Chunk, ChunkStrategyName, Document

#: Written by a build, read by the API. Its shape is owned by this module.
MANIFEST_PATH = DATA_DIR / "index_manifest.json"

#: Default corpus produced by `python -m voxrag.ingest.msmarco`.
DEFAULT_DOCS_PATH = DATA_DIR / "docs.jsonl"

#: Manifest schema version. Bumped if the shape below ever changes, so a stale manifest
#: from an older build is ignored rather than misread by the API.
MANIFEST_VERSION = 1


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------


def load_manifest(path: Path | str | None = None) -> dict[str, Any] | None:
    """Read the manifest of the last build, or None if there is no usable one.

    Never raises: the API calls this on a request path and a missing or truncated manifest
    means "we do not know what is indexed", not "fail the request".
    """
    target = Path(path) if path is not None else MANIFEST_PATH
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("version") != MANIFEST_VERSION:
        return None
    return data


def _write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # Rename so a crashed build cannot leave a half-written manifest that the API would
    # then serve as truth.
    tmp.replace(path)


# --------------------------------------------------------------------------------------
# Corpus reading
# --------------------------------------------------------------------------------------


def _count_lines(path: Path) -> int:
    """Line count by block scan -- the progress bar's total, without parsing any JSON."""
    total = 0
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            total += block.count(b"\n")
    return total


def _iter_docs(path: Path, limit: int | None) -> Iterator[Document]:
    from voxrag.ingest import load_jsonl  # lazy: pulls pyarrow/huggingface_hub in

    for i, doc in enumerate(load_jsonl(path)):
        if limit is not None and i >= limit:
            return
        yield doc


def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{value:.1f}GB"


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------


def resolve_strategies(spec: str) -> list[ChunkStrategyName]:
    """Parse `--strategy`: a name, a comma-separated list, or `all`."""
    text = (spec or "").strip().lower()
    if text in ("all", "*"):
        return list(ChunkStrategyName)
    names: list[ChunkStrategyName] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            name = ChunkStrategyName(part)
        except ValueError as exc:
            known = ", ".join(s.value for s in ChunkStrategyName)
            raise ValueError(f"unknown strategy {part!r}; known strategies: {known}") from exc
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("no strategy selected")
    return names


async def build_index(
    docs_path: Path,
    strategies: Sequence[ChunkStrategyName],
    *,
    store_name: str | None = None,
    batch_size: int | None = None,
    limit: int | None = None,
    lexical: bool = True,
    progress: Any = None,
) -> dict[str, Any]:
    """Chunk, embed and index `docs_path` under each strategy. Returns the manifest."""
    # Imported here, not at module scope: `voxrag.api.routes` imports this module purely
    # for `load_manifest`, and must not pay for onnxruntime + qdrant-client to do it.
    from voxrag.chunking import get_chunker
    from voxrag.embed import get_embedder
    from voxrag.store import get_vector_store
    from voxrag.store.bm25 import get_bm25_index

    if not docs_path.exists():
        raise FileNotFoundError(
            f"no such docs file: {docs_path}. Build it first: "
            f"python -m voxrag.ingest.msmarco --out {docs_path}"
        )

    batch = max(1, batch_size or settings.embed_batch_size)
    started = time.perf_counter()

    embedder = get_embedder()
    if embedder.dim != settings.embed_dim:
        # The store's collection is created with settings.embed_dim; a mismatch either gets
        # rejected on upsert or -- worse, on a store that does not check -- silently indexes
        # vectors that can never be matched. Refuse before writing anything.
        raise ValueError(
            f"embedder emits {embedder.dim} dims but settings.embed_dim is "
            f"{settings.embed_dim}. Set VOXRAG_EMBED_DIM={embedder.dim} and rebuild."
        )

    store = get_vector_store(store_name)
    bm25 = get_bm25_index() if lexical else None

    total_docs = _count_lines(docs_path)
    if limit is not None:
        total_docs = min(total_docs, limit)

    per_strategy: dict[str, dict[str, Any]] = {}
    langs: dict[str, int] = {}
    total_chunks = 0

    for strategy in strategies:
        chunker = get_chunker(strategy)
        task = progress.add_task(f"{strategy.value}", total=total_docs) if progress else None
        s_started = time.perf_counter()
        s_docs = 0
        s_chunks = 0
        s_tokens = 0
        pending: list[Chunk] = []

        for doc in _iter_docs(docs_path, limit):
            pending.extend(chunker.chunk(doc))
            s_docs += 1
            if progress is not None and task is not None:
                progress.advance(task)
            while len(pending) >= batch:
                head, pending = pending[:batch], pending[batch:]
                s_tokens += await _flush(head, embedder, store, bm25)
                s_chunks += len(head)
                for chunk in head:
                    langs[chunk.lang.value] = langs.get(chunk.lang.value, 0) + 1

        if pending:
            s_tokens += await _flush(pending, embedder, store, bm25)
            s_chunks += len(pending)
            for chunk in pending:
                langs[chunk.lang.value] = langs.get(chunk.lang.value, 0) + 1

        if progress is not None and task is not None:
            progress.update(task, completed=total_docs)
        elapsed = time.perf_counter() - s_started
        total_chunks += s_chunks
        per_strategy[strategy.value] = {
            "chunks": s_chunks,
            "docs": s_docs,
            "tokens": s_tokens,
            "chunks_per_doc": round(s_chunks / s_docs, 2) if s_docs else 0.0,
            "mean_tokens": round(s_tokens / s_chunks, 1) if s_chunks else 0.0,
            "wall_s": round(elapsed, 2),
        }

    # Persist. BM25 defers its matrix build until asked, so this is where the whole
    # lexical index is actually compiled -- doing it per batch would be O(n^2).
    if bm25 is not None:
        await asyncio.to_thread(bm25.save)
    save = getattr(store, "save", None)
    if save is not None:
        await save()

    store_info = store.info() if hasattr(store, "info") else {"name": store.name}
    on_disk = _index_bytes(store_info)

    manifest: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "wall_s": round(time.perf_counter() - started, 2),
        "docs_path": str(docs_path),
        "docs": total_docs,
        "batch_size": batch,
        "dim": embedder.dim,
        "embedder": embedder.name,
        "active_strategy": settings.active_strategy.value,
        "total_chunks": total_chunks,
        "vectors": await store.count(),
        "langs": langs,
        "strategies": per_strategy,
        "store": store_info,
        "lexical": bm25.info() if bm25 is not None else None,
        "index_bytes": on_disk,
    }
    _write_manifest(manifest, MANIFEST_PATH)
    return manifest


async def _flush(chunks: list[Chunk], embedder: Any, store: Any, bm25: Any) -> int:
    """Embed one batch and write it to both indexes. Returns the batch's token count."""
    texts = [chunk.text for chunk in chunks]
    # to_thread, not a direct call: encode() is a blocking ONNX Run, and the progress bar's
    # refresh runs on this loop. It also lets the GIL be released inside onnxruntime.
    vectors = await asyncio.to_thread(embedder.encode, texts, is_query=False)
    await store.upsert(chunks, vectors)
    if bm25 is not None:
        bm25.index(chunks)  # accumulate only; no tokenisation happens here
    return sum(chunk.n_tokens for chunk in chunks)


def _index_bytes(store_info: dict[str, Any]) -> dict[str, Any]:
    """On-disk footprint of everything this build wrote.

    A server-mode Qdrant stores its segments inside its own container, so there is no local
    path to measure; the manifest says so explicitly rather than reporting a misleading 0.
    """
    sizes: dict[str, Any] = {"bm25": _dir_bytes(Path(settings.bm25_path))}
    location = store_info.get("location") or store_info.get("path")
    if isinstance(location, str) and not location.startswith(("http://", "https://")):
        sizes[str(store_info.get("name", "store"))] = _dir_bytes(Path(location))
    else:
        sizes[str(store_info.get("name", "store"))] = None  # remote: not on this disk
    sizes["total"] = sum(v for v in sizes.values() if isinstance(v, int))
    return sizes


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m voxrag.index",
        description="Chunk, embed and index a docs.jsonl corpus.",
    )
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS_PATH, help="JSONL of Documents")
    parser.add_argument(
        "--strategy",
        default=settings.active_strategy.value,
        help="chunking strategy, a comma-separated list, or 'all'",
    )
    parser.add_argument("--batch-size", type=int, default=settings.embed_batch_size)
    parser.add_argument("--limit", type=int, default=None, help="cap documents read")
    parser.add_argument(
        "--store", default=None, help="vector store backend (qdrant | qdrant_local | hnsw)"
    )
    parser.add_argument(
        "--no-lexical", action="store_true", help="skip the BM25 arm (dense-only index)"
    )
    parser.add_argument("--json", dest="as_json", action="store_true", help="emit the manifest")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    args = _parse_args(argv)
    console = Console(stderr=True)

    try:
        strategies = resolve_strategies(args.strategy)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    console.print(
        f"indexing [bold]{args.docs}[/bold] -> "
        f"{', '.join(s.value for s in strategies)} "
        f"(store={args.store or settings.vector_store}, batch={args.batch_size})"
    )

    async def run() -> dict[str, Any]:
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            return await build_index(
                args.docs,
                strategies,
                store_name=args.store,
                batch_size=args.batch_size,
                limit=args.limit,
                lexical=not args.no_lexical,
                progress=progress,
            )

    try:
        manifest = asyncio.run(run())
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    except KeyboardInterrupt:
        console.print("[yellow]interrupted; partial index left in place[/yellow]")
        return 130

    if args.as_json:
        json.dump(manifest, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    table = Table(title="index built", header_style="bold")
    table.add_column("strategy")
    table.add_column("chunks", justify="right")
    table.add_column("chunks/doc", justify="right")
    table.add_column("mean tokens", justify="right")
    table.add_column("wall s", justify="right")
    for name, row in manifest["strategies"].items():
        table.add_row(
            name,
            f"{row['chunks']:,}",
            f"{row['chunks_per_doc']:.2f}",
            f"{row['mean_tokens']:.1f}",
            f"{row['wall_s']:.1f}",
        )
    console.print(table)

    sizes = manifest["index_bytes"]
    on_disk = ", ".join(
        f"{k}={'remote' if v is None else _human_bytes(v)}" for k, v in sizes.items() if k != "total"
    )
    console.print(
        f"documents={manifest['docs']:,}  chunks={manifest['total_chunks']:,}  "
        f"vectors={manifest['vectors']:,}  dim={manifest['dim']}\n"
        f"store={manifest['store'].get('name')} ({manifest['store'].get('mode', 'n/a')})  "
        f"on disk: {on_disk}\n"
        f"wall={manifest['wall_s']:.1f}s  manifest={MANIFEST_PATH}"
    )
    if manifest["store"].get("exact"):
        console.print(
            "[yellow]this store does exact brute-force search -- fine for a dev loop, not a "
            "latency claim. Start a real Qdrant (docker compose up -d) before benchmarking."
            "[/yellow]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
