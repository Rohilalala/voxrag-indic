"""Streaming loader for ai4bharat/MSMARCO-XI -> contracts.Document.

The real on-disk schema (verified 2026-08-15 against validation/hinval.parquet, not the
README, which is wrong about several fields)::

    source_lang: string                     # "en"
    target_lang: string                     # "hi"
    meta: struct<frequency_penalty, max_tokens, model_name,
                 presence_penalty, temperature, top_p>   # translation-LLM params
    Answer: string                          # translated gold answer
    query_id: int64
    query_type: string                      # description | numeric | entity | ...
    passages: struct<
        English_passages:    list<string>,
        Translated_passages: list<string>,
        is_selected:         list<int64>>
    Eng_Query: string
    Eng_Answer: string
    query: string                           # translated query

Two things the task brief assumes that the corpus does NOT have: there is no `url` and no
`title` column anywhere, so `Document.url`/`Document.title` are always None. And the three
`passages` child lists are parallel arrays, not a list of structs -- one Document per index
into them.

Layout note, and the reason this file is written the way it is: every parquet here is a
SINGLE row group (validation/hin = 97,941 rows, 461 MB on disk, 1.16 GB uncompressed).
Measured on a 122 MB/286 MB-row-group fixture, pyarrow's batch reader materialises the
*selected column chunks* of a row group up front and then slices batches out of them -- it
does not stream pages -- so the projection is the only lever on peak memory. Hence:
we never call `read()`, we ask for the two `passages` leaf columns by dotted path so
`English_passages` (~40% of the bytes, unused) is never decoded, we process one language at
a time, and we close the file between languages. Peak RSS on that fixture: 217 MB with the
leaf projection vs 256 MB with the whole `passages` struct vs an outright `read()` of
everything. Scaled to hinval that is roughly 0.7 GB rather than 1.9 GB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from voxrag.config import settings
from voxrag.contracts import LANG_FILE_STEM, Document, Lang

# --------------------------------------------------------------------------------------
# Module-level constants. These are ingest-shape facts, not runtime tunables, so they are
# deliberately not in config.Settings.
# --------------------------------------------------------------------------------------

#: Passages shorter than this cannot answer anything and they wreck BM25's average-document
#: -length term, which inflates the score of every other short document in the index.
MIN_PASSAGE_CHARS = 20

#: Rows decoded per arrow batch. Small enough that one batch of ~8 passages/row stays well
#: under a megabyte, large enough that per-batch python overhead is amortised.
BATCH_ROWS = 256

#: blake2b digest_size=8 -> exactly 16 hex chars, as required for doc_id.
DOC_ID_BYTES = 8

#: Dedup keys are 16-byte digests, not the passage strings themselves: at ~170k passages per
#: language the string set costs >130 MB while the digest set costs ~11 MB, and the collision
#: probability at 2^128 is far below the probability of a bit flip in the same run.
DEDUP_DIGEST_BYTES = 16

#: Only the columns we actually emit. Naming the two `passages` children explicitly keeps
#: `English_passages` off the wire and out of memory.
PARQUET_COLUMNS = [
    "query",
    "Answer",
    "query_id",
    "query_type",
    "passages.Translated_passages",
    "passages.is_selected",
]

#: Same list without nested paths, for the pyarrow builds that reject dotted selection.
PARQUET_COLUMNS_FLAT = ["query", "Answer", "query_id", "query_type", "passages"]

#: split -> (directory, filename suffix). `validation/hinval.parquet`, `train/hintrain.parquet`.
SPLIT_LAYOUT: dict[str, tuple[str, str]] = {
    "train": ("train", "train"),
    "validation": ("validation", "val"),
}
SPLIT_ALIASES: dict[str, str] = {
    "val": "validation",
    "valid": "validation",
    "dev": "validation",
    "test": "validation",  # the repo ships no test split; validation is the eval set.
}

#: Telugu is the one language with a validation file but NO train file in the repo. Asking
#: for train/teltrain.parquet is a 404, not a transient error, so it must not be retried.
MISSING_TRAIN_LANGS = frozenset({Lang.TEL})


# --------------------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------------------


@dataclass
class LangStats:
    """Per-language ingest counters. `docs` is what actually reached the index."""

    rows: int = 0
    passages: int = 0
    short: int = 0
    duplicates: int = 0
    docs: int = 0
    chars: int = 0
    error: str | None = None

    @property
    def mean_chars(self) -> float:
        return self.chars / self.docs if self.docs else 0.0

    @property
    def dedup_rate(self) -> float:
        """Fraction of length-valid passages dropped as exact duplicates."""
        considered = self.passages - self.short
        return self.duplicates / considered if considered else 0.0


@dataclass
class IngestStats:
    """Aggregate counters, filled in as `iter_documents` is consumed."""

    per_lang: dict[str, LangStats] = field(default_factory=dict)
    started_at: float = field(default_factory=time.perf_counter)

    def lang(self, lang: Lang) -> LangStats:
        return self.per_lang.setdefault(lang.value, LangStats())

    def _sum(self, attr: str) -> int:
        return sum(getattr(s, attr) for s in self.per_lang.values())

    @property
    def rows(self) -> int:
        return self._sum("rows")

    @property
    def passages(self) -> int:
        return self._sum("passages")

    @property
    def short(self) -> int:
        return self._sum("short")

    @property
    def duplicates(self) -> int:
        return self._sum("duplicates")

    @property
    def docs(self) -> int:
        return self._sum("docs")

    @property
    def chars(self) -> int:
        return self._sum("chars")

    @property
    def mean_chars(self) -> float:
        return self.chars / self.docs if self.docs else 0.0

    @property
    def dedup_rate(self) -> float:
        considered = self.passages - self.short
        return self.duplicates / considered if considered else 0.0

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started_at

    @property
    def errors(self) -> dict[str, str]:
        return {k: s.error for k, s in self.per_lang.items() if s.error}


# --------------------------------------------------------------------------------------
# Resolution / download
# --------------------------------------------------------------------------------------


def normalize_split(split: str) -> str:
    key = split.strip().lower()
    key = SPLIT_ALIASES.get(key, key)
    if key not in SPLIT_LAYOUT:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLIT_LAYOUT)}")
    return key


def as_lang(value: Lang | str) -> Lang:
    """Accept `Lang.HIN`, "hi", "HIN" or the HF stem "hin" -- CLI input is all three."""
    if isinstance(value, Lang):
        return value
    token = value.strip().lower()
    try:
        return Lang(token)
    except ValueError:
        pass
    for lang, stem in LANG_FILE_STEM.items():
        if stem == token:
            return lang
    raise ValueError(f"unknown language {value!r}; expected an ISO code like 'hi' or a stem 'hin'")


def remote_filename(lang: Lang, split: str) -> str:
    """`hi`, `validation` -> `validation/hinval.parquet`."""
    stem = LANG_FILE_STEM.get(lang)
    if stem is None:
        raise ValueError(f"{lang.value} has no file in {settings.hf_dataset} (English is the pivot)")
    directory, suffix = SPLIT_LAYOUT[normalize_split(split)]
    return f"{directory}/{stem}{suffix}.parquet"


def resolve_parquet(lang: Lang, split: str, *, local_dir: Path | None = None) -> Path:
    """Return a local path to the language's parquet, downloading it once if needed.

    hf_hub_download is content-addressed and resumable, so a second call is a no-op even
    across processes. Retries are bounded and only cover transport errors -- a 404 (Telugu
    has no train file) is re-raised immediately rather than retried three times.
    """
    filename = remote_filename(lang, split)
    if local_dir is not None:
        # Accept both the repo layout (validation/hinval.parquet) and a flat directory of
        # basenames, because the practical way to fetch these is `curl -L -C -` per file:
        # hf_hub_download wedges on this repo -- it parks at a chunk boundary with live
        # sockets and no throughput -- and a resumable curl is the reliable escape hatch.
        for candidate in (Path(local_dir) / filename, Path(local_dir) / Path(filename).name):
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
    if normalize_split(split) == "train" and lang in MISSING_TRAIN_LANGS:
        raise FileNotFoundError(f"{settings.hf_dataset} ships no {filename} (train split missing)")

    last: Exception | None = None
    for attempt in range(settings.max_retries + 1):
        try:
            return Path(
                hf_hub_download(
                    settings.hf_dataset,
                    filename,
                    repo_type="dataset",
                    # Public dataset: token is optional and None is the anonymous path.
                    token=settings.hf_token or None,
                )
            )
        except (FileNotFoundError, ValueError):
            # Includes hub LocalEntryNotFoundError (offline, nothing cached). Neither that
            # nor a bad filename gets better by waiting.
            raise
        except Exception as exc:  # transport, rate limit, partial file
            # Matched by name: importing hub's error classes couples this module to a
            # huggingface_hub version that has moved them twice.
            name = type(exc).__name__
            if "EntryNotFound" in name or "RepositoryNotFound" in name:
                raise
            last = exc
            if attempt < settings.max_retries:
                time.sleep(settings.retry_backoff_ms / 1000.0 * (2**attempt))
    raise RuntimeError(f"could not fetch {filename}: {last}") from last


def _open_parquet(path: Path) -> tuple[pq.ParquetFile, list[str]]:
    """Open the file and pick the narrowest column projection this file supports.

    Selecting `passages.Translated_passages` rather than `passages` halves the bytes
    decoded, because the sibling `English_passages` list is never used here. The leaf paths
    are checked rather than assumed: a file written by a different parquet writer names the
    list element `item` instead of `element`, and a wrong projection is a hard error.
    """
    pf = pq.ParquetFile(path, memory_map=True)
    names = {f.name for f in pf.schema_arrow}
    if "passages" not in names:
        raise ValueError(f"{path.name} has no `passages` column; schema is {sorted(names)}")
    leaves = [pf.schema.column(i).path for i in range(len(pf.schema))]
    wanted = ("passages.Translated_passages.", "passages.is_selected.")
    if all(any(leaf.startswith(prefix) for leaf in leaves) for prefix in wanted):
        return pf, list(PARQUET_COLUMNS)
    return pf, list(PARQUET_COLUMNS_FLAT)


# --------------------------------------------------------------------------------------
# Row -> Document
# --------------------------------------------------------------------------------------


def make_doc_id(lang: Lang, row_idx: int, passage_idx: int) -> str:
    """Stable across runs and machines: the index is rebuilt constantly and chunk ids,
    qdrant point ids and cached eval judgements all hang off this value."""
    key = f"{lang.value}:{row_idx}:{passage_idx}".encode()
    return hashlib.blake2b(key, digest_size=DOC_ID_BYTES).hexdigest()


def _clean(text: str) -> str:
    """NFC-normalise and strip.

    NFC matters for Indic retrieval: Devanagari nukta forms, Kannada/Telugu vowel signs and
    Urdu Arabic-presentation forms have multiple canonical encodings, and a BM25 term only
    matches if both sides normalise the same way. Everything downstream measures character
    offsets against `Document.text`, so normalising here (once, offline) is the only place
    it can be done without breaking citation spans.
    """
    return unicodedata.normalize("NFC", text).strip()


def _iter_lang_documents(
    lang: Lang,
    path: Path,
    split: str,
    limit: int | None,
    stats: IngestStats,
) -> Iterator[Document]:
    ls = stats.lang(lang)
    pf, columns = _open_parquet(path)
    seen: set[bytes] = set()
    row_idx = 0
    try:
        for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=columns):
            queries = batch.column("query").to_pylist()
            answers = batch.column("Answer").to_pylist()
            query_ids = batch.column("query_id").to_pylist()
            query_types = batch.column("query_type").to_pylist()
            passage_col = batch.column("passages").to_pylist()

            for i in range(batch.num_rows):
                this_row = row_idx
                row_idx += 1
                ls.rows += 1
                bundle = passage_col[i] or {}
                texts = bundle.get("Translated_passages") or []
                flags = bundle.get("is_selected") or []
                query = queries[i]
                answer = answers[i]
                meta_base = {
                    "query_id": query_ids[i],
                    "query_type": query_types[i],
                    "split": split,
                    "row_idx": this_row,
                }

                for p_idx, raw in enumerate(texts):
                    ls.passages += 1
                    if not raw:
                        ls.short += 1
                        continue
                    text = _clean(raw)
                    if len(text) < MIN_PASSAGE_CHARS:
                        ls.short += 1
                        continue
                    digest = hashlib.blake2b(
                        text.encode("utf-8"), digest_size=DEDUP_DIGEST_BYTES
                    ).digest()
                    if digest in seen:
                        # MS MARCO reuses the same passage across many queries; keeping the
                        # first occurrence keeps output deterministic under a fixed row order.
                        ls.duplicates += 1
                        continue
                    seen.add(digest)

                    ls.docs += 1
                    ls.chars += len(text)
                    yield Document(
                        doc_id=make_doc_id(lang, this_row, p_idx),
                        text=text,
                        lang=lang,
                        # No title/url column exists in this corpus -- see module docstring.
                        title=None,
                        url=None,
                        source_query=query or None,
                        gold_answer=answer or None,
                        is_selected=bool(flags[p_idx]) if p_idx < len(flags) else False,
                        meta={**meta_base, "passage_idx": p_idx},
                    )
                    if limit is not None and ls.docs >= limit:
                        return
    finally:
        pf.close()


def iter_documents(
    langs: Sequence[Lang | str] | None = None,
    limit_per_lang: int | None = None,
    split: str | None = None,
    *,
    stats: IngestStats | None = None,
    local_dir: Path | None = None,
) -> Iterator[Document]:
    """Stream MSMARCO-XI as Documents, one per passage.

    `limit_per_lang` caps emitted **Documents** per language, not source rows: it is the
    number that bounds index size and embedding cost, and the loop stops reading the parquet
    the moment it is hit. (config.ingest_limit_per_lang's comment says "rows"; every row
    carries ~10 passages, so treating it as rows would silently produce a 10x larger index
    and a 10x longer embedding run.)

    `None` for any of langs/limit/split means "take the configured default"; pass
    `limit_per_lang=0` for genuinely unlimited.

    A language whose file cannot be fetched is recorded in `stats` and skipped -- one dead
    download must not take the other languages down with it.
    """
    requested = langs if langs is not None else settings.ingest_langs
    split_name = normalize_split(split or settings.ingest_split)
    limit = limit_per_lang if limit_per_lang is not None else settings.ingest_limit_per_lang
    if limit is not None and limit <= 0:
        limit = None
    st = stats if stats is not None else IngestStats()

    for raw_lang in requested:
        lang = as_lang(raw_lang)
        ls = st.lang(lang)
        try:
            path = resolve_parquet(lang, split_name, local_dir=local_dir)
        except Exception as exc:
            ls.error = f"{type(exc).__name__}: {exc}"
            continue
        try:
            yield from _iter_lang_documents(lang, path, split_name, limit, st)
        except Exception as exc:  # corrupt/truncated parquet: keep the languages we did get
            ls.error = f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------------------
# JSONL
# --------------------------------------------------------------------------------------


def save_jsonl(docs: Iterable[Document], path: Path | str) -> int:
    """Write Documents as one JSON object per line. Returns the number written.

    Writes to a sibling `.tmp` and renames, so an interrupted ingest can never leave a
    half-written docs file that the index builder would happily read.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    n = 0
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            for doc in docs:
                fh.write(doc.model_dump_json())
                fh.write("\n")
                n += 1
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return n


def load_jsonl(path: Path | str) -> Iterator[Document]:
    """Stream Documents back. A malformed line is skipped, not fatal: a 500k-line corpus
    should not be unusable because one line was truncated by a full disk."""
    src = Path(path)
    with src.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Document.model_validate_json(line)
            except Exception as exc:
                print(f"[ingest] skipping {src.name}:{lineno}: {exc}", file=sys.stderr)


def count_jsonl(path: Path | str) -> dict[str, int]:
    """Documents per language in an existing docs file, without building Documents."""
    counts: dict[str, int] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            lang = json.loads(line).get("lang", "?")
            counts[lang] = counts.get(lang, 0) + 1
    return counts


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _summary_table(stats: IngestStats) -> Any:
    from rich.table import Table

    table = Table(title="MSMARCO-XI ingest", title_style="bold", header_style="bold")
    table.add_column("lang")
    table.add_column("rows", justify="right")
    table.add_column("passages", justify="right")
    table.add_column("docs", justify="right")
    table.add_column("short", justify="right")
    table.add_column("dupes", justify="right")
    table.add_column("dedup", justify="right")
    table.add_column("mean chars", justify="right")
    for code, ls in stats.per_lang.items():
        table.add_row(
            f"[red]{code} !" if ls.error else code,
            f"{ls.rows:,}",
            f"{ls.passages:,}",
            f"{ls.docs:,}",
            f"{ls.short:,}",
            f"{ls.duplicates:,}",
            f"{ls.dedup_rate:.1%}",
            f"{ls.mean_chars:.0f}",
        )
    table.add_section()
    table.add_row(
        "[bold]all",
        f"{stats.rows:,}",
        f"{stats.passages:,}",
        f"{stats.docs:,}",
        f"{stats.short:,}",
        f"{stats.duplicates:,}",
        f"{stats.dedup_rate:.1%}",
        f"{stats.mean_chars:.0f}",
    )
    return table


def _run(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    console = Console(stderr=True)
    raw_langs = args.langs.split(",") if args.langs else settings.ingest_langs
    langs = [as_lang(x) for x in raw_langs if str(x).strip()]
    split = normalize_split(args.split or settings.ingest_split)
    # Resolve the limit HERE, not in iter_documents: there, None means "use the configured
    # default", so an explicit `--limit 0` (unlimited) has to be forwarded as 0, never None.
    if args.limit is None:
        limit = settings.ingest_limit_per_lang
    elif args.limit <= 0:
        limit = None
    else:
        limit = args.limit
    stats = IngestStats()

    console.print(
        f"[bold]{settings.hf_dataset}[/] split=[cyan]{split}[/] "
        f"langs=[cyan]{','.join(x.value for x in langs)}[/] "
        f"limit/lang=[cyan]{limit if limit else 'all'}[/] -> [green]{args.out}[/]"
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[lang]}"),
        BarColumn(bar_width=32),
        MofNCompleteColumn(),
        TextColumn("docs  {task.fields[note]}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

    def tracked() -> Iterator[Document]:
        """Wrap the stream so the bar advances per document and per language."""
        with progress:
            tasks: dict[str, int] = {}
            for lang in langs:
                # The bar cannot move during the first-run download (~460MB per language),
                # so say so rather than looking hung for several minutes.
                tasks[lang.value] = progress.add_task(
                    "", total=limit, lang=lang.value, note="[dim]fetching parquet", completed=0
                )
            forwarded = limit if limit is not None else 0
            for doc in iter_documents(
                langs, forwarded, split, stats=stats, local_dir=args.local_dir
            ):
                code = doc.lang.value
                ls = stats.per_lang[code]
                progress.update(
                    tasks[code],
                    completed=ls.docs,
                    note=f"rows={ls.rows:,} dup={ls.duplicates:,}",
                )
                yield doc
            for lang in langs:  # a failed language must not sit at 0% forever
                ls = stats.lang(lang)
                if ls.error:
                    progress.update(tasks[lang.value], note="[red]FAILED", completed=0, total=None)

    written = save_jsonl(tracked(), args.out)

    console.print(_summary_table(stats))
    rate = written / stats.elapsed_s if stats.elapsed_s else 0.0
    console.print(
        f"wrote [bold green]{written:,}[/] docs to [green]{args.out}[/] "
        f"in {stats.elapsed_s:.1f}s ({rate:,.0f} docs/s), "
        f"dedup {stats.dedup_rate:.1%}, mean {stats.mean_chars:.0f} chars"
    )
    for code, err in stats.errors.items():
        console.print(f"[red]{code}: {err}")
    if written == 0:
        # Never leave a 0-byte docs.jsonl behind: the index builder would read it happily
        # and produce an empty, silently useless index.
        Path(args.out).unlink(missing_ok=True)
        console.print("[red]no documents written")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m voxrag.ingest.msmarco",
        description="Stream ai4bharat/MSMARCO-XI passages into a Document JSONL.",
    )
    parser.add_argument(
        "--langs",
        default=",".join(x.value for x in settings.ingest_langs),
        help="comma-separated ISO codes or HF stems, e.g. hi,ta,bn",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=settings.ingest_limit_per_lang,
        help="max documents (passages) per language; 0 or negative means all",
    )
    parser.add_argument("--split", default=settings.ingest_split, help="validation | train")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/docs.jsonl"),
        help="output JSONL path",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="read pre-staged <split>/<stem><suffix>.parquet from here instead of downloading",
    )
    return _run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
