"""Corpus ingest: ai4bharat/MSMARCO-XI parquet -> contracts.Document.

The rest of the system only ever sees `Document`s, from `iter_documents` (fresh from the
HF parquet) or `load_jsonl` (from a materialised `data/docs.jsonl`). Index builders should
prefer the JSONL: it is the reproducibility boundary, since it pins exactly which passages
survived dedup and length filtering for a given run.

Re-exports are resolved lazily (PEP 562) for two reasons: `python -m voxrag.ingest.msmarco`
would otherwise warn about importing the module twice, and importing `voxrag.ingest` from
the request path must not drag pyarrow and huggingface_hub into a serving process that
never touches the corpus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from voxrag.ingest.msmarco import (
        MIN_PASSAGE_CHARS,
        IngestStats,
        LangStats,
        as_lang,
        count_jsonl,
        iter_documents,
        load_jsonl,
        make_doc_id,
        remote_filename,
        resolve_parquet,
        save_jsonl,
    )

__all__ = [
    "MIN_PASSAGE_CHARS",
    "IngestStats",
    "LangStats",
    "as_lang",
    "count_jsonl",
    "iter_documents",
    "load_jsonl",
    "make_doc_id",
    "remote_filename",
    "resolve_parquet",
    "save_jsonl",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from voxrag.ingest import msmarco

    return getattr(msmarco, name)


def __dir__() -> list[str]:
    return sorted(__all__)
