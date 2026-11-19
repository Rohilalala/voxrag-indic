"""Chunking strategies and the registry that selects between them.

Eight strategies, all real, all benchmarked against each other by
`python -m voxrag.chunking.evaluate`. `settings.active_strategy` picks the one that builds
the served index; the rest exist so that choice is evidence-based rather than asserted.

Importing this package is cheap: it pulls in `tokenizers` and `regex` only. SEMANTIC
resolves the embedder lazily on first use, so onnxruntime is never loaded on behalf of a
strategy that is not running.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from voxrag.chunking.base import (
    BaseChunker,
    Span,
    count_tokens,
    split_sentences,
    tokenizer_available,
    truncate_to_tokens,
)
from voxrag.chunking.fixed import FixedChunker, FixedOverlapChunker
from voxrag.chunking.metadata_aware import MetadataAwareChunker
from voxrag.chunking.parent_child import ParentChildChunker
from voxrag.chunking.passage_native import PassageNativeChunker
from voxrag.chunking.recursive import RecursiveChunker
from voxrag.chunking.semantic import SemanticChunker
from voxrag.chunking.sentence import SentenceWindowChunker
from voxrag.config import settings
from voxrag.contracts import Chunk, Chunker, ChunkStrategyName, Document

#: Chunkers are stateless and deterministic, so one shared instance each is correct and
#: avoids re-allocating per document during a multi-million-chunk index build.
REGISTRY: dict[ChunkStrategyName, Chunker] = {
    ChunkStrategyName.FIXED: FixedChunker(),
    ChunkStrategyName.FIXED_OVERLAP: FixedOverlapChunker(),
    ChunkStrategyName.SENTENCE_WINDOW: SentenceWindowChunker(),
    ChunkStrategyName.SEMANTIC: SemanticChunker(),
    ChunkStrategyName.RECURSIVE: RecursiveChunker(),
    ChunkStrategyName.PASSAGE_NATIVE: PassageNativeChunker(),
    ChunkStrategyName.PARENT_CHILD: ParentChildChunker(),
    ChunkStrategyName.METADATA_AWARE: MetadataAwareChunker(),
}


def get_chunker(name: ChunkStrategyName | str | None = None) -> Chunker:
    """Resolve a strategy name to its chunker. `None` means `settings.active_strategy`."""
    if name is None:
        name = settings.active_strategy
    if not isinstance(name, ChunkStrategyName):
        try:
            name = ChunkStrategyName(str(name))
        except ValueError as exc:
            known = ", ".join(s.value for s in ChunkStrategyName)
            raise KeyError(f"unknown chunk strategy {name!r}; known strategies: {known}") from exc
    return REGISTRY[name]


def chunk_documents(
    docs: Iterable[Document], strategy: ChunkStrategyName | str | None = None
) -> Iterator[Chunk]:
    """Stream chunks for many documents.

    A generator, not a list: an ingest run over 20k rows per language would otherwise hold
    every chunk of the corpus in memory before the first vector is written.
    """
    chunker = get_chunker(strategy)
    for doc in docs:
        yield from chunker.chunk(doc)


__all__ = [
    "REGISTRY",
    "BaseChunker",
    "FixedChunker",
    "FixedOverlapChunker",
    "MetadataAwareChunker",
    "ParentChildChunker",
    "PassageNativeChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SentenceWindowChunker",
    "Span",
    "chunk_documents",
    "count_tokens",
    "get_chunker",
    "split_sentences",
    "tokenizer_available",
    "truncate_to_tokens",
]
