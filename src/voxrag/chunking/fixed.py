"""Fixed-size token windows, with and without overlap.

FIXED is the control in the benchmark: it is what "just split it" produces, and every
other strategy in this package is judged against it. It is deliberately left naive --
including its short final window -- because smoothing the baseline hides exactly the
fragmentation that `evaluate.py` is meant to expose.
"""

from __future__ import annotations

from voxrag.chunking.base import BaseChunker, hard_token_split
from voxrag.config import settings
from voxrag.contracts import Chunk, ChunkStrategyName, Document


class FixedChunker(BaseChunker):
    """Contiguous `chunk_target_tokens` windows, no overlap, no boundary awareness.

    Windows are cut on *token* boundaries taken from the encoder's own offsets, so a chunk
    never splits a subword in half and the character offsets remain exact.
    """

    name = ChunkStrategyName.FIXED

    def chunk(self, doc: Document) -> list[Chunk]:
        spans = hard_token_split(doc.text, 0, len(doc.text), settings.chunk_target_tokens)
        return self._from_spans(doc, spans)


class FixedOverlapChunker(BaseChunker):
    """Fixed windows sharing `chunk_overlap_tokens` with their predecessor.

    The overlap exists for one reason: a fact stated across a window boundary ("the treaty
    was signed in | 1919 by ...") is unretrievable from either half alone. With a stride of
    target - overlap, every span of up to `overlap` tokens appears intact in some chunk.
    The cost is index size and duplicate n-grams, both quantified in `evaluate.py`.
    """

    name = ChunkStrategyName.FIXED_OVERLAP

    def chunk(self, doc: Document) -> list[Chunk]:
        target = settings.chunk_target_tokens
        # Clamped rather than trusted: overlap >= target would make the stride zero and
        # the window loop non-terminating.
        overlap = min(settings.chunk_overlap_tokens, target - 1)
        spans = hard_token_split(doc.text, 0, len(doc.text), target, overlap=overlap)
        return self._from_spans(doc, spans)
