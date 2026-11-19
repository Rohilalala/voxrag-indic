"""One chunk per document -- the corpus's own segmentation, used as-is.

MS MARCO passages were cut by humans from web documents to be exactly the unit a search
engine returns: one topic, one answer, a few sentences. Re-cutting them is not obviously
an improvement, and in `evaluate.py` this strategy usually wins on the fragmentation and
redundancy metrics outright while indexing the fewest vectors. Any strategy that cannot
beat it is not earning its extra index cost.

The one hazard is length. A passage longer than the encoder window is embedded from its
prefix only, so the chunk is indexed under a vector that does not represent its tail.
Splitting would make this a different strategy, so instead the condition is measured and
reported (`meta["over_encoder_window"]`, the `>enc%` column in the comparison table).
"""

from __future__ import annotations

from voxrag.chunking.base import BaseChunker, make_span
from voxrag.contracts import Chunk, ChunkStrategyName, Document


class PassageNativeChunker(BaseChunker):
    name = ChunkStrategyName.PASSAGE_NATIVE

    def chunk(self, doc: Document) -> list[Chunk]:
        span = make_span(doc.text, 0, len(doc.text))
        if span is None:
            return []
        return [self._build(doc, 0, span.start, span.end)]
