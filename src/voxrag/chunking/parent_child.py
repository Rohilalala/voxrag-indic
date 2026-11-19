"""Parent-child (small-to-big) chunking: index small, generate big.

Two sizes of the same text serve two different consumers. Retrieval wants the smallest
unit that can carry a single fact, because a `child_chunk_tokens` embedding is dominated
by the one claim it contains instead of being averaged across a paragraph of unrelated
sentences. Generation wants the opposite: enough surrounding text to resolve pronouns,
units and dates.

So only children are emitted as indexable Chunks (`text` = child), each carrying its
parent's full text in `context_text` and its parent's id in `parent_id`. The index holds
`parent_chunk_tokens / child_chunk_tokens` times more vectors than PASSAGE_NATIVE but each
one is far sharper, and the generator still sees whole parents. This is the configured
`active_strategy` for the served index.
"""

from __future__ import annotations

from voxrag.chunking.base import (
    BaseChunker,
    Span,
    merge_spans,
    pack_spans,
    split_sentences,
)
from voxrag.config import settings
from voxrag.contracts import Chunk, ChunkStrategyName, Document


class ParentChildChunker(BaseChunker):
    name = ChunkStrategyName.PARENT_CHILD

    def chunk(self, doc: Document) -> list[Chunk]:
        child_budget = max(1, settings.child_chunk_tokens)
        parent_budget = max(child_budget, settings.parent_chunk_tokens)

        # Splitting to the *child* budget up front means every sentence already fits a
        # child, so children never need a second, boundary-blind pass.
        sentences = split_sentences(doc.text, doc.lang, max_tokens=child_budget)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        ordinal = 0
        for parent_ordinal, parent_group in enumerate(pack_spans(sentences, parent_budget)):
            parent = merge_spans(doc.text, parent_group)
            if parent is None:
                continue
            # Stable, derivable id: the parent is not itself indexed, so nothing can look
            # it up by ordinal -- the id has to encode what it is.
            parent_id = f"{doc.doc_id}:{self.name.value}:parent:{parent_ordinal}"
            children: list[Span] = []
            for child_group in pack_spans(parent_group, child_budget):
                child = merge_spans(doc.text, child_group)
                if child is not None:
                    children.append(child)
            for child_index, child in enumerate(children):
                chunks.append(
                    self._build(
                        doc,
                        ordinal,
                        child.start,
                        child.end,
                        context_text=parent.text,
                        parent_id=parent_id,
                        meta={
                            "parent_ordinal": parent_ordinal,
                            "parent_char_start": parent.start,
                            "parent_char_end": parent.end,
                            "child_index": child_index,
                            "n_children": len(children),
                        },
                    )
                )
                ordinal += 1
        return chunks
