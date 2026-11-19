"""Sentence-window chunking: embed one sentence, generate from its neighbourhood.

The asymmetry is the whole point. A single sentence is a tight, low-noise embedding
target, so dense retrieval matches it precisely; but a single sentence usually lacks the
antecedents ("he", "that year", "the same treaty") an answer needs. So `text` -- the
indexed string -- stays one sentence, while `context_text` -- what the generator reads --
carries `sentence_window_size` sentences on each side. Precision when matching, recall
when generating, with no extra vectors in the index.
"""

from __future__ import annotations

from voxrag.chunking.base import BaseChunker, embed_budget_tokens, split_sentences
from voxrag.config import settings
from voxrag.contracts import Chunk, ChunkStrategyName, Document


class SentenceWindowChunker(BaseChunker):
    name = ChunkStrategyName.SENTENCE_WINDOW

    def chunk(self, doc: Document) -> list[Chunk]:
        # Bounded at the encoder window: an unpunctuated 400-token "sentence" would
        # otherwise be indexed under an embedding of only its first 188 tokens.
        sentences = split_sentences(doc.text, doc.lang, max_tokens=embed_budget_tokens())
        if not sentences:
            return []

        window = max(0, settings.sentence_window_size)
        chunks: list[Chunk] = []
        for i, sent in enumerate(sentences):
            lo = max(0, i - window)
            hi = min(len(sentences) - 1, i + window)
            # Sliced from the original text rather than joined, so the real inter-sentence
            # punctuation and spacing reach the generator intact.
            context = doc.text[sentences[lo].start : sentences[hi].end]
            chunks.append(
                self._build(
                    doc,
                    i,
                    sent.start,
                    sent.end,
                    context_text=context if hi > lo else None,
                    meta={
                        "window_start_sentence": lo,
                        "window_end_sentence": hi,
                        "context_char_start": sentences[lo].start,
                        "context_char_end": sentences[hi].end,
                    },
                )
            )
        return chunks
