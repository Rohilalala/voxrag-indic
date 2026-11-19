"""Recursive separator-ladder chunking, token-budgeted and Indic-aware.

Same idea as LangChain's RecursiveCharacterTextSplitter, with two corrections that matter
for this corpus:

* the budget is counted in **encoder tokens**, not characters -- 400 characters is ~90
  tokens of English and ~200 tokens of Malayalam, so a character budget produces chunks of
  wildly different embedding quality across the 14 languages;
* the sentence rung understands danda and the Arabic full stop, so Hindi and Urdu descend
  to real sentences instead of falling straight through to whitespace.

The ladder is paragraph -> line -> sentence -> clause -> whitespace -> raw token window.
At each rung the span is split, adjacent pieces are greedily re-packed up to the target,
and only pieces still over budget descend further. Splitting on the *coarsest* separator
that fits is what preserves structure: a paragraph that already fits is never touched.
"""

from __future__ import annotations

from voxrag.chunking.base import (
    LINE_RE,
    PARAGRAPH_RE,
    SPACE_RE,
    BaseChunker,
    Span,
    clause_pattern,
    hard_token_split,
    make_span,
    merge_spans,
    pack_spans,
    split_on,
    split_sentences,
)
from voxrag.config import settings
from voxrag.contracts import Chunk, ChunkStrategyName, Document, Lang

#: Index of the last rung; beyond it only raw token windows remain, which always terminate.
_LAST_LEVEL = 5


class RecursiveChunker(BaseChunker):
    name = ChunkStrategyName.RECURSIVE

    def chunk(self, doc: Document) -> list[Chunk]:
        root = make_span(doc.text, 0, len(doc.text))
        if root is None:
            return []
        target = settings.chunk_target_tokens
        spans = self._descend(doc.text, root, doc.lang, target, level=0)
        # Re-pack across the descent output: a document that split at the paragraph rung
        # can leave several short paragraphs that would each become their own chunk, which
        # fragments the index for no retrieval benefit.
        packed: list[Span] = []
        for group in pack_spans(spans, target):
            merged = merge_spans(doc.text, group)
            if merged is not None:
                packed.append(merged)
        return self._from_spans(doc, packed)

    def _split_at_level(self, text: str, span: Span, lang: Lang | None, level: int) -> list[Span]:
        if level == 0:
            return split_on(PARAGRAPH_RE, text, span.start, span.end)
        if level == 1:
            return split_on(LINE_RE, text, span.start, span.end)
        if level == 2:
            return split_sentences(text, lang, start=span.start, end=span.end)
        if level == 3:
            return split_on(clause_pattern(lang), text, span.start, span.end)
        if level == 4:
            return split_on(SPACE_RE, text, span.start, span.end)
        return hard_token_split(text, span.start, span.end, settings.chunk_target_tokens)

    def _descend(
        self, text: str, span: Span, lang: Lang | None, target: int, *, level: int
    ) -> list[Span]:
        if span.n_tokens <= target:
            return [span]
        if level > _LAST_LEVEL:
            return hard_token_split(text, span.start, span.end, target)
        pieces = self._split_at_level(text, span, lang, level)
        if len(pieces) <= 1:
            # This separator does not occur inside the span; drop to the next rung on the
            # same span rather than emitting an over-budget chunk.
            return self._descend(text, span, lang, target, level=level + 1)
        out: list[Span] = []
        for group in pack_spans(pieces, target):
            merged = merge_spans(text, group)
            if merged is None:
                continue
            if merged.n_tokens > target:
                out.extend(self._descend(text, merged, lang, target, level=level + 1))
            else:
                out.append(merged)
        return out
