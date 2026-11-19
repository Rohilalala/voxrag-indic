"""Metadata-aware chunking: make the metadata searchable, not just filterable.

Title, language and the passage's own MS MARCO query are normally payload -- available for
a `where lang = 'hi'` filter and invisible to the vector. Prepending a compact header to
the *embedded* string puts them into the same semantic space as the query, so "Tamil
recipes for pongal" can match on the language and topic signal even when the body never
names either. It also anchors chunks cut from the middle of a document, which otherwise
embed with no indication of what they are about.

The header is deliberately kept out of `context_text`. If the generator saw it, an
extractive answer could quote "Hindi | how tall is mount everest" back at the user as
though it were corpus text. `text` is what gets embedded; `context_text` is what gets
read; they are not the same string here, and that is the entire trick.
"""

from __future__ import annotations

from voxrag.chunking.base import (
    BaseChunker,
    count_tokens,
    merge_spans,
    pack_spans,
    split_sentences,
    truncate_to_tokens,
)
from voxrag.config import settings
from voxrag.contracts import Chunk, ChunkStrategyName, Document, Lang

#: English language names. English rather than endonyms because multilingual-e5 aligns
#: language *names* across scripts far better than it aligns bare ISO codes, and the query
#: side ("in Tamil", "Bengali") arrives in English or transliterated far more often than
#: in the native script.
LANGUAGE_NAMES: dict[Lang, str] = {
    Lang.ASM: "Assamese",
    Lang.BEN: "Bengali",
    Lang.GUJ: "Gujarati",
    Lang.HIN: "Hindi",
    Lang.KAN: "Kannada",
    Lang.MAL: "Malayalam",
    Lang.MAR: "Marathi",
    Lang.NEP: "Nepali",
    Lang.ORI: "Odia",
    Lang.PAN: "Punjabi",
    Lang.SAN: "Sanskrit",
    Lang.TAM: "Tamil",
    Lang.TEL: "Telugu",
    Lang.URD: "Urdu",
    Lang.ENG: "English",
}

#: Per-field header caps. A 60-token MS MARCO query would otherwise consume a third of the
#: chunk budget and dilute the body it is supposed to describe.
_MAX_TITLE_TOKENS = 24
_MAX_QUERY_TOKENS = 24
#: Floor on the body once the header is deducted, so a pathological header cannot reduce
#: chunks to a handful of tokens.
_MIN_BODY_TOKENS = 32

#: Tokens reserved for the "\n" joining header to body. One in this tokenizer; kept as a
#: named constant so the reason survives if the separator ever changes.
_SEPARATOR_TOKENS = 1

_FIELD_SEP = " | "


class MetadataAwareChunker(BaseChunker):
    name = ChunkStrategyName.METADATA_AWARE

    def chunk(self, doc: Document) -> list[Chunk]:
        header = self._header(doc)
        header_tokens = count_tokens(header)
        # Reserve the joining newline too. Budgeting only header+body overflows the target by
        # exactly one token, because `f"{header}\n{body}"` is re-tokenised as a whole and the
        # separator becomes its own piece -- header(1) + body(159) measured apart came back as
        # 161 together against a 160 target. Sub-token overflow still truncates inside the
        # encoder's window, so it is worth being exact rather than nearly right.
        body_budget = max(
            _MIN_BODY_TOKENS,
            settings.chunk_target_tokens - header_tokens - _SEPARATOR_TOKENS,
        )

        sentences = split_sentences(doc.text, doc.lang, max_tokens=body_budget)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        for ordinal, group in enumerate(pack_spans(sentences, body_budget)):
            body = merge_spans(doc.text, group)
            if body is None:
                continue
            chunks.append(
                self._build(
                    doc,
                    ordinal,
                    body.start,
                    body.end,
                    text=f"{header}\n{body.text}",
                    # Always set, never left to fall back to `text`: the fallback would
                    # hand the header to the generator.
                    context_text=body.text,
                    meta={"header": header, "header_tokens": header_tokens},
                )
            )
        return chunks

    @staticmethod
    def _header(doc: Document) -> str:
        """`Title | Language | source query`, with empty and duplicate fields dropped.

        The source query is gated behind `settings.metadata_include_source_query`, and it
        defaults OFF, because including it INVALIDATES this corpus's own benchmark.

        MSMARCO-XI's `source_query` is the query that retrieved the passage, and the
        evaluation's graded queries are exactly those strings. Embedding one into its own
        passage therefore puts the answer key inside the document: measured on the eight-way
        comparison this strategy scored Recall@10 = 0.996 and Recall@20 = 1.000, against
        0.76 for the next best -- a result that says nothing about chunking.

        The technique itself is sound; indexing likely queries alongside a passage is
        doc2query-style expansion and it genuinely helps on queries the index has not seen.
        It just cannot be scored against the queries it memorised. Turn it on for a
        production index, leave it off to measure anything.
        """
        parts: list[str] = []
        title = (doc.title or "").strip()
        if title:
            parts.append(truncate_to_tokens(title, _MAX_TITLE_TOKENS))
        parts.append(LANGUAGE_NAMES.get(doc.lang, doc.lang.value))
        query_type = str(doc.meta.get("query_type") or "").strip()
        if query_type:
            parts.append(query_type)
        if settings.metadata_include_source_query:
            query = (doc.source_query or "").strip()
            # MSMARCO-XI often carries the query as the pseudo-title too; repeating it would
            # double its weight in the embedding for no added signal.
            if query and query.casefold() != title.casefold():
                parts.append(truncate_to_tokens(query, _MAX_QUERY_TOKENS))
        return _FIELD_SEP.join(parts)
