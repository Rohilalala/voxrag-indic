# ruff: noqa: RUF001 -- the "ambiguous" characters flagged here (danda, arabic full stop,
# en dash, directional quotes) are the punctuation this module exists to recognise.
"""Tokenisation and Indic-aware segmentation shared by every chunking strategy.

Two invariants hold everywhere in this package:

1. **A "token" means an encoder token.** Sizes are measured with the tokenizer that
   belongs to ``settings.embed_model_repo``, not with whitespace splitting and not with
   some other model's vocabulary. A chunk sized in the wrong units silently overflows the
   encoder's ``embed_max_tokens`` window and the tail of the chunk is truncated away at
   embed time -- the chunk is then indexed under a vector that does not represent it.

2. **Text is never rewritten, only sliced.** Every strategy works in character offsets
   into ``Document.text`` so ``Chunk.char_start`` / ``char_end`` are true offsets that
   downstream citation highlighting can trust. No normalisation, no re-joining with
   " ".join, no stripping that is not compensated in the offsets.

The document is tokenised exactly once (``token_index``, LRU-cached) and every strategy
slices that one index by binary search. Running all eight strategies over a document
therefore costs one tokenisation, not eight.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import regex

from voxrag.config import settings
from voxrag.contracts import Chunk, ChunkStrategyName, Document, Lang

# --------------------------------------------------------------------------------------
# Module-level constants (not in config.py -- they describe this file's own mechanics)
# --------------------------------------------------------------------------------------

#: Tokens reserved out of `embed_max_tokens` for things the chunk text does not contain:
#: [CLS] + [SEP] plus the two-token "passage: " e5 prefix the embedder prepends.
_ENCODER_RESERVE_TOKENS = 4

#: How many distinct document texts keep their tokenisation cached. Documents are MS MARCO
#: passages (~500 chars), so this is a few MB and removes all re-tokenisation across the
#: eight strategies and across parent/child passes of the same document.
_TOKEN_CACHE_SIZE = 8192

#: Only used when the real tokenizer cannot be loaded (no network, nothing cached).
#: Measured against multilingual-e5-small: Devanagari averages ~3.0 chars/token, Latin
#: ~4.5. Wrong by a little is survivable; using whitespace words would be wrong by 3x.
_FALLBACK_CHARS_PER_TOKEN_LATIN = 4.5
_FALLBACK_CHARS_PER_TOKEN_OTHER = 3.0


# --------------------------------------------------------------------------------------
# Tokeniser
# --------------------------------------------------------------------------------------


def _load_tokenizer() -> Any | None:
    """Load the embedder's tokenizer once, preferring the local HF cache.

    ``hf_hub_download`` is tried before ``Tokenizer.from_pretrained`` because it honours
    the on-disk cache and HF_HUB_OFFLINE, so a container with a warm cache does no network
    IO at import. Returning None is a supported state: `_approx_token_spans` takes over.
    """
    try:
        from tokenizers import Tokenizer
    except Exception:  # pragma: no cover - tokenizers is a hard dependency
        return None
    tok = None
    try:
        from huggingface_hub import hf_hub_download

        tok = Tokenizer.from_file(hf_hub_download(settings.embed_model_repo, "tokenizer.json"))
    except Exception:
        try:
            tok = Tokenizer.from_pretrained(settings.embed_model_repo)
        except Exception:
            return None
    # A tokenizer.json may ship truncation/padding defaults. Both would corrupt offsets:
    # truncation drops the tail of a long document, padding invents (0, 0) spans.
    try:
        tok.no_truncation()
        tok.no_padding()
    except Exception:
        pass
    return tok


_TOKENIZER: Any | None = _load_tokenizer()

_LATIN_RE = regex.compile(r"\p{Latin}")
_NONSPACE_RUN_RE = regex.compile(r"\S+")


def tokenizer_available() -> bool:
    """False when running on the character-ratio fallback; surfaced by evaluate.py."""
    return _TOKENIZER is not None


def _approx_token_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Offset-exact, count-approximate fallback segmentation.

    Character offsets stay contiguous and correct (so chunk boundaries and citations are
    still right); only the token *count* is an estimate from a chars-per-token ratio.
    """
    spans: list[tuple[int, int]] = []
    for m in _NONSPACE_RUN_RE.finditer(text):
        word = m.group()
        ratio = (
            _FALLBACK_CHARS_PER_TOKEN_LATIN
            if _LATIN_RE.search(word)
            else _FALLBACK_CHARS_PER_TOKEN_OTHER
        )
        n = max(1, math.ceil(len(word) / ratio))
        step = len(word) / n
        for i in range(n):
            s = m.start() + int(i * step)
            e = m.end() if i == n - 1 else m.start() + int((i + 1) * step)
            if e > s:
                spans.append((s, e))
    return tuple(spans)


@dataclass(frozen=True, slots=True)
class TokenIndex:
    """Char spans of every encoder token in one text, plus binary-search sidecars."""

    spans: tuple[tuple[int, int], ...]
    _starts: tuple[int, ...]
    _ends: tuple[int, ...]

    @property
    def n(self) -> int:
        return len(self.spans)

    def range(self, start: int, end: int) -> tuple[int, int]:
        """Half-open token index range covering the character range [start, end)."""
        lo = bisect_right(self._ends, start)
        hi = bisect_left(self._starts, end)
        return lo, max(lo, hi)

    def count(self, start: int, end: int) -> int:
        lo, hi = self.range(start, end)
        return hi - lo


@lru_cache(maxsize=_TOKEN_CACHE_SIZE)
def token_index(text: str) -> TokenIndex:
    """Tokenise `text` once and cache it. Every size question in this package goes here."""
    if _TOKENIZER is not None:
        enc = _TOKENIZER.encode(text, add_special_tokens=False)
        # Degenerate zero-width offsets would break bisect monotonicity; drop them.
        raw = tuple((s, e) for s, e in enc.offsets if e > s)
    else:
        raw = _approx_token_spans(text)
    return TokenIndex(raw, tuple(s for s, _ in raw), tuple(e for _, e in raw))


def count_tokens(text: str) -> int:
    """Number of encoder tokens in `text`."""
    return token_index(text).n


def truncate_to_tokens(text: str, n: int) -> str:
    """Longest prefix of `text` that encodes to at most `n` tokens."""
    if n <= 0:
        return ""
    idx = token_index(text)
    if idx.n <= n:
        return text
    return text[: idx.spans[n - 1][1]]


def embed_budget_tokens() -> int:
    """Largest chunk, in tokens, that survives the encoder without truncation."""
    return max(1, settings.embed_max_tokens - _ENCODER_RESERVE_TOKENS)


# --------------------------------------------------------------------------------------
# Spans
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Span:
    """A half-open character range of a document, carrying its own text."""

    start: int
    end: int
    text: str

    @property
    def n_tokens(self) -> int:
        return count_tokens(self.text)


def make_span(text: str, start: int, end: int) -> Span | None:
    """Span for [start, end) with surrounding whitespace trimmed *and offsets adjusted*.

    Returns None when the range holds nothing but whitespace. Trimming without moving the
    offsets is the classic way citation highlighting ends up pointing one line too early.
    """
    start = max(0, start)
    end = min(len(text), end)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end <= start:
        return None
    return Span(start, end, text[start:end])


def merge_spans(text: str, spans: Sequence[Span]) -> Span | None:
    """One span covering `spans`, sliced from the original text so separators survive."""
    if not spans:
        return None
    return make_span(text, spans[0].start, spans[-1].end)


def pack_spans(spans: Sequence[Span], max_tokens: int) -> list[list[Span]]:
    """Greedily group consecutive spans into runs of at most `max_tokens` tokens.

    Summing per-span token counts slightly over-estimates the joined text (SentencePiece
    merges across the join), so the packed group is never *over* budget -- the real count
    is recomputed on the finished chunk.
    """
    groups: list[list[Span]] = []
    current: list[Span] = []
    used = 0
    for sp in spans:
        n = sp.n_tokens
        if current and used + n > max_tokens:
            groups.append(current)
            current = []
            used = 0
        current.append(sp)
        used += n
    if current:
        groups.append(current)
    return groups


# --------------------------------------------------------------------------------------
# Indic-aware sentence segmentation
# --------------------------------------------------------------------------------------

#: Unambiguous sentence terminators outside the Latin set.
#: U+0964 danda and U+0965 double danda end sentences in Devanagari (hi/mr/ne/sa) and are
#: also used in Bengali, Assamese, Gujarati, Oriya and Gurmukhi text. U+06D4 arabic full
#: stop and U+061F arabic question mark do the same job in Urdu. A splitter that only
#: knows ".!?" returns ONE sentence for a Hindi passage, which silently collapses
#: SENTENCE_WINDOW, SEMANTIC, RECURSIVE and PARENT_CHILD into "one chunk per document".
_INDIC_TERMINATORS = "।॥۔؟…‼⁇⁈⁉"
_LATIN_TERMINATORS = ".!?"

_TERMINATOR_RE = regex.compile(
    rf"[{_INDIC_TERMINATORS}]+|[{regex.escape(_LATIN_TERMINATORS)}]+|\n+"
)

#: Closing punctuation that belongs to the sentence that just ended, not the next one.
_CLOSERS = ")]}>\"'”’»›」』༽"

#: Lowercased tokens that end in a period without ending a sentence. Latin-script only;
#: Indic terminators have no such ambiguity, which is why they bypass this check.
_ABBREVIATIONS = frozenset(
    """mr mrs ms dr prof sr jr st vs etc no nos vol vols fig figs ed eds al approx
    inc ltd co corp dept univ govt pvt est ca cf viz esp p pp ch sec art rs
    jan feb mar apr jun jul aug sep sept oct nov dec mon tue wed thu fri sat sun
    kg km cm mm ml lb ft oz hr min sec ave blvd rd""".split()  # noqa: SIM905
)

_TRAILING_WORD_RE = regex.compile(r"[\p{L}\p{N}.]+$")

#: Clause-level separators, one level below a sentence on the recursive ladder. Urdu uses
#: U+060C arabic comma and U+061B arabic semicolon; a plain "," never appears there.
_CLAUSE_PATTERN_ARABIC = regex.compile(r"[،؛,;:—–]\s*")
_CLAUSE_PATTERN_DEFAULT = regex.compile(r"[,;:—–॥]\s*")

#: Structural separators, coarsest first. RECURSIVE walks these in order.
PARAGRAPH_RE = regex.compile(r"\n\s*\n+")
LINE_RE = regex.compile(r"\n+")
SPACE_RE = regex.compile(r"\s+")


def clause_pattern(lang: Lang | None) -> regex.Pattern[str]:
    """Clause separator regex for a language's script."""
    return _CLAUSE_PATTERN_ARABIC if lang is Lang.URD else _CLAUSE_PATTERN_DEFAULT


def _is_latin_abbreviation(text: str, dot_index: int) -> bool:
    """True when the period at `dot_index` closes an abbreviation, not a sentence."""
    m = _TRAILING_WORD_RE.search(text, max(0, dot_index - 16), dot_index)
    if m is None:
        return False
    word = m.group()
    if "." in word:  # "U.S.", "e.g." -- an initialism, not a sentence end
        return True
    if len(word) == 1 and word.isalpha() and word.isupper():  # "J. K. Rowling"
        return True
    return word.lower() in _ABBREVIATIONS


def split_sentences(
    text: str,
    lang: Lang | None = None,
    *,
    max_tokens: int | None = None,
    start: int = 0,
    end: int | None = None,
) -> list[Span]:
    """Split `text` into sentence spans, honouring Indic and Arabic-script terminators.

    Offsets in the returned spans are always into `text`, even when `start`/`end` restrict
    the scan to a sub-range -- that is how RECURSIVE splits a paragraph into sentences
    without losing its place in the document.

    `max_tokens` additionally guarantees no returned span exceeds that size: an over-long
    "sentence" (unpunctuated passages are common in this corpus) is broken down the same
    clause -> whitespace -> token ladder RECURSIVE uses, so sentence-based strategies
    cannot emit a chunk the encoder would truncate.
    """
    spans: list[Span] = []
    n = len(text) if end is None else min(end, len(text))
    pos = max(0, start)
    for m in _TERMINATOR_RE.finditer(text, pos, n):
        term = m.group()
        if term[0] == "\n":
            # A hard line break ends a sentence even without punctuation.
            boundary, resume = m.start(), m.end()
        else:
            stop = m.end()
            while stop < n and text[stop] in _CLOSERS:
                stop += 1
            if term[0] in _LATIN_TERMINATORS:
                nxt = text[stop] if stop < n else ""
                # Requiring whitespace/EOS after the period also rejects decimals ("3.5")
                # and URLs, for free.
                if nxt and not nxt.isspace():
                    continue
                if _is_latin_abbreviation(text, m.start()):
                    continue
            boundary = resume = stop
        sp = make_span(text, pos, boundary)
        if sp is not None:
            spans.append(sp)
        pos = max(pos, resume)
    tail = make_span(text, pos, n)
    if tail is not None:
        spans.append(tail)

    if max_tokens is None:
        return spans
    bounded: list[Span] = []
    for sp in spans:
        if sp.n_tokens <= max_tokens:
            bounded.append(sp)
        else:
            bounded.extend(split_oversized(text, sp, max_tokens, lang))
    return bounded


def split_on(pattern: regex.Pattern[str], text: str, start: int, end: int) -> list[Span]:
    """Split [start, end) on `pattern`, keeping each separator with the piece it closes."""
    out: list[Span] = []
    pos = start
    for m in pattern.finditer(text, start, end):
        if m.end() <= pos:
            continue
        sp = make_span(text, pos, m.end())
        if sp is not None:
            out.append(sp)
        pos = m.end()
    tail = make_span(text, pos, end)
    if tail is not None:
        out.append(tail)
    return out


def hard_token_split(
    text: str, start: int, end: int, max_tokens: int, *, overlap: int = 0
) -> list[Span]:
    """Cut [start, end) into token windows. The floor of the ladder: always terminates."""
    idx = token_index(text)
    lo, hi = idx.range(start, end)
    if hi <= lo:
        return []
    stride = max(1, max_tokens - max(0, min(overlap, max_tokens - 1)))
    out: list[Span] = []
    i = lo
    while i < hi:
        j = min(hi, i + max_tokens)
        sp = make_span(text, max(start, idx.spans[i][0]), min(end, idx.spans[j - 1][1]))
        # With overlap the final window can be fully inside its predecessor; that chunk
        # would be pure duplicate index weight, so drop it.
        if sp is not None and not (out and sp.end <= out[-1].end):
            out.append(sp)
        if j >= hi:
            break
        i += stride
    return out


def split_oversized(text: str, span: Span, max_tokens: int, lang: Lang | None) -> list[Span]:
    """Break one over-long span down the clause -> whitespace -> token ladder."""
    for pattern in (clause_pattern(lang), SPACE_RE):
        pieces = split_on(pattern, text, span.start, span.end)
        if len(pieces) <= 1:
            continue
        out: list[Span] = []
        for group in pack_spans(pieces, max_tokens):
            merged = merge_spans(text, group)
            if merged is None:
                continue
            if merged.n_tokens > max_tokens:
                out.extend(hard_token_split(text, merged.start, merged.end, max_tokens))
            else:
                out.append(merged)
        if out:
            return out
    return hard_token_split(text, span.start, span.end, max_tokens)


# --------------------------------------------------------------------------------------
# Chunker base
# --------------------------------------------------------------------------------------


class BaseChunker(ABC):
    """Shared Chunk construction. Subclasses only decide *where* the boundaries go."""

    name: ChunkStrategyName

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]: ...

    def _build(
        self,
        doc: Document,
        ordinal: int,
        start: int,
        end: int,
        *,
        text: str | None = None,
        context_text: str | None = None,
        parent_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Chunk:
        """Assemble one Chunk with true offsets, true token count and the canonical id.

        `text` overrides the embedded string (METADATA_AWARE prepends a header that is not
        present in doc.text); the offsets always describe the document, never the header.
        """
        body = doc.text[start:end] if text is None else text
        n_tokens = count_tokens(body)
        chunk_meta = dict(meta) if meta else {}
        if n_tokens > embed_budget_tokens():
            # The embedder will truncate this chunk; evaluate.py reports the rate so the
            # tradeoff is visible instead of silent.
            chunk_meta["over_encoder_window"] = True
        return Chunk(
            chunk_id=f"{doc.doc_id}:{self.name.value}:{ordinal}",
            doc_id=doc.doc_id,
            text=body,
            lang=doc.lang,
            strategy=self.name,
            ordinal=ordinal,
            char_start=start,
            char_end=end,
            n_tokens=n_tokens,
            title=doc.title,
            url=doc.url,
            context_text=context_text,
            parent_id=parent_id,
            meta=chunk_meta,
        )

    def _from_spans(self, doc: Document, spans: Sequence[Span]) -> list[Chunk]:
        return [self._build(doc, i, sp.start, sp.end) for i, sp in enumerate(spans)]
