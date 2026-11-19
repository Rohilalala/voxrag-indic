"""Script-aware BM25 over the Indic corpus -- the arm that catches what dense retrieval drops.

MS MARCO queries are stuffed with numerals, product codes, proper nouns and transliterated
rare words. A 384-dim multilingual encoder smears exactly those into their neighbourhoods,
so a lexical arm is not a nicety here: it is how "121 करोड़" beats "a large population".

Two things make or break Indic BM25 and both are handled below:

* Tokenisation. A `str.split()` / `\\w+` tokeniser is ASCII-shaped; on Devanagari, Tamil or
  Telugu it either returns the whole line as one token or shatters syllables. We match
  `[\\p{L}\\p{N}]\\p{M}*` runs with the `regex` module so every combining mark (matra,
  virama, nukta, Arabic harakat) stays welded to the base character it modifies -- a matra
  split off from its consonant is not a shorter word, it is a different, meaningless one.
* Normalisation. ZWNJ/ZWJ (U+200C/U+200D) are invisible and inconsistently typed, so the
  same word arrives spelled two ways and hashes to two vocabulary entries; they are deleted.
  Danda U+0964/U+0965 is sentence punctuation and is replaced by a *space*, not deleted,
  because deleting it fuses the last word of one sentence to the first of the next.

One BM25 matrix per language. IDF is a corpus-level statistic: pooling Tamil and Hindi in
one index makes every Tamil term look rare to a Hindi query and vice versa, and the score
scales stop being comparable. Per-language indexes also let a query skip every index whose
script it cannot possibly match, which is most of them.
"""

from __future__ import annotations

import atexit
import json
import logging
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import bm25s
import numpy as np
import regex
from bm25s.stopwords import STOPWORDS_EN
from bm25s.tokenization import Tokenized
from pydantic import ValidationError

from voxrag.config import settings
from voxrag.contracts import LANG_SCRIPT, Chunk, Lang, ScoredChunk

_LOG = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only when the optional wheel is missing
    import Stemmer as _pystemmer  # PyStemmer
except ImportError:  # a missing stemmer degrades English recall, it does not break serving
    _pystemmer = None  # type: ignore[assignment]
    _LOG.warning("PyStemmer unavailable; the English pivot will be indexed unstemmed")

#: Bumped whenever tokenisation changes. A persisted vocabulary built by an older tokeniser
#: still *loads* -- it just silently stops matching the queries the new one produces, which
#: is the worst possible failure mode -- so the manifest carries this and refuses instead.
TOKENIZER_VERSION: Final = 1

_MANIFEST_FILE: Final = "manifest.json"
_CHUNKS_FILE: Final = "chunks.jsonl"

#: Memory-map the score matrix: startup does not pay to read it, and `warmup()` faults it in
#: sequentially afterwards so the request path never eats a page fault mid-query.
_MMAP: Final = True

#: Runs of letters/digits with their combining marks attached. Non-capturing so `findall`
#: yields whole tokens. Digits are inside the class on purpose: "2011", "500mg" and "sr-22"
#: are precisely the queries the dense arm cannot serve.
_TOKEN_RE: Final = regex.compile(r"(?:[\p{L}\p{N}]\p{M}*)+")

#: Applied before NFC. ZWJ/ZWNJ vanish (invisible, optional, vocabulary-splitting); danda and
#: double danda become spaces (sentence punctuation, and dropping them would fuse two words).
_TRANSLATE: Final = str.maketrans(
    {
        "‌": None,  # ZWNJ
        "‍": None,  # ZWJ
        "﻿": None,  # BOM, arrives inside parquet strings often enough to matter
        "।": " ",  # danda
        "॥": " ",  # double danda
    }
)

_EN_STOPWORDS: Final = frozenset(STOPWORDS_EN)

#: One matcher per Unicode script named in contracts.LANG_SCRIPT, for query script detection.
_SCRIPT_RE: Final[dict[str, regex.Pattern[str]]] = {
    script: regex.compile(rf"\p{{{script}}}") for script in set(LANG_SCRIPT.values())
}

#: PyStemmer's Stemmer objects wrap a non-reentrant C struct, and `search` is called from
#: whatever thread `asyncio.to_thread` hands it. One stemmer per thread, never shared.
_TLS = threading.local()


def _english_stemmer() -> Any:
    stemmer = getattr(_TLS, "stemmer", None)
    if stemmer is None:
        stemmer = _pystemmer.Stemmer("english")
        _TLS.stemmer = stemmer
    return stemmer


def normalize(text: str) -> str:
    """Fold away the invisible spelling variation that would split one word into two terms."""
    # translate before NFC: with the joiners gone, NFC can compose sequences that the joiner
    # was blocking, so the two passes have to run in this order to converge.
    return unicodedata.normalize("NFC", text.translate(_TRANSLATE)).lower()


def tokenize(text: str, lang: Lang | None = None) -> list[str]:
    """Script-aware tokenisation. English is stopworded and stemmed; nothing else is.

    There is no stemmer worth the name for these 14 languages (Snowball covers none of them,
    and a naive suffix stripper on an agglutinative language like Tamil destroys more terms
    than it merges), so we do not pretend. BM25's own IDF already discounts the frequent
    inflections that stemming would have collapsed.
    """
    if not text:
        return []
    tokens = _TOKEN_RE.findall(normalize(text))
    if lang is Lang.ENG and tokens:
        tokens = [t for t in tokens if t not in _EN_STOPWORDS]
        if _pystemmer is not None and tokens:
            # stemWords is a single C call over the whole list; per-token calls are ~4x slower.
            tokens = _english_stemmer().stemWords(tokens)
    return tokens


@dataclass
class _LangIndex:
    """One language's BM25 matrix plus the chunks its document ids point at.

    `chunks` is positional: bm25s returns a document index, and `chunks[i]` hydrates it with
    no dictionary lookup and no second store round trip.
    """

    lang: Lang
    retriever: bm25s.BM25
    chunks: list[Chunk]


class BM25Index:
    """LexicalIndex over per-language BM25 matrices. Implements `contracts.LexicalIndex`.

    `index()` is cheap and additive: it only accumulates chunks. The matrix is (re)built
    lazily on the first search that needs it, or explicitly via `build()`/`save()`. bm25s
    compiles a static CSC score matrix -- there is no append -- so building eagerly inside
    `index()` would rebuild the whole language once per ingest batch, i.e. O(n^2) over an
    ingest that streams. Deferring makes it one build, whoever calls it.
    """

    name = "bm25"

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else Path(settings.bm25_path)
        self.path.mkdir(parents=True, exist_ok=True)

        self._indexes: dict[Lang, _LangIndex] = {}
        #: Authoritative corpus per language; the matrix is derived from it.
        self._chunks: dict[Lang, list[Chunk]] = {}
        #: chunk_id -> position, so a re-indexed chunk replaces rather than duplicates.
        self._pos: dict[Lang, dict[str, int]] = {}
        self._dirty: set[Lang] = set()
        self._unsaved = False
        self._lock = threading.Lock()

        self._load()
        _LOG.info(
            "bm25 index ready: path=%s langs=%s chunks=%d tokenizer=v%d",
            self.path,
            [lang.value for lang in self._indexes],
            sum(len(c) for c in self._chunks.values()),
            TOKENIZER_VERSION,
        )
        atexit.register(self._flush_at_exit)

    # -- LexicalIndex --------------------------------------------------------------------

    def index(self, chunks: list[Chunk]) -> None:
        """Add or replace chunks. Cheap: no tokenisation, no matrix work happens here."""
        if not chunks:
            return
        with self._lock:
            for chunk in chunks:
                corpus = self._chunks.setdefault(chunk.lang, [])
                positions = self._pos.setdefault(chunk.lang, {})
                pos = positions.get(chunk.chunk_id)
                if pos is None:
                    positions[chunk.chunk_id] = len(corpus)
                    corpus.append(chunk)
                else:
                    corpus[pos] = chunk
                self._dirty.add(chunk.lang)
            self._unsaved = True

    def search(
        self, query: str, *, top_k: int = 20, lang: Lang | None = None
    ) -> list[ScoredChunk]:
        if top_k <= 0 or not query or not query.strip():
            return []
        if self._dirty:  # unsynchronised read of a set is fine; build() re-checks under lock
            self.build()

        langs = [lang] if lang is not None else self._candidate_langs(query)
        scored: list[tuple[float, Chunk]] = []
        for target in langs:
            entry = self._indexes.get(target)
            if entry is None:
                continue
            tokens = tokenize(query, target)
            if not tokens:
                continue
            scores = entry.retriever.get_scores(tokens)
            scored.extend(self._top(scores, entry.chunks, top_k))

        # Sorted across languages then truncated: with a single candidate language this is a
        # no-op re-sort of <=top_k items, and with several it is the only way to get one
        # coherent ranking out of independent matrices.
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            ScoredChunk(chunk=chunk, score=score, lexical_score=score, lexical_rank=rank)
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]

    # -- build / persistence -------------------------------------------------------------

    def build(self) -> None:
        """Compile the BM25 matrix for every language touched since the last build."""
        with self._lock:
            dirty = sorted(self._dirty, key=lambda lang: lang.value)
            for lang in dirty:
                corpus = self._chunks.get(lang) or []
                if not corpus:
                    self._indexes.pop(lang, None)
                    self._dirty.discard(lang)
                    continue
                retriever = bm25s.BM25(method="lucene", backend="numpy")
                retriever.index(self._tokenized(lang, corpus), show_progress=False)
                # Swap wholesale: a concurrent search holds its own reference to the previous
                # _LangIndex and stays internally consistent until it finishes.
                self._indexes[lang] = _LangIndex(lang=lang, retriever=retriever, chunks=corpus)
                self._dirty.discard(lang)
            if dirty:
                _LOG.info(
                    "bm25 built %s (%s docs)",
                    [lang.value for lang in dirty],
                    [len(self._chunks.get(lang) or []) for lang in dirty],
                )

    def save(self) -> None:
        """Build anything outstanding, then persist every language under `settings.bm25_path`."""
        self.build()
        with self._lock:
            langs: dict[str, int] = {}
            for lang, entry in self._indexes.items():
                target = self.path / lang.value
                target.mkdir(parents=True, exist_ok=True)
                entry.retriever.save(target, show_progress=False)
                # Own sidecar rather than bm25s' corpus file: it round-trips a full pydantic
                # Chunk (offsets, parent_id, meta) instead of the {id,text} shape bm25s writes.
                with (target / _CHUNKS_FILE).open("w", encoding="utf-8") as fh:
                    fh.writelines(f"{chunk.model_dump_json()}\n" for chunk in entry.chunks)
                langs[lang.value] = len(entry.chunks)
            (self.path / _MANIFEST_FILE).write_text(
                json.dumps({"tokenizer": TOKENIZER_VERSION, "langs": langs}), encoding="utf-8"
            )
            self._unsaved = False
        _LOG.info("bm25 saved %d language index(es) to %s", len(self._indexes), self.path)

    # -- extras used by the API and the benchmark ----------------------------------------

    def warmup(self) -> None:
        """Fault the memory-mapped score matrices in, sequentially, before serving.

        Without this the first query per term pays a page fault per CSC column it touches,
        which lands in the P100 the benchmark reports. One striding read is far cheaper than
        discovering those pages one random column at a time.
        """
        for entry in self._indexes.values():
            data = entry.retriever.scores.get("data")
            if data is not None and len(data):
                float(np.asarray(data).sum())  # one sequential pass; touches every page

    def count(self) -> int:
        return sum(len(chunks) for chunks in self._chunks.values())

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "tokenizer": TOKENIZER_VERSION,
            "mmap": _MMAP,
            "chunks": self.count(),
            "langs": {
                lang.value: {
                    "docs": len(entry.chunks),
                    "vocab": len(entry.retriever.vocab_dict or {}),
                }
                for lang, entry in self._indexes.items()
            },
            "pending_langs": sorted(lang.value for lang in self._dirty),
        }

    # -- internals -----------------------------------------------------------------------

    def _tokenized(self, lang: Lang, corpus: list[Chunk]) -> Tokenized:
        """Tokenise straight into dense vocabulary ids.

        Handing bm25s a `Tokenized` instead of list-of-list-of-str skips its own vocabulary
        pass and, more importantly, never materialises ~10M token *strings* at once: the ids
        are the only intermediate. Ids must be dense from 0 -- bm25s sizes the score matrix
        as `len(vocab)` columns and indexes it by token id.
        """
        vocab: dict[str, int] = {}
        ids: list[list[int]] = []
        for chunk in corpus:
            # Title is indexed alongside the body: on MS MARCO it is short, high-IDF and often
            # carries the entity the query names. Strategies that already inline it just get a
            # slightly higher term frequency for those terms, which is harmless.
            text = f"{chunk.title}\n{chunk.text}" if chunk.title else chunk.text
            doc: list[int] = []
            for token in tokenize(text, lang):
                token_id = vocab.get(token)
                if token_id is None:
                    token_id = vocab[token] = len(vocab)
                doc.append(token_id)
            ids.append(doc)
        return Tokenized(ids=ids, vocab=vocab)

    @staticmethod
    def _top(scores: np.ndarray, chunks: list[Chunk], top_k: int) -> list[tuple[float, Chunk]]:
        """Top-k by partial selection.

        argpartition is O(n) against argsort's O(n log n); at 100k documents that is the
        difference between ~0.4ms and ~6ms, and it happens on every query. Zero and negative
        scores are dropped: a document no query term touched scores exactly 0, and letting it
        through would hand fusion a rank -- and therefore RRF credit -- it did not earn.
        """
        n = min(len(scores), len(chunks))
        if n == 0:
            return []
        k = min(top_k, n)
        window = scores[:n]
        idx = np.argpartition(window, n - k)[n - k :] if k < n else np.arange(n)
        idx = idx[np.argsort(window[idx])[::-1]]
        out: list[tuple[float, Chunk]] = []
        for i in idx:
            score = float(window[i])
            if score <= 0.0:
                break  # idx is descending, so the first non-positive ends the useful run
            out.append((score, chunks[int(i)]))
        return out

    def _candidate_langs(self, query: str) -> list[Lang]:
        """Which language indexes could possibly score this query above zero.

        Scripts do not overlap: a Devanagari query shares no vocabulary with a Tamil matrix,
        so scoring it there costs a full pass over that matrix to produce all-zero scores.
        Detection is by dominant script, which is exact for these corpora -- unlike statistical
        language ID, this cannot be wrong about Hindi vs Tamil, only about Hindi vs Marathi
        (same script), and both of those are searched together anyway.
        """
        available = list(self._indexes)
        if len(available) <= 1:
            return available
        scripts = {LANG_SCRIPT[lang] for lang in available}
        best_script, best_count = "", 0
        for script in scripts:
            count = len(_SCRIPT_RE[script].findall(query))
            if count > best_count:
                best_script, best_count = script, count
        if not best_script:
            return available  # digits/punctuation only: no script to go on, try everything
        matching = [lang for lang in available if LANG_SCRIPT[lang] == best_script]
        return matching or available

    def _load(self) -> None:
        manifest_file = self.path / _MANIFEST_FILE
        if not manifest_file.exists():
            return
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("bm25 manifest unreadable (%s); starting with an empty index", exc)
            return
        if manifest.get("tokenizer") != TOKENIZER_VERSION:
            _LOG.warning(
                "bm25 index at %s was built by tokenizer v%s but this build is v%d; "
                "ignoring it -- re-run the ingest to rebuild",
                self.path,
                manifest.get("tokenizer"),
                TOKENIZER_VERSION,
            )
            return

        for code, expected in (manifest.get("langs") or {}).items():
            try:
                lang = Lang(code)
            except ValueError:
                _LOG.warning("bm25 manifest names unknown language %r; skipped", code)
                continue
            target = self.path / code
            chunks = self._load_chunks(target / _CHUNKS_FILE)
            if chunks is None:
                continue
            try:
                retriever = bm25s.BM25.load(
                    target, mmap=_MMAP, load_corpus=False, show_progress=False
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                _LOG.warning("bm25 index for %s failed to load (%s); skipped", code, exc)
                continue
            n_docs = int(retriever.scores.get("num_docs", 0))
            if n_docs != len(chunks) or len(chunks) != expected:
                # Document ids are positional; a mismatch means every hit would hydrate the
                # wrong chunk. Rebuilding from the sidecar is the only safe recovery.
                _LOG.warning(
                    "bm25 %s is inconsistent (matrix=%d sidecar=%d manifest=%s); rebuilding it",
                    code,
                    n_docs,
                    len(chunks),
                    expected,
                )
                self._adopt(lang, chunks)
                self._dirty.add(lang)
                continue
            self._adopt(lang, chunks)
            self._indexes[lang] = _LangIndex(lang=lang, retriever=retriever, chunks=chunks)

    def _adopt(self, lang: Lang, chunks: list[Chunk]) -> None:
        self._chunks[lang] = chunks
        self._pos[lang] = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}

    @staticmethod
    def _load_chunks(path: Path) -> list[Chunk] | None:
        if not path.exists():
            _LOG.warning("bm25 sidecar %s is missing; that language is skipped", path)
            return None
        chunks: list[Chunk] = []
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        chunks.append(Chunk.model_validate_json(line))
                    except ValidationError:
                        # Positional ids again: a skipped line would shift every later
                        # document, so a corrupt sidecar invalidates the whole language.
                        _LOG.warning("bm25 sidecar %s has a corrupt record; skipped", path)
                        return None
        except OSError as exc:
            _LOG.warning("bm25 sidecar %s unreadable (%s); skipped", path, exc)
            return None
        return chunks

    def _flush_at_exit(self) -> None:
        if not self._unsaved:
            return
        try:
            self.save()
        except Exception as exc:  # interpreter shutdown must never raise out of atexit
            _LOG.warning("bm25 flush on exit failed: %s", exc)


_INSTANCE: BM25Index | None = None
_INSTANCE_LOCK = threading.Lock()


def get_bm25_index() -> BM25Index:
    """Process-wide BM25 index.

    Constructing one reads every language's matrix and sidecar off disk; doing that per
    request would put an index load inside the 200ms budget, so it happens exactly once.
    """
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = BM25Index()
    return _INSTANCE


__all__ = ["TOKENIZER_VERSION", "BM25Index", "get_bm25_index", "normalize", "tokenize"]
