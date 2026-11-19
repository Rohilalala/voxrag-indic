# ruff: noqa: E402 -- the offline env vars below must be set before `voxrag.chunking.base`
# is imported, because that module resolves the e5 tokenizer at *import* time.
"""Shared fixtures for the whole voxrag test-suite.

Three rules this file exists to enforce, because eight test modules import from it:

**Hermetic.** No network, no Qdrant, no Ollama, no HuggingFace download. The `block_network`
fixture below is autouse, so a test that reaches for a socket fails with an explanation
instead of hanging on a CI box with no egress. Mark a test `@pytest.mark.allow_network` to
opt out (nothing in this suite should need to).

**Deterministic.** `FakeEmbedder` is a hash-based stand-in for the ONNX encoder: same text
in, same unit vector out, no model load, ~microseconds. It is *not* random per instance --
two FakeEmbedders built in different tests agree, which is what makes an indexed vector and
a query vector comparable across fixtures.

**Real shapes.** Every fake here implements the Protocol from `voxrag.contracts` it stands
in for, and the models handed out are real pydantic `Document`/`Chunk`/`ScoredChunk`
instances. A test that passes against these fakes is testing the same contract the
production objects satisfy.

Fixture map
-----------
Documents      `hi_doc` `ta_doc` `bn_doc` `en_doc` `sample_docs` `labelled_docs` `make_doc`
Chunks         `make_chunk` `make_scored`
Fakes          `fake_embedder` `fake_store` `fake_lexical`
Utilities      `patch_settings`
Autouse        `block_network` `clear_traces`
"""

from __future__ import annotations

import os

# setdefault, not assignment: a caller who deliberately wants online behaviour (rebuilding
# the HF cache, say) can still export these themselves.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import hashlib
import socket
from collections.abc import Callable, Iterator
from functools import lru_cache
from typing import Any

import numpy as np
import pytest
import regex

from voxrag.config import settings
from voxrag.contracts import Chunk, ChunkStrategyName, Document, Lang, ScoredChunk
from voxrag.telemetry import TRACES

# --------------------------------------------------------------------------------------
# Corpus: four handcrafted passages with known content
# --------------------------------------------------------------------------------------
#
# Shaped like the real thing: MSMARCO-XI carries one Document per *passage*, no title and no
# url (both columns are absent from the parquet), a `source_query`, the row's gold `Answer`,
# and `is_selected` saying whether this passage answers that query. Each passage below is
# three or four sentences, terminated the way its script really terminates -- danda for
# Devanagari and Bengali, full stop for Tamil and Latin -- so sentence-based strategies and
# the script-aware tokeniser both get exercised by the default fixtures.
#
# HI and EN are deliberately about the same fact (New Delhi), which is what makes a
# cross-lingual retrieval assertion possible; TA and BN are distinct topics so a ranking
# test has real negatives to separate.

HI_TEXT = (
    "भारत की राजधानी नई दिल्ली है। यह देश के उत्तरी भाग में यमुना नदी के किनारे स्थित है। "
    "दिल्ली महानगर क्षेत्र में लगभग दो करोड़ लोग रहते हैं। यहाँ का लाल किला विश्व धरोहर स्थल है।"
)
TA_TEXT = (
    "தமிழ்நாட்டின் தலைநகரம் சென்னை ஆகும். இது வங்காள விரிகுடாவின் கரையில் அமைந்துள்ளது. "
    "சென்னையில் உள்ள மெரினா கடற்கரை உலகின் நீளமான கடற்கரைகளில் ஒன்றாகும்."
)
BN_TEXT = (
    "বাংলাদেশের রাজধানী ঢাকা। এই শহরটি বুড়িগঙ্গা নদীর তীরে অবস্থিত। "
    "ঢাকায় প্রায় দুই কোটি মানুষ বাস করে। লালবাগ কেল্লা এখানকার একটি ঐতিহাসিক স্থাপনা।"
)
EN_TEXT = (
    "The capital of India is New Delhi. It lies in the northern part of the country, "
    "on the bank of the Yamuna. About twenty million people live in the Delhi metropolitan "
    "area. The Red Fort here is a World Heritage Site."
)


def _doc(
    doc_id: str,
    text: str,
    lang: Lang,
    *,
    source_query: str,
    gold_answer: str,
    is_selected: bool = True,
) -> Document:
    return Document(
        doc_id=doc_id,
        text=text,
        lang=lang,
        source_query=source_query,
        gold_answer=gold_answer,
        is_selected=is_selected,
    )


@pytest.fixture
def hi_doc() -> Document:
    """Hindi passage: New Delhi is the capital of India. Danda-terminated, 4 sentences."""
    return _doc(
        "hi-0001",
        HI_TEXT,
        Lang.HIN,
        source_query="भारत की राजधानी क्या है",
        gold_answer="नई दिल्ली",
    )


@pytest.fixture
def ta_doc() -> Document:
    """Tamil passage: Chennai is the capital of Tamil Nadu. Period-terminated, 3 sentences."""
    return _doc(
        "ta-0001",
        TA_TEXT,
        Lang.TAM,
        source_query="தமிழ்நாட்டின் தலைநகரம் எது",
        gold_answer="சென்னை",
    )


@pytest.fixture
def bn_doc() -> Document:
    """Bengali passage: Dhaka is the capital of Bangladesh. Danda-terminated, 4 sentences."""
    return _doc(
        "bn-0001",
        BN_TEXT,
        Lang.BEN,
        source_query="বাংলাদেশের রাজধানী কোথায়",
        gold_answer="ঢাকা",
    )


@pytest.fixture
def en_doc() -> Document:
    """English passage stating the same fact as `hi_doc`, for cross-lingual assertions."""
    return _doc(
        "en-0001",
        EN_TEXT,
        Lang.ENG,
        source_query="what is the capital of india",
        gold_answer="New Delhi",
    )


@pytest.fixture
def sample_docs(hi_doc: Document, ta_doc: Document, bn_doc: Document, en_doc: Document) -> list[Document]:
    """The four language fixtures as one list, in hi/ta/bn/en order."""
    return [hi_doc, ta_doc, bn_doc, en_doc]


@pytest.fixture
def labelled_docs(hi_doc: Document) -> list[Document]:
    """One query, four Hindi passages, `is_selected` as ground truth: 2 relevant, 2 not.

    This is the shape retrieval evaluation needs and the only honest source of relevance
    labels in this corpus -- for a given `source_query`, the passages with `is_selected=1`
    are relevant and every other passage is not. Do not label by hand anywhere else.
    """
    query = hi_doc.source_query or ""
    return [
        hi_doc,
        _doc(
            "hi-0002",
            "नई दिल्ली भारत की राजधानी है और यह केंद्र सरकार का मुख्यालय है। "
            "संसद भवन और राष्ट्रपति भवन इसी शहर में हैं।",
            Lang.HIN,
            source_query=query,
            gold_answer="नई दिल्ली",
            is_selected=True,
        ),
        _doc(
            "hi-0003",
            "मुंबई महाराष्ट्र की राजधानी है और यह देश का आर्थिक केंद्र मानी जाती है। "
            "यहाँ बॉलीवुड फिल्म उद्योग स्थित है।",
            Lang.HIN,
            source_query=query,
            gold_answer="नई दिल्ली",
            is_selected=False,
        ),
        _doc(
            "hi-0004",
            "आम भारत का एक लोकप्रिय फल है। गर्मियों के मौसम में इसकी कई किस्में बाजार में आती हैं।",
            Lang.HIN,
            source_query=query,
            gold_answer="नई दिल्ली",
            is_selected=False,
        ),
    ]


@pytest.fixture
def make_doc() -> Callable[..., Document]:
    """Factory for one-off Documents: `make_doc(text, Lang.HIN, title="...")`.

    doc_id is auto-numbered per call unless given, so two documents built from the same text
    are still distinct records -- which is exactly the case chunk-id collision tests need.
    """
    counter = {"n": 0}

    def build(text: str, lang: Lang = Lang.HIN, *, doc_id: str | None = None, **fields: Any) -> Document:
        counter["n"] += 1
        return Document(
            doc_id=doc_id or f"{lang.value}-test-{counter['n']:04d}",
            text=text,
            lang=lang,
            **fields,
        )

    return build


# --------------------------------------------------------------------------------------
# Chunk / ScoredChunk factories
# --------------------------------------------------------------------------------------


@pytest.fixture
def make_chunk() -> Callable[..., Chunk]:
    """Factory for a standalone Chunk, for tests downstream of chunking.

    `n_tokens` defaults to a whitespace word count -- enough for code that only reads it,
    and deliberately not the real tokeniser, which would tie every store/generator test to
    the HF cache. A test that needs the true count should call
    `voxrag.chunking.base.count_tokens` itself.
    """
    counter = {"n": 0}

    def build(
        text: str,
        *,
        doc_id: str = "doc-0001",
        lang: Lang = Lang.HIN,
        strategy: ChunkStrategyName = ChunkStrategyName.PASSAGE_NATIVE,
        ordinal: int | None = None,
        chunk_id: str | None = None,
        **fields: Any,
    ) -> Chunk:
        counter["n"] += 1
        index = counter["n"] - 1 if ordinal is None else ordinal
        return Chunk(
            chunk_id=chunk_id or f"{doc_id}:{strategy.value}:{index}",
            doc_id=doc_id,
            text=text,
            lang=lang,
            strategy=strategy,
            ordinal=index,
            char_start=fields.pop("char_start", 0),
            char_end=fields.pop("char_end", len(text)),
            n_tokens=fields.pop("n_tokens", len(text.split())),
            **fields,
        )

    return build


@pytest.fixture
def make_scored(make_chunk: Callable[..., Chunk]) -> Callable[..., ScoredChunk]:
    """Factory for a ScoredChunk from either an existing Chunk or raw text."""

    def build(chunk: Chunk | str, score: float = 1.0, **fields: Any) -> ScoredChunk:
        if isinstance(chunk, str):
            chunk = make_chunk(chunk)
        return ScoredChunk(chunk=chunk, score=score, **fields)

    return build


# --------------------------------------------------------------------------------------
# FakeEmbedder -- deterministic, hash-based, no model
# --------------------------------------------------------------------------------------

#: Word-ish runs including combining marks, so Indic clusters survive intact. `\w` would
#: cut Devanagari at every matra (category Mn), producing token sets no human would call
#: words and making Hindi look less self-similar than English.
_TOKEN_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+")


@lru_cache(maxsize=8192)
def _token_vector(token: str, dim: int) -> np.ndarray:
    """A fixed random unit vector for one token, seeded from its digest.

    blake2b rather than `hash()`: PYTHONHASHSEED randomises str hashing per process, so a
    `hash()`-seeded fixture would give different vectors on every run and any similarity
    assertion built on it would be flaky by construction.
    """
    seed = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")
    vec = np.random.default_rng(seed).standard_normal(dim, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    vec.flags.writeable = False  # shared across calls; a mutation here would poison the cache
    return vec


class FakeEmbedder:
    """Stand-in for the ONNX e5 encoder. Implements `voxrag.contracts.Embedder`.

    A text's vector is the L2-normalised sum of its tokens' fixed random unit vectors, so
    similarity behaves the way retrieval tests need it to: texts sharing vocabulary score
    high, unrelated texts score near zero, and the same string always gives the same vector.
    It is bag-of-words, so it cannot model word order or cross-lingual meaning -- do not
    write a test whose assertion depends on either.

    `is_query` is recorded in `.calls` but does not change the vector: the real encoder's
    "query: "/"passage: " prefixes are handled inside the encoder, and a fake that moved
    queries into a different subspace would make every retrieval fixture return nothing.
    """

    name = "fake-hash-embedder"

    def __init__(self, dim: int | None = None) -> None:
        self.dim = int(dim if dim is not None else settings.embed_dim)
        #: (texts, is_query) per encode call, for tests asserting batching or prefix use.
        self.calls: list[tuple[tuple[str, ...], bool]] = []

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        self.calls.append((tuple(texts), is_query))
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(text) for text in texts]).astype(np.float32)

    async def encode_async(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        return self.encode(texts, is_query=is_query)

    def _vector(self, text: str) -> np.ndarray:
        tokens = _TOKEN_RE.findall(text.casefold())
        if not tokens:
            # Punctuation-only or empty input still has to yield a unit vector: a zero row
            # would make cosine similarity NaN and every downstream ranking undefined.
            return _token_vector(text.casefold(), self.dim)
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokens:
            vec += _token_vector(token, self.dim)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:  # astronomically unlikely cancellation, but NaN is not an option
            return _token_vector(text.casefold(), self.dim)
        return (vec / norm).astype(np.float32)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """A `FakeEmbedder` at `settings.embed_dim`. Fresh per test so `.calls` is per-test."""
    return FakeEmbedder()


# --------------------------------------------------------------------------------------
# FakeVectorStore -- exact search over a dict
# --------------------------------------------------------------------------------------


class FakeVectorStore:
    """In-memory `voxrag.contracts.VectorStore`: exact cosine, no ANN, no server.

    Exact rather than approximate on purpose. hnswlib and Qdrant are both allowed to miss a
    neighbour; a test asserting "the right chunk comes back first" against an ANN index is
    asserting a probability. Recall behaviour belongs in a benchmark, ranking correctness
    belongs here.
    """

    name = "fake-store"

    def __init__(self, dim: int | None = None) -> None:
        self.dim = int(dim if dim is not None else settings.embed_dim)
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, np.ndarray] = {}

    async def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] != len(chunks):
            raise ValueError(f"vectors shape {vecs.shape} does not match {len(chunks)} chunks")
        if chunks and vecs.shape[1] != self.dim:
            raise ValueError(f"vectors have dim {vecs.shape[1]}, index expects {self.dim}")
        for chunk, vec in zip(chunks, vecs, strict=True):
            self._chunks[chunk.chunk_id] = chunk  # upsert: same id replaces
            self._vectors[chunk.chunk_id] = vec

    async def search(
        self,
        vector: np.ndarray,
        *,
        top_k: int = 20,
        lang: Lang | None = None,
        strategy: ChunkStrategyName | None = None,
    ) -> list[ScoredChunk]:
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(f"query vector has dim {query.shape[0]}, index expects {self.dim}")
        if top_k <= 0:
            return []
        norm = float(np.linalg.norm(query)) or 1.0
        scored: list[tuple[float, str]] = []
        for chunk_id, chunk in self._chunks.items():
            if lang is not None and chunk.lang is not lang:
                continue
            if strategy is not None and chunk.strategy is not strategy:
                continue
            vec = self._vectors[chunk_id]
            denom = norm * (float(np.linalg.norm(vec)) or 1.0)
            scored.append((float(query @ vec) / denom, chunk_id))
        # chunk_id breaks score ties so the order is total and reproducible.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [
            ScoredChunk(
                chunk=self._chunks[chunk_id],
                score=score,
                dense_score=score,
                dense_rank=rank,
            )
            for rank, (score, chunk_id) in enumerate(scored[:top_k], start=1)
        ]

    async def count(self) -> int:
        return len(self._chunks)

    async def health(self) -> bool:
        return True


@pytest.fixture
def fake_store() -> FakeVectorStore:
    """An empty `FakeVectorStore` at `settings.embed_dim`."""
    return FakeVectorStore()


# --------------------------------------------------------------------------------------
# FakeLexicalIndex -- token overlap instead of BM25
# --------------------------------------------------------------------------------------


class FakeLexicalIndex:
    """In-memory `voxrag.contracts.LexicalIndex` scoring by token overlap.

    Not BM25: the scores are not comparable to the real index's and no test should assert
    their magnitude. What it faithfully reproduces is the *shape* fusion depends on -- an
    ordered candidate list with `lexical_score` and 1-based `lexical_rank` populated, empty
    when nothing matches -- without needing bm25s, PyStemmer or a built index on disk.
    """

    name = "fake-lexical"

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)  # replaces, like a rebuilt BM25 index

    def search(self, query: str, *, top_k: int = 20, lang: Lang | None = None) -> list[ScoredChunk]:
        terms = set(_TOKEN_RE.findall(query.casefold()))
        if not terms or top_k <= 0:
            return []
        hits: list[tuple[float, Chunk]] = []
        for chunk in self._chunks:
            if lang is not None and chunk.lang is not lang:
                continue
            tokens = set(_TOKEN_RE.findall(chunk.text.casefold()))
            overlap = len(terms & tokens) / len(terms)
            if overlap > 0:
                hits.append((overlap, chunk))
        hits.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        return [
            ScoredChunk(chunk=chunk, score=score, lexical_score=score, lexical_rank=rank)
            for rank, (score, chunk) in enumerate(hits[:top_k], start=1)
        ]


@pytest.fixture
def fake_lexical() -> FakeLexicalIndex:
    """An empty `FakeLexicalIndex`; call `.index(chunks)` before searching."""
    return FakeLexicalIndex()


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


@pytest.fixture
def patch_settings(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Override `voxrag.config.settings` for one test: `patch_settings(context_top_k=1)`.

    Every module does `from voxrag.config import settings`, i.e. they all hold the same
    object, so setting an attribute on it is visible everywhere immediately. monkeypatch
    restores the original values at teardown; an unknown key raises rather than silently
    creating a field nothing reads.
    """

    def apply(**overrides: Any) -> None:
        for key, value in overrides.items():
            if not hasattr(settings, key):
                raise AttributeError(f"Settings has no field {key!r}")
            monkeypatch.setattr(settings, key, value)

    return apply


@pytest.fixture(autouse=True)
def clear_traces() -> Iterator[None]:
    """Empty the process-wide trace ring around every test.

    `telemetry.TRACES` is a module-level singleton the harness appends to on every request.
    Without this, one module's pipeline tests leak into another's percentile assertions and
    the failure looks like a bug in `report()`.
    """
    TRACES.clear()
    yield
    TRACES.clear()


@pytest.fixture(autouse=True)
def block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn any outbound TCP connect into an immediate, explanatory failure.

    The suite has to pass with the wifi off and on a box with no Qdrant and no Ollama. A
    forgotten real call would otherwise show up as a timeout minutes later, or -- worse --
    as a test that only passes on the machine that has the service running.

    Only `connect` is blocked, not socket creation: `socketpair` (asyncio's self-pipe) and
    plain `socket()` construction stay working. respx is unaffected, since it intercepts
    inside httpx's transport and never reaches this layer.
    """
    if "allow_network" in request.keywords:
        return

    def deny(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "network access is blocked in tests: mock it with respx, or use fake_store / "
            "fake_lexical / fake_embedder. Mark the test @pytest.mark.allow_network only "
            "if it genuinely must reach a socket."
        )

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "allow_network: exempt this test from the autouse socket block"
    )
    config.addinivalue_line(
        "markers",
        "real_embedder: use the real ONNX encoder instead of FakeEmbedder, for assertions "
        "a bag-of-words fake cannot decide (e.g. where a semantic boundary belongs)",
    )


#: What a test module may `from conftest import ...`; everything else is a fixture.
__all__ = [
    "BN_TEXT",
    "EN_TEXT",
    "HI_TEXT",
    "TA_TEXT",
    "FakeEmbedder",
    "FakeLexicalIndex",
    "FakeVectorStore",
]
