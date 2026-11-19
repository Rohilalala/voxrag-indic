"""Chunking invariants, for all eight strategies, through the registry.

The properties asserted here are the ones the rest of the system silently depends on:

* **offsets** -- `doc.text[chunk.char_start:chunk.char_end]` must reproduce the chunk. Every
  citation the UI highlights is that slice; if it drifts, the answer points at the wrong
  words and nothing downstream can detect it;
* **ids** -- stable across runs (an index rebuild must upsert, not duplicate) and unique
  across documents (a collision overwrites someone else's vector);
* **budgets** -- a chunk over the encoder window is embedded from its prefix only, so it is
  indexed under a vector that does not represent its own tail;
* **coverage** -- text that falls between chunks is unretrievable, and nothing reports it;
* **graphemes** -- a boundary inside an Indic cluster corrupts both retrieval and display,
  and is invisible in a diff.

Tests are parametrised over `ChunkStrategyName` and loop the corpus inside, so a failure
names both the strategy and the document. SEMANTIC is pinned to the deterministic
`fake_embedder` (see `deterministic_semantic`) -- otherwise it would load a 118M-parameter
ONNX model and its boundaries would depend on the HF cache.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator, Sequence

import pytest

from conftest import BN_TEXT, EN_TEXT, HI_TEXT, TA_TEXT, FakeEmbedder
from voxrag.chunking import REGISTRY, get_chunker
from voxrag.chunking import base as chunk_base
from voxrag.chunking import semantic as semantic_module
from voxrag.chunking.base import (
    count_tokens,
    embed_budget_tokens,
    token_index,
    tokenizer_available,
)
from voxrag.chunking.metadata_aware import _MIN_BODY_TOKENS
from voxrag.config import settings
from voxrag.contracts import Chunk, Chunker, ChunkStrategyName, Document, Lang

STRATEGIES = list(ChunkStrategyName)

#: Slack allowed between the size of the *cut* and the recounted size of the chunk.
#:
#: Strategies cut on the document's own token index, then `Chunk.n_tokens` is recounted on
#: the sliced substring. SentencePiece prepends a metaspace piece to a slice that begins
#: mid-word, so the recount can come out one token above the cut -- see
#: `test_recorded_token_count_tracks_the_cut`, which pins the drift at exactly this. The cut
#: itself is asserted against the budget with no slack at all.
_RETOKENISE_SLACK = 1

#: Unicode categories that cannot legally start or follow a boundary: a non-spacing mark
#: (matra, nukta, virama sign) or a spacing combining mark belongs to the consonant before
#: it. Plus the two joiners, which bind the clusters on either side of them.
_CLUSTER_INTERNAL_CATEGORIES = frozenset({"Mn", "Mc"})
_ZWNJ = "‌"
_ZWJ = "‍"


# --------------------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------------------

#: Devanagari with danda, U+0964. A splitter that only knows ".!?" returns one sentence.
MR_TEXT = (
    "महाराष्ट्राची राजधानी मुंबई आहे। मुंबई ही देशाची आर्थिक राजधानी मानली जाते। "
    "येथे गेटवे ऑफ इंडिया हे प्रसिद्ध स्मारक आहे।"
)
#: Urdu with the Arabic full stop, U+06D4, and Arabic commas at the clause rung.
UR_TEXT = (
    "پاکستان کا دارالحکومت اسلام آباد ہے۔ یہ شہر مارگلہ کی پہاڑیوں کے دامن میں واقع ہے۔ "
    "یہاں فیصل مسجد اور ایوان صدر موجود ہیں۔"
)
#: Explicit ZWJ/ZWNJ. "क्‍ष" carries a ZWJ after the virama, "ज्ञ‌ान" a ZWNJ, and the
#: Bengali "র‍্য" (ro + ZWJ + jophola) only renders correctly if the joiner survives.
ZWJ_TEXT = (
    "क्‍ष और ज्ञ‌ान दोनों संयुक्ताक्षर हैं। हिन्दी में र्‍ह जैसे रूप भी मिलते हैं। "
    "বাংলায় র‍্যাব এবং ক্‍ষ যুক্তাক্ষর ব্যবহৃত হয়। தமிழில் க்‍ஷ என்ற எழுத்தும் உண்டு."
)

CORPUS: dict[str, Document] = {
    "hi_passage": Document(doc_id="hi-1", text=HI_TEXT, lang=Lang.HIN, source_query="भारत की राजधानी"),
    "ta_passage": Document(doc_id="ta-1", text=TA_TEXT, lang=Lang.TAM, source_query="சென்னை"),
    "bn_passage": Document(doc_id="bn-1", text=BN_TEXT, lang=Lang.BEN, source_query="ঢাকা"),
    "en_passage": Document(doc_id="en-1", text=EN_TEXT, lang=Lang.ENG, source_query="capital of india"),
    "mr_danda": Document(doc_id="mr-1", text=MR_TEXT, lang=Lang.MAR),
    "ur_full_stop": Document(doc_id="ur-1", text=UR_TEXT, lang=Lang.URD),
    "joiners": Document(doc_id="zwj-1", text=ZWJ_TEXT, lang=Lang.HIN),
    # Long enough that every strategy has to cut it more than once.
    "hi_long": Document(
        doc_id="hi-2",
        text=(HI_TEXT + " ") * 6,
        lang=Lang.HIN,
        title="दिल्ली",
        source_query="भारत की राजधानी क्या है",
    ),
    "en_long": Document(
        doc_id="en-2",
        text=(EN_TEXT + " ") * 6,
        lang=Lang.ENG,
        title="New Delhi",
        source_query="what is the capital of india",
    ),
    # Blank lines and hard line breaks: the top two rungs of the RECURSIVE ladder.
    "paragraphs": Document(
        doc_id="para-1",
        text="\n\n".join(f"यह अनुच्छेद संख्या {i} है। इसमें दो वाक्य हैं।" for i in range(12)),
        lang=Lang.HIN,
    ),
    "lines": Document(
        doc_id="line-1",
        text="\n".join(f"पंक्ति संख्या {i} यहाँ लिखी है" for i in range(30)),
        lang=Lang.HIN,
    ),
    # Periods that do not end sentences: abbreviations, initialisms, decimals, a URL.
    "abbreviations": Document(
        doc_id="abbr-1",
        text=(
            "Dr. J. K. Rao paid Rs. 3.5 lakh for it. See https://example.com/a?b=1 for the "
            "details. The U.S. team, i.e. the visitors, arrived at 9 a.m. on Mon. "
        )
        * 4,
        lang=Lang.ENG,
    ),
    "mixed_script": Document(
        doc_id="mix-1",
        text=("भारत India देश। தமிழ் Tamil language. বাংলা text এখানে। اردو متن یہاں۔ ") * 8,
        lang=Lang.HIN,
    ),
}

DEGENERATE: dict[str, Document] = {
    "empty": Document(doc_id="deg-empty", text="", lang=Lang.HIN),
    "whitespace_only": Document(doc_id="deg-ws", text="   \n\n \t ", lang=Lang.HIN),
    "one_character": Document(doc_id="deg-one", text="अ", lang=Lang.HIN),
    "one_terminator": Document(doc_id="deg-danda", text="।", lang=Lang.HIN),
    "punctuation_only": Document(doc_id="deg-punct", text="।। ... !!! ?? -- ,,", lang=Lang.HIN),
    # 1200 words, not one sentence terminator anywhere: common in this corpus.
    "no_terminator": Document(
        doc_id="deg-noterm", text=" ".join(["भारत"] * 1200), lang=Lang.HIN
    ),
    "single_10k_sentence": Document(
        doc_id="deg-10k", text=("क" * 5000) + ("a" * 5000) + "।", lang=Lang.HIN
    ),
}

#: Documents with no sentence terminator at all, exercised by the danda tests below.
INDIC_TERMINATOR_CASES = [
    pytest.param(CORPUS["hi_passage"], "।", 4, id="hindi-danda"),
    pytest.param(CORPUS["mr_danda"], "।", 3, id="marathi-danda"),
    pytest.param(CORPUS["bn_passage"], "।", 4, id="bengali-danda"),
    pytest.param(CORPUS["ur_full_stop"], "۔", 3, id="urdu-full-stop"),
]


# --------------------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def deterministic_semantic(
    request: pytest.FixtureRequest, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin SEMANTIC to the fake embedder.

    Left alone, `_resolve_embedder` imports `voxrag.embed` and loads the real ONNX graph:
    seconds of startup, and boundaries that silently change with the model cache. Seeding the
    resolver's memo keeps the dense code path under test while making it deterministic.

    Opt out with `@pytest.mark.real_embedder` when a test asserts something only a real
    encoder can decide. `FakeEmbedder` is bag-of-words, so it ranks candidate boundaries
    differently from a sentence encoder -- fine for exercising the code path, useless for
    judging whether a boundary landed at the right place.
    """
    if "real_embedder" in request.keywords:
        return
    monkeypatch.setattr(semantic_module, "_embedder_cache", fake_embedder)
    monkeypatch.setattr(semantic_module, "_embedder_resolved", True)


def chunk_all(strategy: ChunkStrategyName, docs: Sequence[Document]) -> Iterator[tuple[Document, list[Chunk]]]:
    chunker = get_chunker(strategy)
    for doc in docs:
        yield doc, chunker.chunk(doc)


def body_of(chunk: Chunk) -> str:
    """The document substring this chunk was cut from.

    For every strategy but METADATA_AWARE that is `chunk.text`. METADATA_AWARE prepends a
    "Title | Language | query" header to the embedded string, so its document-derived body
    is `context_text` -- which is also, deliberately, the only part the generator sees.
    """
    if chunk.strategy is ChunkStrategyName.METADATA_AWARE:
        assert chunk.context_text is not None, "METADATA_AWARE must always set context_text"
        return chunk.context_text
    return chunk.text


def budget_of(chunk: Chunk) -> int | None:
    """The token ceiling the chunk's strategy promises, or None where it promises none.

    PASSAGE_NATIVE is the None: it indexes the corpus's own segmentation as-is and reports
    over-long passages through `meta["over_encoder_window"]` instead of splitting them.
    """
    strategy = chunk.strategy
    if strategy in (
        ChunkStrategyName.FIXED,
        ChunkStrategyName.FIXED_OVERLAP,
        ChunkStrategyName.RECURSIVE,
    ):
        return settings.chunk_target_tokens
    if strategy is ChunkStrategyName.SENTENCE_WINDOW:
        return embed_budget_tokens()
    if strategy is ChunkStrategyName.SEMANTIC:
        return max(settings.semantic_max_tokens, settings.semantic_min_tokens)
    if strategy is ChunkStrategyName.PARENT_CHILD:
        return settings.child_chunk_tokens
    if strategy is ChunkStrategyName.METADATA_AWARE:
        # The header is not part of the body budget; the body gets what is left of the
        # target, down to a floor.
        return max(_MIN_BODY_TOKENS, settings.chunk_target_tokens - chunk.meta["header_tokens"])
    return None


def covered_positions(chunks: Sequence[Chunk]) -> set[int]:
    positions: set[int] = set()
    for chunk in chunks:
        positions.update(range(chunk.char_start, chunk.char_end))
    return positions


def is_cluster_internal(char: str) -> bool:
    return unicodedata.category(char) in _CLUSTER_INTERNAL_CATEGORIES or char in (_ZWJ, _ZWNJ)


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------


def test_registry_covers_every_strategy_exactly_once() -> None:
    assert set(REGISTRY) == set(ChunkStrategyName)
    for name, chunker in REGISTRY.items():
        # A chunker filed under the wrong key would stamp the wrong `strategy` onto every
        # chunk, and the store's strategy filter would then silently return nothing.
        assert chunker.name is name
        assert isinstance(chunker, Chunker)


def test_get_chunker_resolves_names_values_and_the_configured_default() -> None:
    for name in ChunkStrategyName:
        assert get_chunker(name) is REGISTRY[name]
        assert get_chunker(name.value) is REGISTRY[name]
    assert get_chunker(None) is REGISTRY[settings.active_strategy]
    with pytest.raises(KeyError):
        get_chunker("no_such_strategy")


# --------------------------------------------------------------------------------------
# Offsets
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.value)
def test_offsets_round_trip_to_the_document(strategy: ChunkStrategyName) -> None:
    """`doc.text[char_start:char_end]` recovers the chunk's source text -- every chunk.

    Asserted exhaustively rather than on a sample because this is what makes a citation
    point at real words; an off-by-one here is invisible until a user reads the highlight.
    """
    for doc, chunks in chunk_all(strategy, list(CORPUS.values())):
        for chunk in chunks:
            where = f"{strategy.value}/{doc.doc_id}#{chunk.ordinal}"
            assert 0 <= chunk.char_start < chunk.char_end <= len(doc.text), where
            assert doc.text[chunk.char_start : chunk.char_end] == body_of(chunk), where


def test_metadata_aware_embeds_the_header_but_never_shows_it() -> None:
    """The header is in `text` (embedded) and out of `context_text` (read by the generator).

    If it leaked into the generation text an extractive answer could quote
    "New Delhi | Hindi | what is the capital of india" back at the user as corpus content.
    """
    doc = CORPUS["hi_long"]
    chunks = get_chunker(ChunkStrategyName.METADATA_AWARE).chunk(doc)
    assert chunks
    for chunk in chunks:
        header = chunk.meta["header"]
        body = doc.text[chunk.char_start : chunk.char_end]
        assert chunk.text == f"{header}\n{body}"
        assert chunk.context_text == body
        assert header not in chunk.context_text
        assert chunk.generation_text == body
        assert chunk.meta["header_tokens"] == count_tokens(header)
    # Title and language name reach the embedded string.
    header = chunks[0].meta["header"]
    assert doc.title is not None and doc.title in header
    assert "Hindi" in header
    # The source query does NOT, by default. See test_metadata_source_query_is_opt_in.
    assert doc.source_query is not None and doc.source_query not in header


def test_metadata_source_query_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """`source_query` is excluded by default and included only when explicitly enabled.

    This default is a correctness guard, not a preference. MSMARCO-XI's `source_query` is the
    same string the evaluation grades with, so embedding it into its own passage leaks the
    answer key -- it drove METADATA_AWARE to Recall@10 = 0.996 against 0.764 for the next
    best strategy. The capability is kept for production use (doc2query-style expansion);
    the default keeps the benchmark honest.
    """
    doc = CORPUS["hi_long"]
    assert doc.source_query

    default_header = get_chunker(ChunkStrategyName.METADATA_AWARE).chunk(doc)[0].meta["header"]
    assert doc.source_query not in default_header

    monkeypatch.setattr(settings, "metadata_include_source_query", True)
    opted_in = get_chunker(ChunkStrategyName.METADATA_AWARE).chunk(doc)[0].meta["header"]
    assert doc.source_query in opted_in


# --------------------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.value)
def test_chunking_is_deterministic(strategy: ChunkStrategyName) -> None:
    """Chunking the same Document twice yields byte-identical chunks, ids included.

    Re-indexing relies on this: an id that moved between runs upserts a second copy of the
    same text instead of replacing the first.
    """
    chunker = get_chunker(strategy)
    for doc in CORPUS.values():
        first = [chunk.model_dump() for chunk in chunker.chunk(doc)]
        second = [chunk.model_dump() for chunk in chunker.chunk(doc)]
        assert first == second, f"{strategy.value}/{doc.doc_id} is not deterministic"


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.value)
def test_chunk_ids_are_unique_within_and_across_documents(strategy: ChunkStrategyName) -> None:
    seen: dict[str, str] = {}
    for doc, chunks in chunk_all(strategy, list(CORPUS.values())):
        for chunk in chunks:
            assert chunk.chunk_id not in seen, (
                f"{strategy.value}: {chunk.chunk_id} produced by both "
                f"{seen.get(chunk.chunk_id)} and {doc.doc_id}"
            )
            seen[chunk.chunk_id] = doc.doc_id
            assert chunk.doc_id == doc.doc_id


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.value)
def test_identical_text_under_two_doc_ids_does_not_collide(strategy: ChunkStrategyName) -> None:
    """Two Documents holding the same passage must still get disjoint chunk ids.

    MS MARCO repeats passages across queries, so this is the ordinary case, not an edge one:
    colliding ids would let one row's chunks overwrite another's vectors in the store.
    """
    text = CORPUS["hi_long"].text
    left = Document(doc_id="dup-a", text=text, lang=Lang.HIN)
    right = Document(doc_id="dup-b", text=text, lang=Lang.HIN)
    chunker = get_chunker(strategy)
    left_ids = {chunk.chunk_id for chunk in chunker.chunk(left)}
    right_ids = {chunk.chunk_id for chunk in chunker.chunk(right)}
    assert left_ids and right_ids
    assert left_ids.isdisjoint(right_ids)


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.value)
def test_ordinals_are_zero_based_and_contiguous(strategy: ChunkStrategyName) -> None:
    for doc, chunks in chunk_all(strategy, list(CORPUS.values())):
        ordinals = [chunk.ordinal for chunk in chunks]
        assert ordinals == list(range(len(chunks))), f"{strategy.value}/{doc.doc_id}"


# --------------------------------------------------------------------------------------
# Size
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.value)
def test_no_chunk_is_empty_or_whitespace_only(strategy: ChunkStrategyName) -> None:
    """An all-whitespace chunk still costs a vector and can never be retrieved by anything."""
    for doc, chunks in chunk_all(strategy, list(CORPUS.values())):
        for chunk in chunks:
            assert chunk.text.strip(), f"{strategy.value}/{doc.doc_id}#{chunk.ordinal} is blank"
            assert body_of(chunk).strip()


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.value)
def test_the_cut_never_exceeds_the_strategy_budget(strategy: ChunkStrategyName) -> None:
    """Measured on the document's own token index -- the units the split was made in."""
    for doc, chunks in chunk_all(strategy, list(CORPUS.values())):
        index = token_index(doc.text)
        for chunk in chunks:
            budget = budget_of(chunk)
            if budget is None:
                continue
            size = index.count(chunk.char_start, chunk.char_end)
            assert size <= budget, (
                f"{strategy.value}/{doc.doc_id}#{chunk.ordinal}: cut is {size} tokens, "
                f"budget is {budget}"
            )


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.value)
def test_recorded_token_count_tracks_the_cut(strategy: ChunkStrategyName) -> None:
    """`n_tokens` is the recount of the emitted string, and never drifts from the cut by
    more than one token.

    The two numbers can differ at all because re-encoding a slice is not the same as
    counting tokens inside the document: a slice starting mid-word picks up a leading
    metaspace piece. Bounding the drift is what lets the budget assertions above be trusted
    -- an unbounded drift would mean the strategies are not really sizing in encoder tokens.
    """
    for doc, chunks in chunk_all(strategy, list(CORPUS.values())):
        index = token_index(doc.text)
        for chunk in chunks:
            assert chunk.n_tokens == count_tokens(chunk.text)
            drift = count_tokens(body_of(chunk)) - index.count(chunk.char_start, chunk.char_end)
            assert abs(drift) <= _RETOKENISE_SLACK, (
                f"{strategy.value}/{doc.doc_id}#{chunk.ordinal}: recount drifts {drift} tokens"
            )


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.value)
def test_chunks_over_the_encoder_window_are_flagged(strategy: ChunkStrategyName) -> None:
    """The encoder truncates past `embed_max_tokens`; the flag is what keeps that visible.

    PASSAGE_NATIVE and SEMANTIC can both legitimately exceed the window (the corpus's own
    passages are sometimes long, and `semantic_max_tokens` is deliberately above it), so the
    invariant is not "never over" but "never over without saying so".
    """
    limit = embed_budget_tokens()
    for doc, chunks in chunk_all(strategy, list(CORPUS.values()) + list(DEGENERATE.values())):
        for chunk in chunks:
            flagged = bool(chunk.meta.get("over_encoder_window", False))
            assert flagged == (chunk.n_tokens > limit), (
                f"{strategy.value}/{doc.doc_id}#{chunk.ordinal}: {chunk.n_tokens} tokens, "
                f"limit {limit}, flag {flagged}"
            )


# --------------------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.value)
def test_every_non_whitespace_character_lands_in_some_chunk(strategy: ChunkStrategyName) -> None:
    """No text falls between chunks.

    Whitespace between spans is expected -- `make_span` trims it out of both neighbours and
    adjusts the offsets -- but a dropped word is a fact the index can never return, with
    nothing anywhere reporting its absence.
    """
    for doc, chunks in chunk_all(strategy, list(CORPUS.values())):
        covered = covered_positions(chunks)
        missing = [i for i, char in enumerate(doc.text) if not char.isspace() and i not in covered]
        assert not missing, (
            f"{strategy.value}/{doc.doc_id}: {len(missing)} characters covered by no chunk, "
            f"first at {missing[0]} ({doc.text[missing[0]]!r})"
        )


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.value)
def test_chunk_spans_run_in_document_order_and_only_overlap_by_design(
    strategy: ChunkStrategyName,
) -> None:
    """FIXED_OVERLAP's windows share text on purpose; nobody else's may.

    PARENT_CHILD duplicates too, but in `context_text` (every child carries its whole
    parent), never in the indexed spans -- so its children stay disjoint like everyone
    else's and the duplication costs no extra vectors.
    """
    for doc, chunks in chunk_all(strategy, list(CORPUS.values())):
        starts = [chunk.char_start for chunk in chunks]
        assert starts == sorted(starts), f"{strategy.value}/{doc.doc_id} is out of document order"
        overlaps = [
            (a.ordinal, b.ordinal)
            for a, b in zip(chunks, chunks[1:], strict=False)
            if b.char_start < a.char_end
        ]
        if strategy is not ChunkStrategyName.FIXED_OVERLAP:
            assert not overlaps, f"{strategy.value}/{doc.doc_id} overlaps at {overlaps}"


def test_fixed_overlap_keeps_boundary_spanning_text_intact() -> None:
    """The reason the overlap exists: a fact split by a FIXED boundary survives in
    FIXED_OVERLAP, because consecutive windows share `chunk_overlap_tokens`."""
    doc = Document(doc_id="ov-1", text=(HI_TEXT + " ") * 12, lang=Lang.HIN)
    fixed = get_chunker(ChunkStrategyName.FIXED).chunk(doc)
    overlapped = get_chunker(ChunkStrategyName.FIXED_OVERLAP).chunk(doc)
    assert len(fixed) > 2 and len(overlapped) > 2

    index = token_index(doc.text)
    for previous, current in zip(overlapped, overlapped[1:], strict=False):
        shared = index.count(current.char_start, previous.char_end)
        assert 0 < shared <= settings.chunk_overlap_tokens

    # Text that a FIXED boundary cuts in half is whole inside some FIXED_OVERLAP window.
    for boundary in [chunk.char_end for chunk in fixed[:-1]]:
        straddling = doc.text[boundary - 20 : boundary + 20].strip()
        assert any(straddling in chunk.text for chunk in overlapped), (
            f"text across the fixed boundary at {boundary} survives in no overlapped window"
        )


# --------------------------------------------------------------------------------------
# Strategy-specific contracts
# --------------------------------------------------------------------------------------


def test_parent_child_children_resolve_to_their_parent() -> None:
    """Every child names a parent, children of one parent are contiguous and disjoint, and
    `generation_text` returns the parent -- that asymmetry is the whole strategy."""
    doc = CORPUS["hi_long"]
    chunks = get_chunker(ChunkStrategyName.PARENT_CHILD).chunk(doc)
    assert len(chunks) > 1

    families: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        assert chunk.parent_id, "every parent_child chunk must name its parent"
        families.setdefault(chunk.parent_id, []).append(chunk)

    covered: set[int] = set()
    for parent_id, children in families.items():
        parent_start = children[0].meta["parent_char_start"]
        parent_end = children[0].meta["parent_char_end"]
        parent_text = doc.text[parent_start:parent_end]
        # The id has to be derivable, because parents are never indexed and nothing can look
        # one up by ordinal.
        assert parent_id == f"{doc.doc_id}:{ChunkStrategyName.PARENT_CHILD.value}:parent:{children[0].meta['parent_ordinal']}"
        assert [child.meta["child_index"] for child in children] == list(range(len(children)))
        assert all(child.meta["n_children"] == len(children) for child in children)
        for child in children:
            assert child.meta["parent_char_start"] == parent_start
            assert child.meta["parent_char_end"] == parent_end
            assert parent_start <= child.char_start < child.char_end <= parent_end
            # What the generator reads is the parent, not the child that was matched.
            assert child.context_text == parent_text
            assert child.generation_text == parent_text
            assert child.text in parent_text
            assert child.n_tokens <= settings.child_chunk_tokens
        assert count_tokens(parent_text) <= settings.parent_chunk_tokens
        # Children partition their parent rather than sampling it.
        span = covered_positions(children)
        assert not span & covered
        covered |= span


def test_sentence_window_context_is_a_superset_of_the_indexed_sentence() -> None:
    doc = CORPUS["hi_long"]
    chunks = get_chunker(ChunkStrategyName.SENTENCE_WINDOW).chunk(doc)
    assert len(chunks) > 2 * settings.sentence_window_size

    for chunk in chunks:
        context = chunk.generation_text
        assert chunk.text in context
        assert len(context) >= len(chunk.text)
        start = chunk.meta["context_char_start"]
        end = chunk.meta["context_char_end"]
        assert start <= chunk.char_start < chunk.char_end <= end
        assert doc.text[start:end] == context
        # The window is the configured number of sentences on each side, clamped at the ends.
        lo, hi = chunk.meta["window_start_sentence"], chunk.meta["window_end_sentence"]
        assert hi - lo <= 2 * settings.sentence_window_size
        assert lo <= chunk.ordinal <= hi

    # An interior sentence must actually gain neighbours, otherwise the strategy is just
    # sentence splitting with extra bookkeeping.
    middle = chunks[len(chunks) // 2]
    assert len(middle.generation_text) > len(middle.text)


def test_passage_native_returns_the_passage_unchanged() -> None:
    """One chunk per document, covering all of it: the corpus is already human-segmented."""
    for doc in CORPUS.values():
        chunks = get_chunker(ChunkStrategyName.PASSAGE_NATIVE).chunk(doc)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.text == doc.text.strip()
        assert chunk.ordinal == 0
        assert chunk.context_text is None
        assert chunk.generation_text == chunk.text


@pytest.mark.real_embedder
@pytest.mark.allow_network
def test_semantic_splits_where_the_topic_changes() -> None:
    """A boundary lands at the vocabulary shift, not at an arbitrary token count.

    This is the one chunking test that needs the REAL encoder, and it is marked
    `allow_network` so the model can resolve rather than falling back mid-test.

    `FakeEmbedder` cannot stand in here. It is bag-of-words, so pooling several sentences
    dilutes the topic words against shared function words and moves the detected seam:
    measured on this exact fixture, the fake peaks at gap 2 with a one-sentence window but
    at gap 3 with the two-sentence window the chunker actually uses, while the real encoder
    peaks at gap 2 (the true seam) with two and at gap 4 with one. The two disagree about
    which window is correct, so a fake-backed assertion here would be testing the fake.
    """
    capital = "नई दिल्ली भारत की राजधानी है। दिल्ली में संसद भवन स्थित है। दिल्ली का लाल किला प्रसिद्ध है।"
    fruit = "आम एक मीठा फल है। आम गर्मियों में पकता है। आम की कई किस्में बाजार में मिलती हैं।"
    doc = Document(doc_id="topic-1", text=f"{capital} {fruit}", lang=Lang.HIN)

    chunks = get_chunker(ChunkStrategyName.SEMANTIC).chunk(doc)
    if not chunks or chunks[0].meta.get("semantic_backend") != "dense":
        pytest.skip("real ONNX encoder unavailable; semantic backend fell back to lexical")

    assert len(chunks) > 1, "semantic chunking found no topic boundary between two topics"
    # The seam between the two topics is a real chunk boundary.
    assert any(chunk.char_end == len(capital) for chunk in chunks)


def test_semantic_degrades_to_lexical_distances_without_an_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No embedder is a degradation, not a failure: the strategy still splits, using
    IDF-weighted token overlap, and says so in `meta`."""
    monkeypatch.setattr(semantic_module, "_embedder_cache", None)
    monkeypatch.setattr(semantic_module, "_embedder_resolved", True)

    doc = CORPUS["hi_long"]
    chunks = get_chunker(ChunkStrategyName.SEMANTIC).chunk(doc)
    assert chunks
    for chunk in chunks:
        assert chunk.meta["semantic_backend"] == "lexical_fallback"
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.n_tokens <= budget_of(chunk) + _RETOKENISE_SLACK


# --------------------------------------------------------------------------------------
# Indic correctness
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("doc", "terminator", "n_sentences"), INDIC_TERMINATOR_CASES)
def test_indic_terminators_end_sentences(
    doc: Document, terminator: str, n_sentences: int
) -> None:
    """Danda (U+0964) and the Arabic full stop (U+06D4) end sentences.

    Without them a whole Hindi or Urdu passage is one "sentence", which collapses
    SENTENCE_WINDOW, SEMANTIC, RECURSIVE and PARENT_CHILD into one-chunk-per-document and
    quietly removes four of the eight strategies from the benchmark.
    """
    assert not any(mark in doc.text for mark in ".!?"), "the fixture must not contain a Latin stop"

    chunks = get_chunker(ChunkStrategyName.SENTENCE_WINDOW).chunk(doc)
    assert len(chunks) == n_sentences
    for chunk in chunks:
        assert chunk.text.endswith(terminator)
        # One terminator per chunk: the split happened at every one of them, not just the last.
        assert chunk.text.count(terminator) == 1


@pytest.mark.skipif(
    not tokenizer_available(),
    reason="the character-ratio fallback estimates token spans and makes no grapheme promise",
)
@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.value)
def test_no_chunk_boundary_splits_a_grapheme_cluster(strategy: ChunkStrategyName) -> None:
    """A boundary never lands between a base consonant and its combining marks or joiners.

    "क्‍ष" cut after the virama leaves "क्" at the end of one chunk and "ष" at the start of
    the next: both halves are misspelt words that match nothing, and the damage is invisible
    in a diff because the characters are all still there.
    """
    docs = list(CORPUS.values()) + list(DEGENERATE.values())
    for doc, chunks in chunk_all(strategy, docs):
        for chunk in chunks:
            for edge, position in (("start", chunk.char_start), ("end", chunk.char_end)):
                if not 0 < position < len(doc.text):
                    continue
                char = doc.text[position]
                assert not is_cluster_internal(char), (
                    f"{strategy.value}/{doc.doc_id}#{chunk.ordinal} {edge} at {position} "
                    f"splits a cluster before U+{ord(char):04X}"
                )


def test_joiners_survive_chunking_intact() -> None:
    """Every ZWJ/ZWNJ in the source still sits inside some chunk, in every strategy.

    Guards the fixture as much as the code: a joiner lost to an editor or a copy-paste would
    leave `test_no_chunk_boundary_splits_a_grapheme_cluster` asserting nothing about joiners
    while still passing.
    """
    doc = CORPUS["joiners"]
    joiners = [i for i, char in enumerate(doc.text) if char in (_ZWJ, _ZWNJ)]
    assert len(joiners) >= 4, "the joiner fixture no longer contains joiners"

    for strategy in STRATEGIES:
        covered = covered_positions(get_chunker(strategy).chunk(doc))
        lost = [i for i in joiners if i not in covered]
        assert not lost, f"{strategy.value} dropped joiners at {lost}"


# --------------------------------------------------------------------------------------
# Degenerate input
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.value)
@pytest.mark.parametrize("case", list(DEGENERATE), ids=list(DEGENERATE))
def test_degenerate_documents_keep_every_invariant(
    strategy: ChunkStrategyName, case: str
) -> None:
    """Empty text, one character, no terminator, one enormous sentence, only punctuation.

    None of these may raise, produce a blank chunk, or lose text. They are not hypothetical:
    MSMARCO-XI passages include unpunctuated walls of text and punctuation-only fragments.
    """
    doc = DEGENERATE[case]
    chunks = get_chunker(strategy).chunk(doc)

    if not doc.text.strip():
        assert chunks == [], f"{strategy.value} emitted chunks for blank text"
        return

    assert chunks, f"{strategy.value} dropped {case} entirely"
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)

    covered = covered_positions(chunks)
    for chunk in chunks:
        assert chunk.text.strip()
        assert doc.text[chunk.char_start : chunk.char_end] == body_of(chunk)
        budget = budget_of(chunk)
        if budget is not None:
            assert chunk.n_tokens <= budget + _RETOKENISE_SLACK
    assert all(char.isspace() or i in covered for i, char in enumerate(doc.text))


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.value)
def test_one_enormous_sentence_is_still_split(strategy: ChunkStrategyName) -> None:
    """10,000 characters with no interior boundary must not become one over-long chunk.

    Sentence-based strategies have nothing to split on here, so this exercises the bottom of
    the ladder -- clause, whitespace, then raw token windows. PASSAGE_NATIVE is exempt by
    design and is checked by the encoder-window flag instead.
    """
    doc = DEGENERATE["single_10k_sentence"]
    chunks = get_chunker(strategy).chunk(doc)
    assert chunks

    if strategy is ChunkStrategyName.PASSAGE_NATIVE:
        assert len(chunks) == 1
        assert chunks[0].meta["over_encoder_window"] is True
        return

    budget = budget_of(chunks[0])
    assert budget is not None
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.n_tokens <= budget + _RETOKENISE_SLACK


# --------------------------------------------------------------------------------------
# Tokeniser fallback
# --------------------------------------------------------------------------------------


@pytest.fixture
def without_tokenizer(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Force the character-ratio fallback that runs when the e5 tokenizer cannot be loaded.

    `token_index` is LRU-cached on text, so the cache has to be dropped on the way in *and*
    on the way out; leaving approximate spans cached would corrupt every later test.
    """
    monkeypatch.setattr(chunk_base, "_TOKENIZER", None)
    chunk_base.token_index.cache_clear()
    assert not tokenizer_available()
    yield
    chunk_base.token_index.cache_clear()


@pytest.mark.usefixtures("without_tokenizer")
def test_offsets_stay_exact_without_the_real_tokenizer() -> None:
    """On the fallback the token *count* is an estimate, but the offsets must still be true.

    That is the contract `_approx_token_spans` states, and it is what keeps citations correct
    on a machine with a cold HF cache -- a build that silently produced wrong highlights
    offline would be far worse than one that chunked slightly differently.
    """
    docs = [CORPUS["hi_passage"], CORPUS["en_long"], CORPUS["mixed_script"]]
    for strategy in STRATEGIES:
        for doc, chunks in chunk_all(strategy, docs):
            assert chunks
            covered = covered_positions(chunks)
            for chunk in chunks:
                assert doc.text[chunk.char_start : chunk.char_end] == body_of(chunk)
                assert chunk.text.strip()
            missing = [
                i for i, char in enumerate(doc.text) if not char.isspace() and i not in covered
            ]
            assert not missing, f"{strategy.value}/{doc.doc_id} lost {len(missing)} characters"
