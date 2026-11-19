"""BM25 lexical arm: Indic tokenisation, per-language IDF, ranking and persistence.

The crux here is tokenisation. Every other property of BM25 is arithmetic that numpy gets
right; what actually breaks on this corpus is turning a Devanagari/Tamil/Bengali string into
terms. Two failure modes are silent -- they never raise, they just quietly stop matching:

* the whole line comes back as one token (an ASCII-shaped `\\w+` or `str.split`), so the
  vocabulary has one entry per document and every query scores zero;
* a combining mark is split off its base consonant, so "की" becomes "क" + "ी" and the index
  holds terms that no human ever typed.

So the tokeniser tests assert on concrete, hand-checked token lists rather than on shapes.
Everything below is hermetic: BM25Index takes an explicit path, so nothing touches
`settings.bm25_path` or the 120k-document corpus.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from voxrag.contracts import Chunk, ChunkStrategyName, Lang
from voxrag.store import bm25 as bm25_mod
from voxrag.store.bm25 import TOKENIZER_VERSION, BM25Index, normalize, tokenize

ZWNJ = "‌"
ZWJ = "‍"

# Sentences with a known, hand-counted word breakdown. The danda (।) closes the Hindi and
# Bengali ones on purpose: it is the punctuation that a naive tokeniser fuses across.
HINDI_SENTENCE = "भारत की जनसंख्या लगभग 121 करोड़ है।"
HINDI_TOKENS = ["भारत", "की", "जनसंख्या", "लगभग", "121", "करोड़", "है"]

TAMIL_SENTENCE = "தமிழ்நாட்டின் தலைநகரம் சென்னை ஆகும்."
TAMIL_TOKENS = ["தமிழ்நாட்டின்", "தலைநகரம்", "சென்னை", "ஆகும்"]

BENGALI_SENTENCE = "বাংলাদেশের রাজধানী ঢাকা শহর।"
BENGALI_TOKENS = ["বাংলাদেশের", "রাজধানী", "ঢাকা", "শহর"]


def _chunk(chunk_id: str, text: str, lang: Lang, title: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=text,
        lang=lang,
        strategy=ChunkStrategyName.PASSAGE_NATIVE,
        title=title,
    )


#: Five Hindi passages sharing common words ("भारत", "की", "है") so IDF has something to
#: discount, plus one carrying a rare product code that only exact matching can find.
HINDI_CHUNKS = [
    _chunk("hi-1", "भारत की जनसंख्या लगभग 121 करोड़ है।", Lang.HIN),
    _chunk("hi-2", "दिल्ली भारत की राजधानी है।", Lang.HIN),
    _chunk("hi-3", "मुंबई महाराष्ट्र की राजधानी और वित्तीय केंद्र है।", Lang.HIN),
    _chunk("hi-4", "गंगा नदी भारत की सबसे लंबी नदी है।", Lang.HIN),
    _chunk("hi-5", "क्यूआर२२ एक दुर्लभ उत्पाद कोड है।", Lang.HIN),
]

TAMIL_CHUNKS = [
    _chunk("ta-1", "தமிழ்நாட்டின் தலைநகரம் சென்னை ஆகும்.", Lang.TAM),
    _chunk("ta-2", "காவிரி ஆறு தமிழ்நாட்டில் பாய்கிறது.", Lang.TAM),
    _chunk("ta-3", "மதுரை மீனாட்சி அம்மன் கோயில் புகழ்பெற்றது.", Lang.TAM),
]

BENGALI_CHUNKS = [
    _chunk("bn-1", "বাংলাদেশের রাজধানী ঢাকা শহর।", Lang.BEN),
    _chunk("bn-2", "পদ্মা নদী বাংলাদেশের সবচেয়ে বড় নদী।", Lang.BEN),
]


@pytest.fixture
def bm25_path(tmp_path: Path) -> Path:
    """A private index directory. Never `settings.bm25_path`."""
    return tmp_path / "bm25"


@pytest.fixture
def indexed(bm25_path: Path) -> BM25Index:
    index = BM25Index(bm25_path)
    index.index(HINDI_CHUNKS)
    index.index(TAMIL_CHUNKS)
    index.index(BENGALI_CHUNKS)
    return index


# ======================================================================================
# Tokenisation -- the part that actually breaks on Indic script
# ======================================================================================


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        (HINDI_SENTENCE, HINDI_TOKENS),
        (TAMIL_SENTENCE, TAMIL_TOKENS),
        (BENGALI_SENTENCE, BENGALI_TOKENS),
    ],
    ids=["devanagari", "tamil", "bengali"],
)
def test_indic_text_tokenises_into_words(sentence: str, expected: list[str]) -> None:
    tokens = tokenize(sentence)
    assert tokens == expected
    # The two silent failure modes, stated explicitly so a regression names itself.
    assert len(tokens) > 1, "whole line came back as one token"
    assert all(tokens), "tokeniser emitted an empty token"
    assert sentence.strip() not in tokens


@pytest.mark.parametrize(
    "sentence", [HINDI_SENTENCE, TAMIL_SENTENCE, BENGALI_SENTENCE], ids=["hi", "ta", "bn"]
)
def test_combining_marks_stay_welded_to_their_base(sentence: str) -> None:
    """A matra/virama/nukta split off its consonant is a different, meaningless term."""
    for token in tokenize(sentence):
        assert unicodedata.category(token[0]) not in {"Mn", "Mc", "Me"}, (
            f"token {token!r} starts with a bare combining mark"
        )


def test_specific_marked_words_survive_whole() -> None:
    hindi = tokenize(HINDI_SENTENCE)
    # क + ी (U+0940 vowel sign II): two codepoints, one term.
    assert "की" in hindi and len("की") == 2
    # करोड़ carries a nukta (U+093C); dropping or detaching it changes the word.
    assert "करोड़" in hindi
    assert "़" in "".join(hindi)
    # Tamil virama (U+0BCD) sits inside the word, not at a boundary.
    tamil = tokenize(TAMIL_SENTENCE)
    assert "தமிழ்நாட்டின்" in tamil
    assert "்" in tamil[0]


def test_danda_separates_sentences_instead_of_fusing_words() -> None:
    """Danda maps to a space, not to nothing: deleting it would weld two words into one."""
    assert tokenize("एक।दो") == ["एक", "दो"]
    assert tokenize("एक॥दो") == ["एक", "दो"]
    assert "।" not in tokenize(HINDI_SENTENCE)
    assert "।" not in normalize(HINDI_SENTENCE)


def test_zero_width_joiners_do_not_split_the_vocabulary() -> None:
    """ZWJ/ZWNJ are invisible and inconsistently typed; both spellings must be one term."""
    plain = "हिन्दी"
    with_zwj = "हिन्" + ZWJ + "दी"
    with_zwnj = "हिन्" + ZWNJ + "दी"
    assert with_zwj != plain and with_zwnj != plain  # the inputs really do differ
    assert tokenize(with_zwj) == tokenize(plain) == [plain]
    assert tokenize(with_zwnj) == tokenize(plain)
    assert ZWJ not in normalize(with_zwj)
    assert ZWNJ not in normalize(with_zwnj)


def test_normalisation_folds_case_and_unifies_nukta_spellings() -> None:
    assert normalize("QR22") == "qr22"
    # क़ has two legal spellings: precomposed U+0958, and U+0915 + nukta U+093C. Typists
    # produce both, so without NFC the same word occupies two vocabulary slots.
    precomposed = "क़लम"
    decomposed = "क़लम"
    assert precomposed != decomposed
    assert normalize(precomposed) == normalize(decomposed)
    assert tokenize(precomposed) == tokenize(decomposed) == [normalize(decomposed)]


def test_punctuation_is_a_boundary_and_never_part_of_a_term() -> None:
    """Indic corpora carry ASCII punctuation too. A whitespace-splitting tokeniser leaves it
    glued on ("भारत," and "(121)"), which is a different vocabulary entry from the word."""
    assert tokenize("भारत, जनसंख्या (121 करोड़) है।") == [
        "भारत",
        "जनसंख्या",
        "121",
        "करोड़",
        "है",
    ]
    assert tokenize('சென்னை, தலைநகரம் "தமிழ்நாடு"!') == ["சென்னை", "தலைநகரம்", "தமிழ்நாடு"]


def test_digits_and_codes_are_kept_as_terms() -> None:
    """Numerals and product codes are exactly what the dense arm smears away."""
    assert tokenize("500mg") == ["500mg"]
    assert tokenize("sr-22") == ["sr", "22"]  # the hyphen is a separator, both halves survive
    assert "121" in tokenize(HINDI_SENTENCE)
    assert tokenize("वर्ष 2011 में") == ["वर्ष", "2011", "में"]


def test_english_is_stopworded_and_stemmed() -> None:
    tokens = tokenize("The running dogs chased 3 cats", Lang.ENG)
    assert "the" not in tokens
    assert "3" in tokens
    if bm25_mod._pystemmer is None:  # optional wheel; recall degrades, behaviour does not break
        pytest.skip("PyStemmer unavailable, so the English pivot is indexed unstemmed")
    assert tokens == ["run", "dog", "chase", "3", "cat"]


def test_indic_text_is_not_stemmed_or_stopworded() -> None:
    """No Snowball stemmer covers these scripts, so nothing may be stripped from them."""
    assert tokenize(HINDI_SENTENCE, Lang.HIN) == tokenize(HINDI_SENTENCE)
    assert tokenize(TAMIL_SENTENCE, Lang.TAM) == TAMIL_TOKENS


def test_empty_text_tokenises_to_nothing() -> None:
    assert tokenize("") == []
    assert tokenize("   \n\t ") == []
    assert tokenize("।।।") == []  # punctuation only


# ======================================================================================
# Ranking
# ======================================================================================


def test_exact_rare_term_ranks_its_document_first(indexed: BM25Index) -> None:
    hits = indexed.search("क्यूआर२२", top_k=5)
    assert hits, "a term present verbatim in the corpus returned nothing"
    assert hits[0].chunk.chunk_id == "hi-5"


def test_documents_no_query_term_touched_are_not_returned(indexed: BM25Index) -> None:
    """A zero score must not become a rank -- RRF would pay it credit it did not earn."""
    hits = indexed.search("क्यूआर२२", top_k=5)
    assert len(hits) == 1  # only hi-5 contains the term, the other four score exactly 0
    assert all(hit.score > 0.0 for hit in hits)


def test_hits_carry_one_based_lexical_rank_and_score(indexed: BM25Index) -> None:
    """RRF reads `lexical_rank` directly; a 0-based rank there is silently wrong."""
    hits = indexed.search("भारत की राजधानी", top_k=4, lang=Lang.HIN)
    assert len(hits) >= 2
    assert [hit.lexical_rank for hit in hits] == list(range(1, len(hits) + 1))
    assert all(hit.lexical_score == hit.score for hit in hits)
    assert [hit.score for hit in hits] == sorted((h.score for h in hits), reverse=True)


def test_title_is_indexed_alongside_the_body(bm25_path: Path) -> None:
    index = BM25Index(bm25_path)
    index.index(
        [
            _chunk("t-1", "यह एक सामान्य विवरण है।", Lang.HIN, title="कोणार्क सूर्य मंदिर"),
            _chunk("t-2", "यह भी एक सामान्य विवरण है।", Lang.HIN),
        ]
    )
    hits = index.search("कोणार्क", top_k=5)
    assert [hit.chunk.chunk_id for hit in hits] == ["t-1"]


def test_normalisation_survives_into_the_index_not_just_the_tokeniser(bm25_path: Path) -> None:
    """The tokeniser tests above prove the rules; this proves they still hold once a document
    and a query have travelled through the vocabulary on opposite sides of the index.

    Both halves are retrieval failures no exception would ever surface. A ZWNJ typed into the
    passage but not the query puts the two spellings in different vocabulary slots, so the
    document scores zero for its own word. A deleted rather than spaced danda welds the last
    word of one sentence to the first of the next, so neither word is findable on its own --
    and MS MARCO passages run sentences together exactly like this.
    """
    index = BM25Index(bm25_path)
    index.index(
        [
            # "हिन्दी" spelled with an invisible ZWNJ; the query below spells it plainly.
            _chunk("n-1", "हिन्" + ZWNJ + "दी एक भाषा है।", Lang.HIN),
            # No space around the danda, so the fused form would be "व्याकरणसाहित्य".
            _chunk("n-2", "व्याकरण।साहित्य एक विषय है।", Lang.HIN),
            _chunk("n-3", "यह एक असंबंधित वाक्य है।", Lang.HIN),
        ]
    )

    assert [h.chunk.chunk_id for h in index.search("हिन्दी", top_k=3)] == ["n-1"]
    # Both sides of the danda are independently findable, which the fused token would not be.
    assert [h.chunk.chunk_id for h in index.search("व्याकरण", top_k=3)] == ["n-2"]
    assert [h.chunk.chunk_id for h in index.search("साहित्य", top_k=3)] == ["n-2"]
    assert index.search("व्याकरणसाहित्य", top_k=3) == []


def test_query_script_selects_its_own_language_index(indexed: BM25Index) -> None:
    """Scripts do not overlap, so a Tamil query must never be answered from Hindi."""
    for query, lang in (("தலைநகரம்", Lang.TAM), ("রাজধানী", Lang.BEN), ("राजधानी", Lang.HIN)):
        hits = indexed.search(query, top_k=5)
        assert hits, f"{query!r} matched nothing"
        assert {hit.chunk.lang for hit in hits} == {lang}


def test_explicit_lang_overrides_script_detection(indexed: BM25Index) -> None:
    hits = indexed.search("राजधानी", top_k=5, lang=Lang.TAM)
    assert hits == []  # no Devanagari term exists in the Tamil matrix


def test_per_language_indexes_keep_idf_separate(tmp_path: Path) -> None:
    """IDF and avgdl are corpus-level statistics; a shared matrix would let an unrelated
    language move Hindi scores. Same Hindi corpus, same query, two different neighbourhoods:
    the numbers must be bit-identical."""
    alone = BM25Index(tmp_path / "alone")
    alone.index(HINDI_CHUNKS)
    hindi_only = [(h.chunk.chunk_id, h.score) for h in alone.search("भारत की राजधानी", top_k=5)]

    crowded = BM25Index(tmp_path / "crowded")
    crowded.index(HINDI_CHUNKS)
    # 60 more documents in two other scripts -- 13x the Hindi corpus.
    crowded.index(
        [
            _chunk(f"pad-{i}-{c.chunk_id}", c.text, c.lang)
            for i in range(20)
            for c in TAMIL_CHUNKS
        ]
    )
    with_neighbours = [
        (h.chunk.chunk_id, h.score) for h in crowded.search("भारत की राजधानी", top_k=5)
    ]

    assert hindi_only, "the control query matched nothing"
    assert with_neighbours == hindi_only
    assert set(crowded.info()["langs"]) == {"hi", "ta"}


def test_nonpositive_top_k_and_blank_queries_return_empty(indexed: BM25Index) -> None:
    assert indexed.search("भारत", top_k=0) == []
    assert indexed.search("भारत", top_k=-3) == []
    assert indexed.search("", top_k=5) == []
    assert indexed.search("   ", top_k=5) == []


def test_search_on_an_empty_index_returns_empty(bm25_path: Path) -> None:
    assert BM25Index(bm25_path).search("भारत", top_k=5) == []


# ======================================================================================
# Build laziness, replacement, persistence
# ======================================================================================


def test_matrix_is_built_lazily_on_first_search(bm25_path: Path) -> None:
    """`index()` is additive and cheap; building per batch would be O(n^2) over an ingest."""
    index = BM25Index(bm25_path)
    index.index(HINDI_CHUNKS)
    assert index.info()["pending_langs"] == ["hi"]
    assert index.info()["langs"] == {}  # nothing compiled yet

    hits = index.search("क्यूआर२२", top_k=3)
    assert [hit.chunk.chunk_id for hit in hits] == ["hi-5"]
    assert index.info()["pending_langs"] == []


def test_reindexing_a_chunk_id_replaces_rather_than_duplicates(bm25_path: Path) -> None:
    index = BM25Index(bm25_path)
    index.index(HINDI_CHUNKS)
    assert index.count() == 5

    index.index([_chunk("hi-5", "क्यूआर२२ अब एक नया विवरण है।", Lang.HIN)])
    assert index.count() == 5

    hits = index.search("क्यूआर२२", top_k=5)
    assert [hit.chunk.chunk_id for hit in hits] == ["hi-5"]
    assert hits[0].chunk.text == "क्यूआर२२ अब एक नया विवरण है।"


def test_persistence_round_trips(bm25_path: Path) -> None:
    original = BM25Index(bm25_path)
    original.index(HINDI_CHUNKS)
    original.index(TAMIL_CHUNKS)
    before = [(h.chunk.chunk_id, h.score) for h in original.search("भारत की राजधानी", top_k=5)]
    original.save()

    reloaded = BM25Index(bm25_path)
    assert reloaded.count() == original.count()
    assert reloaded.info()["langs"] == original.info()["langs"]
    after = [(h.chunk.chunk_id, h.score) for h in reloaded.search("भारत की राजधानी", top_k=5)]
    assert after == before

    # The sidecar must round-trip the whole pydantic Chunk, not just {id, text}.
    hit = reloaded.search("क्यूआर२२", top_k=1)[0].chunk
    assert hit == next(c for c in HINDI_CHUNKS if c.chunk_id == "hi-5")


def test_indexing_after_a_reload_keeps_the_persisted_documents(bm25_path: Path) -> None:
    """Adding one chunk to a loaded index must not throw away what was on disk.

    The matrix is compiled from `self._chunks[lang]`, and `index()` only marks the language
    dirty -- so the reload path has to have adopted the sidecar corpus into that same dict.
    If it had merely loaded the matrix, the next `index()` call would mark the language dirty
    and the rebuild would compile a corpus containing *only* the newly added chunk, silently
    deleting the entire persisted index on the first incremental ingest.
    """
    built = BM25Index(bm25_path)
    built.index(HINDI_CHUNKS)
    built.save()

    reloaded = BM25Index(bm25_path)
    reloaded.index([_chunk("hi-6", "कोणार्क मंदिर ओडिशा में है।", Lang.HIN)])

    assert reloaded.count() == len(HINDI_CHUNKS) + 1
    assert [h.chunk.chunk_id for h in reloaded.search("कोणार्क", top_k=5)] == ["hi-6"]
    # The pre-existing documents are still matchable, and still hydrate their own text.
    old = reloaded.search("क्यूआर२२", top_k=5)
    assert [h.chunk.chunk_id for h in old] == ["hi-5"]
    assert old[0].chunk == next(c for c in HINDI_CHUNKS if c.chunk_id == "hi-5")


def test_stale_tokenizer_version_is_refused_rather_than_silently_mismatched(
    bm25_path: Path,
) -> None:
    """A vocabulary built by an older tokeniser loads fine and then stops matching -- the
    worst failure mode available. The manifest exists to make that detectable."""
    built = BM25Index(bm25_path)
    built.index(HINDI_CHUNKS)
    built.save()

    manifest = bm25_path / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["tokenizer"] = TOKENIZER_VERSION + 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    reloaded = BM25Index(bm25_path)
    assert reloaded.count() == 0
    assert reloaded.search("क्यूआर२२", top_k=5) == []


def test_corrupt_sidecar_invalidates_only_its_own_language(bm25_path: Path) -> None:
    """Document ids are positional, so a skipped line would shift every later document.
    Dropping the whole language is the only safe recovery -- and it must not take out the
    languages whose sidecars are intact."""
    built = BM25Index(bm25_path)
    built.index(HINDI_CHUNKS)
    built.index(TAMIL_CHUNKS)
    built.save()
    (bm25_path / "hi" / "chunks.jsonl").write_text('{"not":"a chunk"}\n', encoding="utf-8")

    reloaded = BM25Index(bm25_path)
    assert reloaded.search("क्यूआर२२", top_k=5) == []
    assert [h.chunk.chunk_id for h in reloaded.search("தலைநகரம்", top_k=5)] == ["ta-1"]
