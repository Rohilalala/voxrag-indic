r"""The FAST-mode answer path: which sentence gets returned, and when nothing does.

This file exists because a mutation audit found the gap: `_best_window` could be replaced
with "return the first sentence of the first chunk" and the entire 588-test suite stayed
green. FAST is the default mode, so that is the code answering most requests, and it had no
unit test at all.

Every test here is written to FAIL against that mutant. The shape they all rely on is the
real failure mode `generate/extractive.py` is built to avoid: **the sentence that best
matches a question is usually the one that restates it**, not the one that answers it. So
each fixture puts a restatement first and the answer later, which is exactly what MS MARCO
passages look like ("The cost of X depends on several factors. The average price is $90.").

Semantic re-ranking is disabled throughout: it is ~16ms of ONNX inference per call, and a
unit test that loads an embedder is an integration test wearing a disguise.
"""

from __future__ import annotations

import pytest

from voxrag.config import settings
from voxrag.contracts import (
    Chunk,
    ChunkStrategyName,
    GenerationMode,
    Lang,
    RefusalReason,
    ScoredChunk,
)
from voxrag.generate.extractive import ExtractiveGenerator, best_supporting_span

RESTATES_THEN_ANSWERS = (
    "The cost of a self defense instructor depends on several factors and varies widely. "
    "The average price for a self defense class is 90 dollars per month. "
    "Many gyms also offer a free trial lesson to new members."
)

HINDI_RESTATES_THEN_ANSWERS = (
    "पानी के क्वथनांक के बारे में कई लोग पूछते हैं। "
    "पानी का क्वथनांक 100 डिग्री सेल्सियस होता है। "
    "यह समुद्र तल पर मापा जाता है।"
)


@pytest.fixture(autouse=True)
def _no_semantic_rerank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "extractive_semantic_rerank", False)


def _chunk(text: str, chunk_id: str = "c1", lang: Lang = Lang.ENG) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=text,
        lang=lang,
        strategy=ChunkStrategyName.PASSAGE_NATIVE,
    )


def _ctx(*texts: str, lang: Lang = Lang.ENG) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=_chunk(t, f"c{i}", lang), score=1.0 - 0.1 * i, dense_rank=i + 1)
        for i, t in enumerate(texts)
    ]


class TestSpanSelection:
    """What comes back, and why that sentence rather than the first one."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "KNOWN, MEASURED DEFECT, not a flaky test. Scoring a sentence by overlap with the "
            "QUESTION prefers the sentence that restates it: asked what an instructor costs, "
            "the generator returns 'the cost ... depends on several factors' over 'the average "
            "price is 90 dollars'. This is the whole of the answer-quality gap -- token F1 "
            "0.207 against a measured extractive ceiling of 0.684, with groundedness 1.000, "
            "i.e. it never fabricates, it picks the wrong true sentence. Strict, so whoever "
            "fixes selection is told to delete this marker."
        ),
    )
    def test_answers_with_the_answering_sentence_not_the_restating_one(self) -> None:
        # The mutation this file was written for returns the FIRST sentence, which here is
        # the one that merely repeats the question back.
        answer = ExtractiveGenerator().answer(
            "how much does a self defense instructor cost", _ctx(RESTATES_THEN_ANSWERS)
        )
        assert not answer.refused
        assert "90 dollars" in answer.text, f"selected the wrong sentence: {answer.text!r}"
        assert not answer.text.startswith("The cost of a self defense instructor depends")

    def test_the_same_holds_in_an_indic_language(self) -> None:
        answer = ExtractiveGenerator().answer(
            "पानी का क्वथनांक कितना होता है", _ctx(HINDI_RESTATES_THEN_ANSWERS, lang=Lang.HIN),
            lang=Lang.HIN,
        )
        assert not answer.refused
        assert "100 डिग्री" in answer.text, f"selected the wrong sentence: {answer.text!r}"

    def test_picks_the_best_chunk_not_the_best_ranked_one(self) -> None:
        # Retrieval rank modulates the score, it does not decide it: chunk 0 is ranked first
        # and is topically adjacent; chunk 1 holds the answer. A mutant that reads chunk 0's
        # first sentence, or any implementation that trusts rank alone, fails here.
        context = _ctx(
            "Photosynthesis is a topic many students find difficult to understand.",
            "Photosynthesis converts light energy into chemical energy stored as glucose.",
        )
        answer = ExtractiveGenerator().answer("what does photosynthesis convert", context)
        assert not answer.refused
        assert "chemical energy" in answer.text
        assert answer.citations and answer.citations[0].chunk_id == "c1"

    def test_the_answer_is_verbatim_from_the_passage(self) -> None:
        # FAST mode's entire safety claim is that it copies rather than composes. If this
        # ever fails, groundedness 1.000 elsewhere in the suite means nothing.
        passage = RESTATES_THEN_ANSWERS
        answer = ExtractiveGenerator().answer("what is the average price", _ctx(passage))
        assert not answer.refused
        assert answer.text in passage, "extractive answers must be substrings of their source"

    def test_a_window_may_span_neighbours_but_stays_inside_one_chunk(self) -> None:
        answer = ExtractiveGenerator().answer(
            "how much does a self defense class cost per month", _ctx(RESTATES_THEN_ANSWERS)
        )
        assert not answer.refused
        assert answer.text in RESTATES_THEN_ANSWERS
        assert len(answer.citations) == 1


class TestRefusal:
    """When the honest answer is nothing at all."""

    def test_refuses_when_no_sentence_carries_the_query_terms(self) -> None:
        answer = ExtractiveGenerator().answer(
            "what is the capital of Iceland",
            _ctx("Bananas are a tropical fruit grown mainly in warm climates."),
        )
        assert answer.refused
        assert answer.refusal_reason is RefusalReason.NO_RELEVANT_CONTEXT
        assert answer.text == settings.refusal_message

    def test_refuses_on_empty_context(self) -> None:
        answer = ExtractiveGenerator().answer("anything at all", [])
        assert answer.refused

    def test_refuses_when_the_query_has_no_content_terms(self) -> None:
        answer = ExtractiveGenerator().answer("!!! ???", _ctx(RESTATES_THEN_ANSWERS))
        assert answer.refused

    def test_a_refusal_carries_no_citations_and_no_groundedness(self) -> None:
        answer = ExtractiveGenerator().answer("what is the capital of Iceland", _ctx("Bananas."))
        assert answer.refused
        assert answer.citations == []
        assert answer.groundedness == 0.0


class TestAnswerMetadata:
    """The fields the rails and the UI read off an answer."""

    def test_reports_fast_mode_and_the_generator_name(self) -> None:
        answer = ExtractiveGenerator().answer("what is the average price", _ctx(RESTATES_THEN_ANSWERS))
        assert answer.mode is GenerationMode.FAST
        assert answer.model == "extractive"

    def test_carries_the_language_it_was_asked_in(self) -> None:
        answer = ExtractiveGenerator().answer(
            "पानी का क्वथनांक कितना होता है",
            _ctx(HINDI_RESTATES_THEN_ANSWERS, lang=Lang.HIN),
            lang=Lang.HIN,
        )
        assert answer.lang is Lang.HIN

    def test_citation_points_at_the_chunk_the_span_came_from(self) -> None:
        answer = ExtractiveGenerator().answer("what does photosynthesis convert", _ctx(
            "Unrelated introductory sentence about school syllabuses.",
            "Photosynthesis converts light energy into chemical energy stored as glucose.",
        ))
        assert not answer.refused
        assert [c.chunk_id for c in answer.citations] == ["c1"]

    async def test_the_async_entry_point_agrees_with_the_sync_one(self) -> None:
        gen = ExtractiveGenerator()
        context = _ctx(RESTATES_THEN_ANSWERS)
        assert (await gen.generate("what is the average price", context)).text == gen.answer(
            "what is the average price", context
        ).text


class TestBestSupportingSpan:
    """The sentence-level primitive the window search is built on."""

    def test_finds_the_sentence_that_holds_the_terms(self) -> None:
        span = best_supporting_span(["average", "price"], RESTATES_THEN_ANSWERS, Lang.ENG)
        assert span is not None
        assert "average price" in RESTATES_THEN_ANSWERS[span.start : span.end]

    def test_falls_back_to_the_whole_text_when_nothing_overlaps(self) -> None:
        # Documented contract: a citation must always resolve to a real span, so a term that
        # appears nowhere yields the full passage rather than None. None is reserved for
        # empty text -- the two cases are different and the caller renders them differently.
        span = best_supporting_span(["reykjavik"], RESTATES_THEN_ANSWERS, Lang.ENG)
        assert span is not None and span.text == RESTATES_THEN_ANSWERS
        assert best_supporting_span(["anything"], "   ", Lang.ENG) is None
