"""Extractive answer generation: the FAST path, and the one inside the 200ms budget.

The answer is a contiguous run of 1-3 sentences **copied verbatim** out of a retrieved
chunk. That single design choice buys three things at once:

* **Latency.** No model is loaded, no network is touched, no text is embedded. The whole
  stage is BM25 arithmetic over the ~40 sentences of the top `context_top_k` chunks, which
  measures in hundreds of microseconds -- leaving the 200ms budget to retrieval.
* **Groundedness by construction.** The answer text is a substring of the context, so
  hallucination is not merely unlikely, it is unrepresentable. `groundedness` is 1.0 as a
  fact about the algorithm, not as an estimate.
* **Exact citations.** Because we slice rather than paraphrase, every character of the
  answer has a known origin: the emitted `Citation` offsets satisfy
  ``answer[c.answer_start:c.answer_end] == chunk_text[c.source_start:c.source_end]``.

Scoring is BM25 over *sentences*, with the sentence collection of this request as the term
statistics. A global IDF table would be better in principle and unusable in practice here:
the corpus is 14 languages and the index is rebuilt per benchmark run, whereas local IDF
needs nothing but the text already in hand and still does the job that matters -- pushing
down the function words that appear in every sentence of the context and pushing up the
terms that actually discriminate between them.

Tokenisation is the encoder's own SentencePiece vocabulary, imported from
`voxrag.chunking.base`. Not whitespace: Indic morphology glues postpositions and case onto
the stem, so "भारत" in the query and "भारतमें" in the passage share no whitespace word but do
share a subword. It is also free -- the chunk texts were tokenised at index time and
`token_index` is LRU-cached across the process.
"""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import regex

from voxrag.chunking.base import (
    Span,
    TokenIndex,
    make_span,
    merge_spans,
    split_sentences,
    token_index,
)
from voxrag.config import settings
from voxrag.contracts import (
    Answer,
    Citation,
    GenerationMode,
    Lang,
    RefusalReason,
    ScoredChunk,
)

# --------------------------------------------------------------------------------------
# Module-level constants (not in config.py -- they describe this file's own mechanics)
# --------------------------------------------------------------------------------------

#: The chunk's retrieval score *modulates* the sentence's lexical score instead of being
#: added to it. Additive was tried and is wrong: the top chunk always normalises to 1.0 on
#: the retrieval arm, so its sentences inherited a constant floor and an off-topic query got
#: a confident answer out of the best-ranked chunk. Multiplying means no lexical support ->
#: no score, while a well-ranked chunk is still worth 1/0.6 of a bottom-ranked one when the
#: text is equally good, which is exactly what the retrieval score should be worth here.
#:
#: 0.30 was proposed by the coordinate-descent sweep (`data/tune_extractive.json`: 0.2245 ->
#: 0.2273 on the 350 queries it fitted) and REJECTED on a sample it was not fitted to --
#: `--per-lang 40 --sample-seed 1`, 560 queries scored over identical cached evidence:
#:   F1 0.2182 -> 0.2161   (-0.0021, paired 95% CI [-0.0072, +0.0030], p=0.44)
#: Re-run on a second occasion: -0.0010, CI [-0.0060, +0.0040], p=0.70. Same sign, same
#: conclusion, and the spread between the two runs is itself larger than the effect.
#: It rewrites 50 of the 560 answers and loses on net. Six of the eight constants that sweep
#: proposed are already the shipped values; this and _BM25_B were its only real changes, and
#: the two of them together measured -0.0010 held out. Do not re-ship them off that artifact.
_RETRIEVAL_PRIOR_FLOOR = 0.60

#: BM25 saturation and length normalisation. `b` is below the usual 0.75 because the
#: "documents" are sentences: their length variance is mostly grammar, not verbosity, and
#: a hard length penalty would systematically prefer sentence fragments over full answers.
_BM25_K1 = 1.2
#: The sweep above also wanted `b` at 0.30 (0.2273 -> 0.2295 on its own sample). Held out it
#: is the only one of the eight that came back positive -- F1 0.2182 -> 0.2213, +0.0032 -- but
#: 95% CI [-0.0023, +0.0092], p=0.28 over the 33 answers of 560 it changes, so it is not
#: distinguishable from zero and is not shipped on a p of 0.28. A second run agreed: +0.0039,
#: CI [-0.0014, +0.0097], p=0.18 -- still spanning zero.
#:
#: Reproduce with `scripts/confirm_extractive_paired.py`, which is committed BECAUSE this
#: comment cites it. `tune_extractive.py --confirm` prints f1/delta only and computes no CI or
#: p-value, so an earlier version of this comment named a command that could not produce the
#: numbers beside it -- a claim whose evidence does not exist in the repo is not a measurement.
_BM25_B = 0.6

#: Refusal floor on lexical coverage in [0,1] -- the fraction of the query's IDF mass found
#: in the answer's anchor sentence. Below this the system says so instead of guessing.
_MIN_LEX_COVERAGE = 0.20

#: Expansion to a neighbouring sentence is worth `bonus * (its coverage - threshold)`, where
#: the threshold is *relative to the anchor*: a companion must carry at least this fraction
#: of the anchor's coverage to earn its place, so a strong anchor answers alone and a
#: two-sentence answer only happens when the second sentence really is part of the answer.
#: A plain absolute floor cannot do this -- it admits topical filler next to a strong anchor
#: and rejects a genuine continuation next to a weak one.
_NEIGHBOUR_BONUS = 0.35
_NEIGHBOUR_RELATIVE = 0.55
_NEIGHBOUR_FLOOR = 0.10

#: Absolute-cosine band that maps an e5 similarity onto a [0,1] confidence for the refusal
#: gate. Calibrated on this corpus rather than guessed -- measured query/sentence cosines:
#:   relevant   0.8151 .. 0.8790   (Hindi definition questions vs their answer sentences)
#:   irrelevant 0.7253 .. 0.7592   (bank password, cooking pasta, weather vs corporate text)
#: The floor sits inside that 0.056 gap, so irrelevant pairs clamp to 0.0 and are refused,
#: while the weakest genuine match still clears _MIN_LEX_COVERAGE.
#:
#: These numbers are specific to multilingual-e5-small, whose cosines are compressed into a
#: narrow band. Changing `embed_model_repo` REQUIRES re-running that calibration; a model
#: with a wider range would make this floor reject everything.
_SEM_FLOOR = 0.78
_SEM_CEILING = 0.88

_MAX_ANSWER_SENTENCES = 3
#: Ceiling on a *multi*-sentence answer. A single sentence is always allowed through even
#: if it is longer: truncating it would break the verbatim-copy invariant the citations rest on.
_MAX_ANSWER_TOKENS = 96

#: Tokens that are pure punctuation, symbols, separators, control chars or bare combining
#: marks carry no retrieval signal and would otherwise dominate tf in short sentences.
_NOISE_TOKEN_RE = regex.compile(r"^[\p{P}\p{S}\p{Z}\p{C}\p{M}]*$")


# --------------------------------------------------------------------------------------
# Term extraction (shared with the abstractive generator)
# --------------------------------------------------------------------------------------


def _terms(text: str, idx: TokenIndex, lo: int, hi: int) -> list[str]:
    """Normalised content tokens for the half-open token range [lo, hi) of `text`."""
    out: list[str] = []
    for start, end in idx.spans[lo:hi]:
        # Slicing the ORIGINAL text (not the tokeniser's own token string) keeps this free
        # of vocabulary artefacts like the SentencePiece word-boundary marker.
        term = text[start:end].strip().casefold()
        if term and not _NOISE_TOKEN_RE.match(term):
            out.append(term)
    return out


def content_terms(text: str) -> list[str]:
    """Normalised, noise-free encoder tokens of `text`, in order, with repeats."""
    idx = token_index(text)
    return _terms(text, idx, 0, idx.n)


def _idf(n_docs: int, df: int) -> float:
    """BM25 IDF over the sentence collection.

    `df` is clamped to at least 1 by the caller, which makes a query term that appears in no
    sentence exactly as informative as the rarest term that does. Left unclamped, a single
    out-of-context query word would take such a large share of the IDF mass that every
    sentence's coverage collapses and the generator refuses correct answers.
    """
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def best_supporting_span(terms: Sequence[str], text: str, lang: Lang | None = None) -> Span | None:
    """The sentence of `text` with the highest unique-term overlap with `terms`.

    Used by the abstractive generator to point a citation at the specific line of a chunk
    that supports a claim, instead of at the whole chunk. Falls back to the full text when
    nothing overlaps, so a citation always resolves to a real span.
    """
    if not text.strip():
        return None
    wanted = set(terms)
    whole = make_span(text, 0, len(text))
    if not wanted:
        return whole
    idx = token_index(text)
    best: Span | None = None
    best_hits = 0
    for sp in split_sentences(text, lang):
        lo, hi = idx.range(sp.start, sp.end)
        hits = len(wanted.intersection(_terms(text, idx, lo, hi)))
        if hits > best_hits:
            best, best_hits = sp, hits
    return best or whole


# --------------------------------------------------------------------------------------
# Candidate sentences
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Sentence:
    """One scored candidate sentence, with its offsets into its chunk's own text."""

    span: Span
    terms: tuple[str, ...]
    n_tokens: int
    lex: float = 0.0
    #: Cosine to the query min-maxed across the shortlist. For RANKING only -- min-max always
    #: awards 1.0 to the best candidate, so it says nothing about absolute relevance.
    sem: float = 0.0
    #: Absolute cosine mapped to a confidence in [0,1]. For the REFUSAL GATE only.
    sem_conf: float = 0.0
    combined: float = 0.0

    @property
    def support(self) -> float:
        """Absolute evidence that this sentence answers the query.

        The coverage floor gates on this rather than on `lex` alone: a sentence that is
        clearly on-topic semantically but shares no surface tokens with the query is a
        legitimate answer on a translated corpus, not a miss.

        Deliberately uses `sem_conf` and NOT `sem`. Gating on the min-maxed score would make
        the system incapable of refusing anything -- the best of any shortlist scores 1.0,
        even when every candidate is garbage, so "what is my bank password" would answer.
        """
        return max(self.lex, self.sem_conf)


def _normalised_chunk_scores(context: Sequence[ScoredChunk]) -> list[float]:
    """Min-max the retrieval scores into [0,1].

    Fused scores are only comparable inside one result set (see `contracts.ScoredChunk`) and
    their absolute scale differs per retriever -- RRF sums, cosines and cross-encoder logits
    all land here. Min-max is the only normalisation that is correct for all three.
    """
    scores = [sc.score for sc in context]
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


class ExtractiveGenerator:
    """`contracts.Generator` in FAST mode: verbatim sentence extraction."""

    name = "extractive"
    mode = GenerationMode.FAST

    async def generate(
        self, query: str, context: list[ScoredChunk], *, lang: Lang | None = None
    ) -> Answer:
        # Pure-Python scoring is a few hundred microseconds over at most `context_top_k`
        # chunks, and a to_thread hand-off would cost more than the work plus add scheduler
        # jitter to the P100 we report -- so that path stays inline. Semantic re-ranking is
        # different: it is ~16ms of blocking ONNX inference, which would stall the event loop
        # and serialise concurrent requests, so only that path is offloaded.
        if settings.extractive_semantic_rerank:
            return await asyncio.to_thread(self.answer, query, context, lang=lang)
        return self.answer(query, context, lang=lang)

    async def health(self) -> bool:
        # No model, no socket, no file handle. There is nothing that can be unhealthy.
        return True

    def answer(
        self, query: str, context: Sequence[ScoredChunk], *, lang: Lang | None = None
    ) -> Answer:
        """Synchronous entry point -- used by the benchmark, which times without an event loop."""
        chunks = list(context)[: settings.context_top_k]
        if not chunks:
            return self._refuse(lang)

        q_terms = content_terms(query)
        if not q_terms:
            # Punctuation or an empty transcript: nothing to match on, so nothing is relevant.
            return self._refuse(lang)

        sentences = self._candidates(chunks)
        n_sent = sum(len(row) for row in sentences)
        if n_sent == 0:
            return self._refuse(lang)

        self._score(sentences, q_terms, n_sent, _normalised_chunk_scores(chunks))
        if settings.extractive_semantic_rerank:
            self._semantic_rescore(query, sentences)

        best = self._best_window(sentences)
        if best is None:
            return self._refuse(lang)
        pos, lo, hi, anchor_support = best
        if anchor_support < _MIN_LEX_COVERAGE:
            return self._refuse(lang)
        return self._build_answer(chunks[pos], sentences[pos][lo:hi], lang)

    # -- internals ----------------------------------------------------------------------

    def _candidates(self, chunks: Sequence[ScoredChunk]) -> list[list[_Sentence]]:
        """Segment each chunk into sentences and tag each with its content tokens.

        The chunk is tokenised **once** and every sentence's token range is recovered from
        that single index by binary search (`TokenIndex.range`). Calling `count_tokens` per
        sentence instead would re-run the tokeniser dozens of times per request.
        """
        rows: list[list[_Sentence]] = []
        for sc in chunks:
            text = sc.chunk.generation_text
            idx = token_index(text)
            row: list[_Sentence] = []
            for sp in split_sentences(text, sc.chunk.lang):
                lo, hi = idx.range(sp.start, sp.end)
                terms = _terms(text, idx, lo, hi)
                if terms:
                    row.append(_Sentence(span=sp, terms=tuple(terms), n_tokens=max(1, hi - lo)))
            rows.append(row)
        return rows

    def _score(
        self,
        sentences: list[list[_Sentence]],
        q_terms: Sequence[str],
        n_sent: int,
        chunk_scores: Sequence[float],
    ) -> None:
        """BM25 the query against every candidate sentence, then fuse with retrieval score."""
        df: Counter[str] = Counter()
        total_len = 0
        for row in sentences:
            for s in row:
                df.update(set(s.terms))
                total_len += s.n_tokens
        avg_len = max(1.0, total_len / n_sent)

        # dict.fromkeys de-duplicates while preserving order, so the IDF mass is per distinct
        # query term -- a term repeated in the query must not count twice in the denominator.
        idf = {t: _idf(n_sent, max(1, min(df.get(t, 0), n_sent))) for t in dict.fromkeys(q_terms)}
        idf_total = sum(idf.values()) or 1.0

        for pos, row in enumerate(sentences):
            prior = chunk_scores[pos]
            for s in row:
                tf = Counter(s.terms)
                length_norm = _BM25_K1 * (1.0 - _BM25_B + _BM25_B * s.n_tokens / avg_len)
                acc = 0.0
                for term, weight in idf.items():
                    f = tf.get(term, 0)
                    if f:
                        acc += weight * (f * (_BM25_K1 + 1.0)) / (f + length_norm)
                # A single occurrence at average length contributes ~1.0 x its IDF, so the
                # ratio to the total IDF mass reads as "fraction of the query covered".
                # Repetition in a short sentence can push it just over 1; clamp so the
                # refusal floor and the neighbour bonus stay on a comparable scale.
                s.lex = min(1.0, acc / idf_total)
                s.combined = s.lex * (
                    _RETRIEVAL_PRIOR_FLOOR + (1.0 - _RETRIEVAL_PRIOR_FLOOR) * prior
                )

    def _semantic_rescore(self, query: str, sentences: list[list[_Sentence]]) -> None:
        """Re-rank the strongest candidates by embedding cosine and blend into `combined`.

        Only the shortlist is embedded. Encoding costs ~2ms per sentence, so scoring every
        sentence of every retrieved chunk would cost more than the entire rest of the
        pipeline; the BM25-and-prior pass above is a good enough filter to decide which few
        are worth paying for. The query embedding is almost always an LRU hit, because the
        pipeline already embedded this exact string to run retrieval.
        """
        # Imported here, not at module scope: this module is otherwise dependency-free and
        # importable without onnxruntime, which the benchmark and the tests rely on.
        from voxrag.embed import get_embedder

        flat = [s for row in sentences for s in row]
        if len(flat) < 2:
            return
        shortlist = sorted(flat, key=lambda s: s.combined, reverse=True)[
            : max(2, settings.extractive_semantic_candidates)
        ]

        try:
            embedder = get_embedder()
            qv = embedder.encode([query], is_query=True)[0]
            mat = embedder.encode([s.span.text for s in shortlist], is_query=False)
        except Exception:
            # Never lose an answer because an optional re-ranker failed: the lexical ordering
            # computed above is still valid, just weaker.
            return

        # Vectors are L2-normalised by the encoder, so the dot product is the cosine.
        sims = [float(mat[i] @ qv) for i in range(len(shortlist))]
        lo, hi = min(sims), max(sims)
        spread = hi - lo
        weight = settings.extractive_semantic_weight

        # Rescale EVERY sentence, not just the shortlist. Blending only the shortlisted ones
        # would leave the rest on the old scale, where an unshortlisted sentence could
        # outrank a semantically better one purely because its score was never shrunk.
        for sent in flat:
            sent.combined *= 1.0 - weight

        for sent, sim in zip(shortlist, sims, strict=True):
            # Min-max inside the shortlist. Absolute e5 cosines sit in a narrow band, so raw
            # values would barely separate candidates -- good for ranking, useless as a
            # confidence, which is why the gate below uses the absolute value instead.
            sent.sem = (sim - lo) / spread if spread > 1e-6 else 1.0
            sent.combined += weight * sent.sem
            sent.sem_conf = min(
                1.0, max(0.0, (sim - _SEM_FLOOR) / (_SEM_CEILING - _SEM_FLOOR))
            )

    def _best_window(self, sentences: list[list[_Sentence]]) -> tuple[int, int, int, float] | None:
        """Best contiguous 1-3 sentence run: (chunk_pos, start, end, anchor coverage).

        Every window is scored as "its strongest sentence, plus whatever its companions add".
        Anchoring on the maximum means a window is never credited twice for one good
        sentence, and the signed bonus is what decides the window's length: a companion
        below the threshold subtracts, so padding an answer with filler always loses.
        """
        best: tuple[int, int, int, float] | None = None
        best_score = -math.inf
        for pos, row in enumerate(sentences):
            n = len(row)
            for i in range(n):
                tokens = 0
                for j in range(i, min(i + _MAX_ANSWER_SENTENCES, n)):
                    tokens += row[j].n_tokens
                    # A lone sentence is always eligible: truncating it to fit would break
                    # the verbatim-copy invariant that makes the citations exact.
                    if j > i and tokens > _MAX_ANSWER_TOKENS:
                        break
                    window = row[i : j + 1]
                    anchor = max(window, key=lambda s: s.combined)
                    threshold = max(_NEIGHBOUR_FLOOR, _NEIGHBOUR_RELATIVE * anchor.support)
                    score = anchor.combined + _NEIGHBOUR_BONUS * sum(
                        s.support - threshold for s in window if s is not anchor
                    )
                    if score > best_score:
                        best, best_score = (pos, i, j + 1, anchor.support), score
        return best

    def _build_answer(
        self, sc: ScoredChunk, window: Sequence[_Sentence], lang: Lang | None
    ) -> Answer:
        """Slice the window out of the chunk verbatim and map every sentence to a citation."""
        # `generation_text` -- not `text` -- is the citation frame: it is the string the
        # generator was given (the parent passage under PARENT_CHILD, the window under
        # SENTENCE_WINDOW), so source offsets index what a reader is actually shown.
        source = sc.chunk.generation_text
        merged = merge_spans(source, [s.span for s in window])
        if merged is None:  # whitespace-only window; cannot happen for scored sentences
            return self._refuse(lang)

        citations: list[Citation] = []
        for s in window:
            a_start = s.span.start - merged.start
            a_end = s.span.end - merged.start
            if a_start < 0 or a_end > len(merged.text) or a_end <= a_start:
                continue
            citations.append(
                Citation(
                    chunk_id=sc.chunk.chunk_id,
                    doc_id=sc.chunk.doc_id,
                    answer_start=a_start,
                    answer_end=a_end,
                    source_start=s.span.start,
                    source_end=s.span.end,
                    quote=s.span.text,
                )
            )
        return Answer(
            text=merged.text,
            mode=GenerationMode.FAST,
            citations=citations,
            # Verbatim copy: every character of the answer is in the cited source. This is a
            # property of the algorithm, not a measurement, so it is not estimated.
            groundedness=1.0,
            model=self.name,
            lang=lang or sc.chunk.lang,
        )

    def _refuse(self, lang: Lang | None) -> Answer:
        return Answer(
            text=settings.refusal_message,
            mode=GenerationMode.FAST,
            groundedness=0.0,
            refused=True,
            refusal_reason=RefusalReason.NO_RELEVANT_CONTEXT,
            model=self.name,
            lang=lang,
        )


__all__ = ["ExtractiveGenerator", "best_supporting_span", "content_terms"]
