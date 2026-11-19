"""Output guardrails: what the system checks before it will let an answer out.

Three failure modes, three rails. The generator can invent facts (GroundednessRail), it can
invent the *provenance* of facts (CitationRail), and it can produce a half-formed refusal that
still leaks a guess (RefusalConsistencyRail). All three end the same way -- the answer is
replaced by `settings.refusal_message` -- because a wrong answer delivered confidently is worse
than no answer, and this pipeline is graded on knowing when not to answer.

Groundedness here is lexical overlap with the retrieved context, not an LLM-as-judge call.
That is a deliberate trade: a judge model would be more faithful and would cost 300ms+ of the
200ms budget, while overlap costs ~200us and catches the failure that actually happens -- a
model emitting an entity, a date or a figure that appears nowhere in what it was given.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import regex

from voxrag.config import settings
from voxrag.contracts import (
    Answer,
    Chunk,
    Citation,
    GuardrailVerdict,
    Lang,
    RefusalReason,
    ScoredChunk,
)
from voxrag.guardrails import lexicons
from voxrag.guardrails.grounding import same_language_hits

GROUNDEDNESS_RAIL: Final = "groundedness"
CITATION_RAIL: Final = "citations"
REFUSAL_RAIL: Final = "refusal_consistency"

_DIGIT_RE: Final = regex.compile(r"\p{N}")

#: Bracketed citation markers the generator is told to emit. Indic digits are included
#: because a model answering in Hindi may cite with "[१]".
#:
#: Must accept a GROUPED citation -- "[1, 3]" and "[2; 4]" -- not just "[1]". gemma4 emits
#: grouped markers unprompted when one sentence draws on two passages, and a single-marker
#: pattern let those digits through as "fabricated figures", which refused an otherwise
#: perfect Bengali answer while the same query in Hindi passed. That is the kind of bug that
#: looks like "the model is bad at Bengali" and is actually a regex.
_CITATION_MARKER_RE: Final = regex.compile(
    r"\[\s*\p{Nd}{1,2}(?:\s*[,;]\s*\p{Nd}{1,2})*\s*\]"
)

#: A shared prefix of this length counts a token as supported. There is no usable stemmer for
#: these 14 languages (Snowball covers none of them), and their morphology is overwhelmingly
#: suffixal -- "भारत"/"भारतीय", "मुंबई"/"मुंबईत", English "report"/"reported". Without this,
#: every inflected restatement of a fact in the context reads as an invented word and a
#: perfectly grounded abstractive answer gets refused.
_PREFIX_LEN: Final = 5
_PREFIX_MIN_TOKEN: Final = 6


def _normalize(text: str) -> str:
    """Lowercase, fold joiners/danda, and fold Indic digits to ASCII.

    Digit folding is what lets an answer's "2011" match a Devanagari passage's "२०११". Without
    it every numeral in an Indic answer looks fabricated and the numeric check below fires on
    every single request.
    """
    return text.translate(lexicons.NORMALIZE_FOLD).translate(lexicons.DIGIT_FOLD).lower()


def content_tokens(text: str) -> list[str]:
    """Content-bearing tokens, in order, stopwords and single characters removed.

    A list rather than a set: a hallucinated word repeated four times should count four times
    against the answer, and set semantics would forgive it three of those.
    """
    return [
        token
        for token in lexicons.TOKEN_RE.findall(_normalize(text))
        if token not in lexicons.ALL_STOPWORDS and (len(token) > 1 or token.isdigit())
    ]


@lru_cache(maxsize=2048)
def _support_index(text: str) -> tuple[frozenset[str], frozenset[str]]:
    """(tokens, prefixes) for one context passage.

    Cached on the passage text: a benchmark run replays the same top-k chunks across hundreds
    of queries, and this turns the second and later appearances of a chunk into a dict hit.
    Bounded so a long ingest cannot grow it without limit.
    """
    tokens = frozenset(lexicons.TOKEN_RE.findall(_normalize(text)))
    prefixes = frozenset(token[:_PREFIX_LEN] for token in tokens if len(token) >= _PREFIX_MIN_TOKEN)
    return tokens, prefixes


@dataclass(frozen=True)
class GroundednessReport:
    """Measurement behind `GroundednessRail`, reusable by the harness and the benchmark."""

    score: float
    n_tokens: int
    n_supported: int
    unsupported: tuple[str, ...] = ()
    #: Numerals present in the answer and in neither the context nor the query. Reported
    #: separately because a fabricated figure is a hallucination even at high overall overlap.
    fabricated_numbers: tuple[str, ...] = ()


def measure_groundedness(
    answer_text: str, context: list[ScoredChunk] | None, query: str = ""
) -> GroundednessReport:
    """Fraction of the answer's content words that the retrieved context supports."""
    # Strip citation markers before measuring. The generator is INSTRUCTED to end sentences
    # with [1]/[2], so those digits are our own protocol, not claims the model made about the
    # world -- and counting them made this rail self-defeating: every cited answer reported a
    # "fabricated figure" equal to its own citation index and was refused. Measured on one
    # real answer: 0.82 with fabricated_numbers=('3',) versus 0.90 and clean once stripped.
    #
    # Bounded to 1-2 digits so a genuine bracketed year like [2011] is still checked; the
    # marker index can never exceed settings.context_top_k.
    tokens = content_tokens(_CITATION_MARKER_RE.sub(" ", answer_text))
    if not tokens:
        return GroundednessReport(0.0, 0, 0)

    supported_tokens: set[str] = set()
    prefixes: set[str] = set()
    for scored in context or []:
        # generation_text, not text: the generator was shown the parent/window text, so that
        # is the evidence it could legitimately have used.
        chunk_tokens, chunk_prefixes = _support_index(scored.chunk.generation_text)
        supported_tokens |= chunk_tokens
        prefixes |= chunk_prefixes

    unsupported: list[str] = []
    n_supported = 0
    for token in tokens:
        if token in supported_tokens or (
            len(token) >= _PREFIX_MIN_TOKEN and token[:_PREFIX_LEN] in prefixes
        ):
            n_supported += 1
        else:
            unsupported.append(token)

    # Numerals are checked exactly (no prefix leniency: 1991 must not be excused by 1990) and
    # are forgiven if they came from the query, which the user supplied and the generator did
    # not invent.
    query_tokens = frozenset(lexicons.TOKEN_RE.findall(_normalize(query))) if query else frozenset()
    fabricated = tuple(
        dict.fromkeys(
            token
            for token in tokens
            if _DIGIT_RE.search(token)
            and token not in supported_tokens
            and token not in query_tokens
        )
    )
    return GroundednessReport(
        score=n_supported / len(tokens),
        n_tokens=len(tokens),
        n_supported=n_supported,
        unsupported=tuple(dict.fromkeys(unsupported))[:8],
        fabricated_numbers=fabricated[:8],
    )


class GroundednessRail:
    """The hallucination check. Implements `contracts.Guardrail`.

    Blocks when the answer's content words are not carried by the retrieved passages, or when
    it states a figure that appears nowhere in them. The refusal message replaces the answer.
    """

    name = GROUNDEDNESS_RAIL
    phase = "output"

    def __init__(self, min_groundedness: float | None = None) -> None:
        self.min_groundedness = (
            settings.min_groundedness if min_groundedness is None else min_groundedness
        )

    async def check(
        self,
        *,
        query: str,
        answer: Answer | None = None,
        context: list[ScoredChunk] | None = None,
        lang: Lang | None = None,
    ) -> GuardrailVerdict:
        if answer is None:
            return GuardrailVerdict(name=self.name, allowed=True, detail="no answer to inspect")
        if answer.refused:
            # A refusal is not required to be grounded in anything -- RefusalConsistencyRail
            # owns that case, and measuring the refusal message against the corpus would fail
            # every time and mask the real reason for the refusal.
            return GuardrailVerdict(
                name=self.name, allowed=True, detail="answer is a refusal; groundedness not applicable"
            )

        # Lexical overlap cannot judge an answer written in one language against evidence in
        # another: the two share no tokens even when the answer is a faithful reading of the
        # passage, so the score is ~0 and every cross-language answer is refused as ungrounded.
        # Measured: of the English queries CrossLanguageRail had already ALLOWED -- i.e. a
        # verifier had confirmed the passages answer them -- 90% were then refused here, which
        # is most of the reason the English path answers nothing end to end.
        #
        # Skipping is safe precisely because that rail exists: a cross-language pair reaching
        # this point in QUALITY has been through an entailment check, and in FAST it was refused
        # before generation. This is the same argument, and the same helper, that
        # `grounding.EntityGroundingRail` uses to skip the pairs it cannot judge lexically --
        # a rail that cannot see the question must say so, not answer it with a zero.
        if context and not same_language_hits(context, lang):
            return GuardrailVerdict(
                name=self.name,
                allowed=True,
                detail=(
                    f"evidence is not in {lang.value if lang else 'the query language'}; lexical "
                    "groundedness cannot judge a cross-language answer (CrossLanguageRail owns it)"
                ),
            )

        report = measure_groundedness(answer.text, context, query)

        if report.n_tokens == 0:
            return self._block("answer carries no content words", 1.0)
        if report.fabricated_numbers:
            figures = ", ".join(report.fabricated_numbers[:3])
            return self._block(
                f"groundedness={report.score:.2f}; figures absent from the retrieved passages: {figures}",
                # Certain: a numeral is either present in the context or it is not.
                1.0,
            )
        if report.score < self.min_groundedness:
            missing = ", ".join(report.unsupported[:5])
            return self._block(
                f"groundedness={report.score:.2f} < {self.min_groundedness:.2f} "
                f"({report.n_supported}/{report.n_tokens} content words supported); unsupported: {missing}",
                min(1.0, (self.min_groundedness - report.score) / self.min_groundedness),
            )
        return GuardrailVerdict(
            name=self.name,
            allowed=True,
            detail=f"groundedness={report.score:.2f} ({report.n_supported}/{report.n_tokens} content words supported)",
        )

    def _block(self, detail: str, score: float) -> GuardrailVerdict:
        return GuardrailVerdict(
            name=self.name,
            allowed=False,
            reason=RefusalReason.UNGROUNDED_ANSWER,
            score=score,
            detail=detail,
            rewritten_text=settings.refusal_message,
        )


class CitationRail:
    """Every citation must resolve to a chunk that was actually retrieved.

    Implements `contracts.Guardrail`. A fabricated citation is a hallucination wearing a badge:
    it is worse than an uncited wrong sentence, because the UI renders it as provenance and the
    user stops checking. Four hard failures -- unknown chunk, doc_id mismatch, offsets outside
    the source, quote absent from the source -- and one soft one.
    """

    name = CITATION_RAIL
    phase = "output"

    async def check(
        self,
        *,
        query: str,
        answer: Answer | None = None,
        context: list[ScoredChunk] | None = None,
        lang: Lang | None = None,
    ) -> GuardrailVerdict:
        if answer is None or not answer.citations:
            # Absence of citations is GroundednessRail's problem, not this rail's: the quality
            # generator may answer well without emitting spans, and blocking that here would
            # make citations mandatory in a way the contract does not require.
            return GuardrailVerdict(name=self.name, allowed=True, detail="no citations to verify")

        retrieved = {scored.chunk.chunk_id: scored.chunk for scored in (context or [])}
        problems: list[str] = []
        drifted = 0

        for i, citation in enumerate(answer.citations):
            chunk = retrieved.get(citation.chunk_id)
            if chunk is None:
                problems.append(f"[{i}] cites {citation.chunk_id!r}, which was never retrieved")
                continue
            if citation.doc_id and citation.doc_id != chunk.doc_id:
                problems.append(
                    f"[{i}] claims doc {citation.doc_id!r} but {citation.chunk_id!r} belongs to {chunk.doc_id!r}"
                )
                continue

            problem, drift = self._check_offsets(i, citation, chunk_texts(chunk), answer.text)
            if problem:
                problems.append(problem)
            drifted += drift

        if problems:
            return GuardrailVerdict(
                name=self.name,
                allowed=False,
                reason=RefusalReason.UNGROUNDED_ANSWER,
                score=1.0,  # structural, not statistical: the pointer resolves or it does not
                detail="; ".join(problems[:3]) + (f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""),
                rewritten_text=settings.refusal_message,
            )
        detail = f"{len(answer.citations)} citation(s) resolve to retrieved chunks"
        if drifted:
            detail += f"; {drifted} with offset drift (quote present, offsets stale)"
        return GuardrailVerdict(name=self.name, allowed=True, detail=detail)

    @staticmethod
    def _check_offsets(
        i: int, citation: Citation, sources: tuple[str, ...], answer_text: str
    ) -> tuple[str | None, int]:
        """Returns (hard problem or None, 1 if the citation merely drifted)."""
        if not 0 <= citation.answer_start <= citation.answer_end <= len(answer_text):
            return (
                f"[{i}] answer span [{citation.answer_start}:{citation.answer_end}] "
                f"is outside the {len(answer_text)}-character answer",
                0,
            )
        in_range = [src for src in sources if 0 <= citation.source_start <= citation.source_end <= len(src)]
        if not in_range:
            return (
                f"[{i}] source span [{citation.source_start}:{citation.source_end}] is outside "
                f"chunk {citation.chunk_id!r} ({len(sources[0])} characters)",
                0,
            )
        quote = citation.quote.strip()
        if not quote:
            return None, 0
        if any(src[citation.source_start : citation.source_end].strip() == quote for src in in_range):
            return None, 0
        if any(quote in src for src in sources):
            # The quote is genuinely in the chunk but the offsets point elsewhere -- a
            # highlighting bug, not a fabrication. Downgraded on purpose: refusing a correct
            # answer over an off-by-N span would be a worse failure than mis-highlighting it.
            return None, 1
        return f"[{i}] quotes {quote[:40]!r}, which does not appear in chunk {citation.chunk_id!r}", 0


def chunk_texts(chunk: Chunk) -> tuple[str, ...]:
    """The texts a citation's offsets may legitimately index into.

    `Chunk.generation_text` is what the generator saw, so it is the primary reference, but a
    generator that indexed into the narrower embedded `text` is producing valid provenance too
    -- for PARENT_CHILD and SENTENCE_WINDOW the two differ by the surrounding context.
    """
    generation = chunk.generation_text
    return (generation,) if generation == chunk.text else (generation, chunk.text)


class RefusalConsistencyRail:
    """A refusal must look like a refusal. Implements `contracts.Guardrail`.

    Two directions, both real. A generator that sets `refused=True` while leaving a partial
    answer (or its citations) in place leaks the guess it just decided not to make. A generator
    that emits the refusal *message* without setting the flag makes a refusal look like an
    answer to every downstream consumer -- the analytics, the UI, and the benchmark all read
    the flag, not the prose.
    """

    name = REFUSAL_RAIL
    phase = "output"

    async def check(
        self,
        *,
        query: str,
        answer: Answer | None = None,
        context: list[ScoredChunk] | None = None,
        lang: Lang | None = None,
    ) -> GuardrailVerdict:
        if answer is None:
            return GuardrailVerdict(name=self.name, allowed=True, detail="no answer to inspect")

        expected = settings.refusal_message.strip()
        text = answer.text.strip()

        if answer.refused:
            faults: list[str] = []
            if text != expected:
                faults.append(f"emitted {text[:60]!r} instead of the configured refusal message")
            if answer.citations:
                faults.append(f"carries {len(answer.citations)} citation(s)")
            if faults:
                return GuardrailVerdict(
                    name=self.name,
                    allowed=False,
                    # Keep the generator's own reason when it gave one: overwriting an honest
                    # OFF_TOPIC with a generic code would lose the only diagnosis we have.
                    reason=answer.refusal_reason or RefusalReason.LOW_CONFIDENCE,
                    score=1.0,
                    detail="refusal is malformed: " + "; ".join(faults),
                    rewritten_text=settings.refusal_message,
                )
            return GuardrailVerdict(
                name=self.name,
                allowed=True,
                detail=f"well-formed refusal ({(answer.refusal_reason or RefusalReason.LOW_CONFIDENCE).value})",
            )

        if text == expected:
            return GuardrailVerdict(
                name=self.name,
                allowed=False,
                reason=RefusalReason.NO_RELEVANT_CONTEXT,
                score=1.0,
                detail="answer is the refusal message but refused=False; flagging it as the refusal it is",
                rewritten_text=settings.refusal_message,
            )
        return GuardrailVerdict(name=self.name, allowed=True, detail="answer is not a refusal")


def apply_verdict(answer: Answer, verdict: GuardrailVerdict) -> Answer:
    """Rewrite an answer according to a blocking verdict.

    The rails decide; something has to enforce. Kept here so every caller -- the harness, the
    API, the benchmark -- performs the same substitution: refusal text, refusal flag, reason,
    and citations dropped, because provenance for text that is no longer shown is provenance
    for nothing.
    """
    if verdict.allowed:
        return answer
    return answer.model_copy(
        update={
            "text": verdict.rewritten_text or settings.refusal_message,
            "refused": True,
            "refusal_reason": verdict.reason or RefusalReason.LOW_CONFIDENCE,
            "citations": [],
            "groundedness": 0.0,
        }
    )


__all__ = [
    "CITATION_RAIL",
    "GROUNDEDNESS_RAIL",
    "REFUSAL_RAIL",
    "CitationRail",
    "GroundednessRail",
    "GroundednessReport",
    "RefusalConsistencyRail",
    "apply_verdict",
    "content_tokens",
    "measure_groundedness",
]
