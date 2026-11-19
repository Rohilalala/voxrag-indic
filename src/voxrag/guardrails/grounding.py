"""Entity grounding: does the retrieved passage mention what the question is ABOUT?

The problem this exists for, stated precisely. Semantic similarity measures topical proximity,
not answerability. Measured against the live 2.27M-chunk index:

    "what is the price of bitcoin right now"   dense 0.8803   off-topic
    "what is a checking account"               dense 0.8748   legitimate
    "what is the capital of India"             dense 0.8484   off-topic, answered about Stuttgart
    "कॉर्पोरेशन क्या है?"                          dense 0.8793   legitimate

An off-topic query outscores a legitimate one, so NO cosine threshold separates them. That is
structural: MS MARCO covers nearly every topic, so every query has a topical neighbour, and
"passage about a city being a state capital" sits next to "passage naming India's capital" in
embedding space.

What distinguishes them is not similarity but ENTITY OVERLAP. The Stuttgart passage never says
India. The weather passage answers "what is tomorrow's weather" but never says Goa. The
question's distinctive terms -- the ones that make it THIS question rather than a neighbouring
one -- are absent from the evidence.

So this rail asks a narrower question than "is this relevant": are the specific things the user
asked about actually present in what we retrieved? Rare terms carry that weight; common ones do
not. "what is the capital of India" is really a question about *India*, and *capital* alone is
satisfied by any passage about any capital.

WHAT THIS RAIL CANNOT DO, stated up front. It is lexical, so it only works when query and
passage share a language. An English query against an Odia passage has no token overlap even
when the passage answers it perfectly, so a cross-language pair is SKIPPED rather than refused
-- refusing there would break every cross-lingual query, which is a capability this system
otherwise has. That left English queries against this Indic corpus unprotected by this rail; the
gap is now `guardrails/cross_language.py`, which runs in the same set and owns exactly the pairs
this rail hands back. `same_language_hits` below is the single definition of that boundary, called
by both so they cannot disagree about which one is responsible for a request.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Final

from voxrag.config import settings
from voxrag.contracts import (
    Answer,
    GuardrailVerdict,
    Lang,
    RefusalReason,
    ScoredChunk,
)
from voxrag.guardrails import lexicons

GROUNDING_RAIL: Final = "entity_grounding"

#: Terms this short are matched as whole words only. Indic scripts pack a lot into few
#: characters, so a two-character substring match is almost always an accident.
_MIN_TERM_CHARS: Final = 2


def same_language_hits(context: Sequence[ScoredChunk], lang: Lang | None) -> list[ScoredChunk]:
    """The retrieved hits that share the query's language -- empty means a cross-language pair.

    Shared with `guardrails.cross_language.CrossLanguageRail`, which fires on exactly the case
    this returns nothing for. Two copies of the expression would be two definitions of
    "cross-language": one rail would skip a pair the other never picked up, and the gap would
    be invisible because both rails would be reporting themselves as having run.

    `lang is None` means the detector could not name a language, so there is nothing for the
    evidence to be cross-language WITH; every hit counts as same-language and this rail keeps
    the case rather than handing it to a verifier that has no question to ask.
    """
    return [hit for hit in context if lang is None or hit.chunk.lang is lang]


def _terms(text: str) -> list[str]:
    """Content tokens, Indic-aware, stopwords removed.

    Shares `lexicons` normalisation with BM25 and the groundedness rail so that all three
    agree on what a word is -- a rail using a different tokeniser than the index it guards
    produces disagreements that look like bugs in retrieval.
    """
    folded = text.translate(lexicons.NORMALIZE_FOLD).translate(lexicons.DIGIT_FOLD).casefold()
    return [
        token
        for token in lexicons.TOKEN_RE.findall(folded)
        if token not in lexicons.ALL_STOPWORDS and len(token) >= _MIN_TERM_CHARS
    ]


def distinctive_terms(
    query: str, context_texts: list[str], *, top_n: int = 3
) -> list[tuple[str, float]]:
    """The query terms that make it THIS question, rarest first, with their weights.

    Rarity is measured against the retrieved passages themselves rather than a global corpus
    frequency table. That sounds like a shortcut and is actually the right signal here: a term
    appearing in every retrieved passage is what the whole neighbourhood is about and cannot
    discriminate within it, while a term appearing in none is precisely the thing the
    neighbourhood is missing. "capital" is in most passages retrieved for a capital-city
    question; "india" is in none of the wrong ones.
    """
    query_terms = list(dict.fromkeys(_terms(query)))
    if not query_terms:
        return []

    per_doc = [set(_terms(text)) for text in context_texts]
    doc_freq: Counter[str] = Counter()
    for terms in per_doc:
        for term in query_terms:
            if term in terms:
                doc_freq[term] += 1

    n_docs = max(1, len(per_doc))
    # Classic IDF. Ties broken by length so a longer, more specific term wins -- with few
    # passages the frequencies collapse into a handful of values and the tiebreak does real work.
    scored = [
        (term, math.log((n_docs + 1) / (doc_freq[term] + 1)), len(term)) for term in query_terms
    ]
    scored.sort(key=lambda row: (row[1], row[2]), reverse=True)
    return [(term, idf) for term, idf, _ in scored[:top_n]]


class EntityGroundingRail:
    """Refuses when the question's distinctive terms are absent from the evidence.

    Implements `contracts.Guardrail`. Runs after retrieval, alongside `OffTopicRail`.
    """

    name = GROUNDING_RAIL
    phase = "input"

    def __init__(self, min_coverage: float | None = None) -> None:
        self.min_coverage = (
            settings.min_entity_grounding if min_coverage is None else min_coverage
        )

    async def check(
        self,
        *,
        query: str,
        answer: Answer | None = None,
        context: list[ScoredChunk] | None = None,
        lang: Lang | None = None,
    ) -> GuardrailVerdict:
        if not context:
            return GuardrailVerdict(
                name=self.name, allowed=True, detail="no context; OffTopicRail owns this case"
            )

        # Only compare within a language. See the module docstring: a cross-language pair has
        # no token overlap even when it is a perfect answer, so refusing on absence would
        # break every cross-lingual query rather than catching an off-topic one. The pair is
        # not unguarded -- `CrossLanguageRail` runs in the same set and owns it.
        same_lang = same_language_hits(context, lang)
        if not same_lang:
            return GuardrailVerdict(
                name=self.name,
                allowed=True,
                detail=(
                    f"retrieved passages are not in {lang.value if lang else 'the query language'}; "
                    "lexical grounding cannot judge a cross-language pair"
                ),
            )

        texts = [hit.chunk.generation_text for hit in same_lang]
        key_terms = distinctive_terms(query, texts)
        if not key_terms:
            return GuardrailVerdict(
                name=self.name, allowed=True, detail="query carries no distinctive terms"
            )

        haystack = set()
        for text in texts:
            haystack.update(_terms(text))

        # Coverage is IDF-WEIGHTED, not a plain count, and that distinction is the whole rail.
        # "what is the capital of India" yields ['india', 'capital']; a passage about Stuttgart
        # contains 'capital' but not 'india', so an unweighted 1-of-2 would pass it. Weighting
        # makes the missing rare term dominate: 'india' appears in none of the retrieved
        # passages and so carries nearly all the mass, and the pair scores near zero.
        total = sum(idf for _, idf in key_terms)
        if total <= 0.0:
            # Every key term is in every passage. Nothing here can discriminate, and absence
            # of evidence for off-topic is not evidence of off-topic.
            return GuardrailVerdict(
                name=self.name,
                allowed=True,
                detail="all query terms are common to every retrieved passage",
            )

        covered = sum(idf for term, idf in key_terms if term in haystack)
        coverage = covered / total
        missing = [term for term, _ in key_terms if term not in haystack]

        if coverage < self.min_coverage:
            return GuardrailVerdict(
                name=self.name,
                allowed=False,
                reason=RefusalReason.OFF_TOPIC,
                score=1.0 - coverage,
                detail=(
                    f"the passages never mention {', '.join(repr(t) for t in missing)} — "
                    f"they are about something adjacent, not about this. "
                    f"grounding {coverage:.2f} < {self.min_coverage:.2f}"
                ),
            )

        return GuardrailVerdict(
            name=self.name,
            allowed=True,
            detail=(
                f"grounding {coverage:.2f} on "
                f"{', '.join(f'{t}={i:.2f}' for t, i in key_terms)}"
            ),
        )


__all__ = ["GROUNDING_RAIL", "EntityGroundingRail", "distinctive_terms", "same_language_hits"]
