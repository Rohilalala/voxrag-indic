"""Relevance judgements derived from the corpus itself.

MSMARCO-XI ships the labels we need. Every row is a query plus ~10 passages, each flagged
`is_selected`, and the ingest carries both through onto every Document as `source_query` and
`is_selected`. Grouping documents by (lang, source_query) therefore reconstructs the original
retrieval task exactly: the selected passages are the relevant set, the rest are known
negatives drawn from the same corpus.

That last part is what makes the numbers meaningful. Negatives sampled at random from a
corpus are trivially separable and inflate every metric; MS MARCO's negatives were retrieved
for *this* query by a real search engine, so they are genuinely confusable.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from voxrag.contracts import Lang


@dataclass(frozen=True)
class EvalQuery:
    """One graded query: its text, its language, and the documents that answer it."""

    query: str
    lang: Lang
    #: doc_ids flagged is_selected for this query. Never empty.
    relevant: frozenset[str]
    #: doc_ids that appeared for this query and were NOT selected -- the hard negatives.
    negatives: frozenset[str]
    gold_answer: str | None = None

    @property
    def n_candidates(self) -> int:
        return len(self.relevant) + len(self.negatives)


@dataclass
class QrelStats:
    """Corpus-level shape of the judgement set, reported so the metrics can be read fairly."""

    per_lang: dict[str, int] = field(default_factory=dict)
    total_queries: int = 0
    total_relevant: int = 0
    total_candidates: int = 0
    dropped_all_relevant: int = 0
    dropped_no_relevant: int = 0

    @property
    def mean_relevant(self) -> float:
        return self.total_relevant / self.total_queries if self.total_queries else 0.0

    @property
    def mean_candidates(self) -> float:
        return self.total_candidates / self.total_queries if self.total_queries else 0.0


def _iter_docs(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_qrels(
    docs_path: Path | str,
    *,
    langs: Sequence[Lang] | None = None,
    max_queries: int | None = None,
    max_per_lang: int | None = None,
) -> tuple[list[EvalQuery], QrelStats]:
    """Build the graded query set from a docs JSONL.

    Queries with no relevant passage are unanswerable and are dropped. Queries whose passages
    are ALL relevant are dropped too: they cannot separate a good ranker from a bad one, and
    keeping them would lift every system's score by the same amount while hiding differences.
    """
    wanted = {lang.value for lang in langs} if langs else None
    grouped: dict[tuple[str, str], dict[str, set[str] | str | None]] = defaultdict(
        lambda: {"relevant": set(), "negatives": set(), "answer": None}
    )

    for doc in _iter_docs(Path(docs_path)):
        query = (doc.get("source_query") or "").strip()
        if not query:
            continue
        lang = doc["lang"]
        if wanted is not None and lang not in wanted:
            continue
        bucket = grouped[(lang, query)]
        key = "relevant" if doc.get("is_selected") else "negatives"
        bucket[key].add(doc["doc_id"])  # type: ignore[union-attr]
        if bucket["answer"] is None and doc.get("gold_answer"):
            bucket["answer"] = doc["gold_answer"]

    stats = QrelStats()
    per_lang_count: dict[str, int] = defaultdict(int)
    queries: list[EvalQuery] = []

    for (lang, query), bucket in grouped.items():
        relevant: set[str] = bucket["relevant"]  # type: ignore[assignment]
        negatives: set[str] = bucket["negatives"]  # type: ignore[assignment]
        if not relevant:
            stats.dropped_no_relevant += 1
            continue
        if not negatives:
            stats.dropped_all_relevant += 1
            continue
        if max_per_lang is not None and per_lang_count[lang] >= max_per_lang:
            continue

        queries.append(
            EvalQuery(
                query=query,
                lang=Lang(lang),
                relevant=frozenset(relevant),
                negatives=frozenset(negatives),
                gold_answer=bucket["answer"],  # type: ignore[arg-type]
            )
        )
        per_lang_count[lang] += 1
        stats.total_relevant += len(relevant)
        stats.total_candidates += len(relevant) + len(negatives)
        if max_queries is not None and len(queries) >= max_queries:
            break

    stats.total_queries = len(queries)
    stats.per_lang = dict(sorted(per_lang_count.items()))
    return queries, stats


def balanced_sample(
    queries: Iterable[EvalQuery], per_lang: int, *, seed: int = 0
) -> list[EvalQuery]:
    """Take up to `per_lang` queries from each language, deterministically.

    Balanced rather than proportional: a strategy that wins on Devanagari and loses on Tamil
    is a finding, and an unbalanced sample would average it into invisibility.

    Deterministic ordering (sorted by query text, then strided) rather than random.shuffle so
    that two runs -- a baseline and a tuned configuration -- are scored on exactly the same
    queries. Comparing tuning results across different samples measures the sample.
    """
    by_lang: dict[Lang, list[EvalQuery]] = defaultdict(list)
    for query in queries:
        by_lang[query.lang].append(query)

    out: list[EvalQuery] = []
    for lang in sorted(by_lang, key=lambda x: x.value):
        pool = sorted(by_lang[lang], key=lambda q: q.query)
        if len(pool) <= per_lang:
            out.extend(pool)
            continue
        # Stride across the sorted pool instead of taking a prefix: a prefix is biased toward
        # whatever sorts first, which for these scripts means a narrow slice of the alphabet.
        step = len(pool) / per_lang
        out.extend(pool[int(i * step)] for i in range(per_lang))
    return out


__all__ = ["EvalQuery", "QrelStats", "balanced_sample", "load_qrels"]
