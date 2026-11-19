"""Evaluation harness: how good is this system, measured rather than asserted.

`qrels`     -- relevance judgements reconstructed from MSMARCO-XI's own `is_selected` flags
`retrieval` -- Recall@k / MRR@10 / nDCG@10, with a dense-vs-lexical-vs-hybrid ablation and a
               fair cross-strategy comparison (rankings are collapsed to documents first)

Everything here is measured against ground truth that shipped with the corpus, so the numbers
are reproducible by anyone with the dataset rather than being our own opinion of our output.
"""

from voxrag.eval.qrels import EvalQuery, QrelStats, balanced_sample, load_qrels
from voxrag.eval.retrieval import RANK_CUTOFF, RECALL_AT, evaluate, score_ranking

__all__ = [
    "RANK_CUTOFF",
    "RECALL_AT",
    "EvalQuery",
    "QrelStats",
    "balanced_sample",
    "evaluate",
    "load_qrels",
    "score_ranking",
]
