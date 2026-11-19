"""Independent validation of the entity-grounding rail, on pairs nobody authored.

This project has twice produced a perfect-looking guardrail score by writing both the rule
and the cases that test it (`data/eval_guardrail.json` reports F1 1.000 and means nothing).
So the pairs here are constructed mechanically and the labels come from the corpus, not from
a judgement call:

  POSITIVE  a real query together with the passages retrieved for it, kept only when the
            passage the query came from is among them. The evidence demonstrably contains
            the answer, so a refusal is a FALSE refusal.
  NEGATIVE  the same query against the passages retrieved for a DIFFERENT query in the same
            language. Same language, so the rail engages (it skips cross-language pairs by
            design), and different subject, so a refusal is CORRECT.

It then scores the current matching rule against alternatives on those two sets at once,
because the rail can trivially win either one alone: refusing nothing scores 0% false
refusals, refusing everything catches 100% of the negatives.

    .venv/bin/python scripts/eval_grounding.py --per-lang 40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Any

from voxrag.config import DATA_DIR, settings
from voxrag.contracts import Lang
from voxrag.guardrails.grounding import _terms, distinctive_terms

DEFAULT_DOCS = DATA_DIR / "docs_bench14.jsonl"


def load_pairs(path: Path, per_lang: int, seed: int) -> dict[Lang, list[tuple[str, str]]]:
    """(query, doc_id) per language, from the balanced corpus slice."""
    seen: dict[str, set[str]] = {}
    pairs: dict[str, list[tuple[str, str]]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            doc = json.loads(line)
            lang = doc["lang"]
            query = (doc.get("source_query") or "").strip()
            if len(query) < 8 or query in seen.setdefault(lang, set()):
                continue
            seen[lang].add(query)
            pairs.setdefault(lang, []).append((query, doc["doc_id"]))

    rng = random.Random(seed)
    out: dict[Lang, list[tuple[str, str]]] = {}
    for code, items in pairs.items():
        try:
            lang = Lang(code)
        except ValueError:
            continue
        # Oversample: only the queries whose own passage is retrieved become positives.
        out[lang] = rng.sample(items, k=min(per_lang * 3, len(items)))
    return out


def present_exact(term: str, haystack: set[str], _prefix_k: int) -> bool:
    return term in haystack


def present_prefix(term: str, haystack: set[str], prefix_k: int) -> bool:
    """Match on a shared leading `prefix_k` characters.

    Indic morphology is suffixal: the query says नकदी where the passage says नकद, अर्जयतः
    where the passage says अर्जयन्ति. Comparing truncated forms merges those. `prefix_k` is
    the whole trade -- too short and unrelated words collide, too long and nothing merges --
    so it is a parameter this script sweeps rather than a constant somebody picked.
    """
    if term in haystack:
        return True
    if len(term) < prefix_k:
        return False
    head = term[:prefix_k]
    return any(token.startswith(head) for token in haystack if len(token) >= prefix_k)


RULES: dict[str, Any] = {
    "exact (current)": (present_exact, 0),
    "prefix k=3": (present_prefix, 3),
    "prefix k=4": (present_prefix, 4),
    "prefix k=5": (present_prefix, 5),
    "prefix k=6": (present_prefix, 6),
}


def refuses(query: str, texts: list[str], rule: Any, prefix_k: int, threshold: float) -> bool:
    """The rail's decision, with the term-presence test swapped out."""
    key_terms = distinctive_terms(query, texts)
    if not key_terms:
        return False
    haystack: set[str] = set()
    for text in texts:
        haystack.update(_terms(text))
    total = sum(idf for _, idf in key_terms)
    if total <= 0.0:
        return False
    covered = sum(idf for term, idf in key_terms if rule(term, haystack, prefix_k))
    return (covered / total) < threshold


async def collect(
    pairs: dict[Lang, list[tuple[str, str]]], per_lang: int
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
    """Retrieve once per query; return positive and negative (query, passages) pairs."""
    from voxrag.embed import get_embedder
    from voxrag.store.hybrid import get_hybrid_retriever
    from voxrag.telemetry import Trace

    embedder = get_embedder()
    retriever = get_hybrid_retriever()
    await retriever.warmup()

    positives: list[tuple[str, list[str]]] = []
    negatives: list[tuple[str, list[str]]] = []
    for lang, items in sorted(pairs.items(), key=lambda kv: kv[0].value):
        kept: list[tuple[str, list[str]]] = []
        for query, doc_id in items:
            if len(kept) >= per_lang:
                break
            vector = embedder.encode([query], is_query=True)[0]
            hits = await retriever.retrieve(query, vector, trace=Trace(request_id="grounding-eval"), lang=lang)
            same_lang = [hit for hit in hits if hit.chunk.lang is lang]
            if not same_lang or not any(hit.chunk.doc_id == doc_id for hit in same_lang):
                continue  # the answer passage was not retrieved; this pair proves nothing
            kept.append((query, [hit.chunk.generation_text for hit in same_lang]))
        positives.extend(kept)
        # Rotate by one within the language: every query gets somebody else's evidence.
        negatives.extend(
            (query, kept[(i + 1) % len(kept)][1]) for i, (query, _) in enumerate(kept) if len(kept) > 1
        )
    return positives, negatives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--per-lang", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--threshold", type=float, default=settings.min_entity_grounding, help="min coverage"
    )
    parser.add_argument("--thresholds", default="0.02,0.05,0.1,0.2,0.34,0.5,0.67,0.9")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    pairs = load_pairs(args.docs, args.per_lang, args.seed)
    positives, negatives = asyncio.run(collect(pairs, args.per_lang))
    print(
        f"{len(positives)} positive pairs (answer passage retrieved), "
        f"{len(negatives)} negative pairs (another query's passages), "
        f"threshold {args.threshold}"
    )

    report: dict[str, Any] = {
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "rules": {},
        "thresholds": {},
    }

    def score(rule: Any, prefix_k: int, threshold: float) -> tuple[float, float, float]:
        false_refusals = sum(refuses(q, t, rule, prefix_k, threshold) for q, t in positives)
        caught = sum(refuses(q, t, rule, prefix_k, threshold) for q, t in negatives)
        fr = false_refusals / len(positives) if positives else 0.0
        catch = caught / len(negatives) if negatives else 0.0
        return fr, catch, ((1 - fr) + catch) / 2

    print(f"\nterm matching, at threshold {args.threshold}")
    print(f"{'rule':16s} {'false refusals':>15s} {'off-topic caught':>17s} {'balanced acc':>13s}")
    for name, (rule, prefix_k) in RULES.items():
        fr, catch, balanced = score(rule, prefix_k, args.threshold)
        report["rules"][name] = {
            "false_refusal_rate": round(fr, 4),
            "off_topic_caught": round(catch, 4),
            "balanced_accuracy": round(balanced, 4),
        }
        print(f"{name:16s} {fr:15.1%} {catch:17.1%} {balanced:13.1%}")

    # The threshold was calibrated against whatever retrieval was doing at the time. Improve
    # retrieval and the coverage distribution moves under it, so it is re-swept, not assumed.
    print("\nrefusal threshold, exact matching")
    print(f"{'threshold':16s} {'false refusals':>15s} {'off-topic caught':>17s} {'balanced acc':>13s}")
    for threshold in [float(x) for x in args.thresholds.split(",")]:
        fr, catch, balanced = score(present_exact, 0, threshold)
        report["thresholds"][f"{threshold:g}"] = {
            "false_refusal_rate": round(fr, 4),
            "off_topic_caught": round(catch, 4),
            "balanced_accuracy": round(balanced, 4),
        }
        marker = "  <- configured" if abs(threshold - settings.min_entity_grounding) < 1e-9 else ""
        print(f"{threshold:<16g} {fr:15.1%} {catch:17.1%} {balanced:13.1%}{marker}")

    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
