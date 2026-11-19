"""Train the character-n-gram classifier that separates languages sharing one script.

Script blocks answer "which alphabet is this", which is all a codepoint range can answer.
Four of this corpus's languages share Devanagari (hi/mr/ne/sa) and two share the Bengali
script (bn/as), so for those the script tells you the family and nothing else -- and the
`_SCRIPT_DEFAULT` tie-break that used to resolve them sent every Marathi, Nepali, Sanskrit
and Assamese query to the wrong index partition.

This trains one multinomial Naive Bayes model per ambiguous family over character 3-grams of
real MSMARCO-XI queries, holds out 20% by query hash, and writes the pruned model next to the
detector that loads it. Feature choice was measured, not assumed: char-3 beat char-2 (0.960),
char-4 (0.975 at the same pruning), word unigrams (0.816) and the char-2+3 and word+char-3
combinations on held-out Devanagari accuracy, at a third of char-4's model size.

    .venv/bin/python scripts/train_lang_id.py

Re-run it only when the corpus changes; the model it writes is committed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

from voxrag.config import DATA_DIR, settings
from voxrag.contracts import LANG_SCRIPT, Lang
from voxrag.guardrails.lang_id import MODEL_PATH, NGRAM_N, SCHEMA, ngrams

#: Features kept per language, by frequency. 4000 costs 0.2 points of held-out Devanagari
#: accuracy against keeping everything and cuts the model from 170k features to 23k.
TOP_K_PER_LANG = 4_000

#: Queries whose hash falls in this bucket are held out and never trained on.
TEST_BUCKET = 5


def _is_test(query: str) -> bool:
    digest = hashlib.blake2b(query.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % TEST_BUCKET == 0


def ambiguous_families() -> dict[str, list[Lang]]:
    """Script -> the indexed languages written in it, for scripts carrying more than one."""
    by_script: dict[str, list[Lang]] = defaultdict(list)
    for lang in settings.ingest_langs:
        by_script[LANG_SCRIPT[lang]].append(lang)
    return {script: langs for script, langs in by_script.items() if len(langs) > 1}


def train_family(train: dict[str, list[str]], top_k: int) -> dict[str, dict]:
    """Multinomial Naive Bayes over character n-grams, add-one smoothed."""
    counts = {lang: Counter(g for q in qs for g in ngrams(q)) for lang, qs in train.items()}
    keep: set[str] = set()
    for counter in counts.values():
        keep.update(gram for gram, _ in counter.most_common(top_k))

    vocab: set[str] = set()
    pruned: dict[str, Counter] = {}
    for lang, counter in counts.items():
        kept = Counter({g: n for g, n in counter.items() if g in keep})
        pruned[lang] = kept
        vocab.update(kept)

    size = len(vocab) or 1
    model: dict[str, dict] = {}
    for lang, counter in pruned.items():
        total = sum(counter.values())
        model[lang] = {
            # 4 decimals: the classifier compares sums of thousands of these, and the
            # rounding error is orders of magnitude below the margins it decides on.
            "logp": {g: round(math.log((n + 1.0) / (total + size)), 4) for g, n in counter.items()},
            "unseen": round(math.log(1.0 / (total + size)), 4),
        }
    return model


def predict(model: dict[str, dict], text: str) -> str:
    best, best_score = "", -math.inf
    grams = ngrams(text)
    for lang, params in model.items():
        logp = params["logp"]
        unseen = params["unseen"]
        score = sum(logp.get(g, unseen) for g in grams)
        if score > best_score:
            best, best_score = lang, score
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DATA_DIR / "queries_by_lang.json")
    parser.add_argument("--top-k", type=int, default=TOP_K_PER_LANG)
    parser.add_argument("--out", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    pools: dict[str, list[str]] = json.loads(args.pool.read_text(encoding="utf-8"))
    families = ambiguous_families()

    payload: dict[str, object] = {"schema": SCHEMA, "n": NGRAM_N, "families": {}}
    evaluation: dict[str, object] = {}

    for script, langs in sorted(families.items()):
        codes = [lang.value for lang in langs]
        missing = [code for code in codes if code not in pools]
        if missing:
            print(f"{script}: no query pool for {missing}; skipping this family")
            continue
        train: dict[str, list[str]] = {}
        test: dict[str, list[str]] = {}
        for code in codes:
            train[code] = [q for q in pools[code] if not _is_test(q)]
            test[code] = [q for q in pools[code] if _is_test(q)]

        model = train_family(train, args.top_k)
        payload["families"][script] = model  # type: ignore[index]

        confusion: dict[str, Counter] = {code: Counter() for code in codes}
        started = time.perf_counter()
        n = 0
        for code in codes:
            for query in test[code]:
                confusion[code][predict(model, query)] += 1
                n += 1
        per_query_us = (time.perf_counter() - started) / max(n, 1) * 1e6
        correct = sum(confusion[code][code] for code in codes)
        accuracy = correct / n if n else 0.0
        evaluation[script] = {
            "languages": codes,
            "train_queries": {code: len(train[code]) for code in codes},
            "test_queries": {code: len(test[code]) for code in codes},
            "held_out_accuracy": round(accuracy, 4),
            "per_language_recall": {
                code: round(confusion[code][code] / max(sum(confusion[code].values()), 1), 4)
                for code in codes
            },
            "confusion": {code: dict(counter) for code, counter in confusion.items()},
            "features": sum(len(m["logp"]) for m in model.values()),
            "inference_us_per_query": round(per_query_us, 2),
        }
        print(f"\n=== {script}: {', '.join(codes)} ===")
        print(f"held-out accuracy {accuracy:.4f} over {n:,} queries, {per_query_us:.1f}us/query")
        for code in codes:
            row = confusion[code]
            total = max(sum(row.values()), 1)
            wrong = ", ".join(f"{k}:{v}" for k, v in row.most_common() if k != code) or "-"
            print(f"  {code}: recall {row[code] / total:.4f}   confused with {wrong}")

    payload["evaluation"] = evaluation
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    print(f"\nmodel written to {args.out} ({args.out.stat().st_size / 1024:.0f} KB gzipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
