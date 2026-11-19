"""Contrastive fine-tuning of the retriever on MSMARCO-XI's own relevance labels.

The corpus hands us exactly the supervision this needs: `is_selected` marks which passage
answers which query, and the passages that were retrieved for that query but NOT selected are
hard negatives -- confusable by construction, because a real search engine returned them for
that query. Random negatives would be trivially separable and teach the model nothing.

READ THIS BEFORE RUNNING IT
---------------------------
Fine-tuning the embedder INVALIDATES EVERY VECTOR IN THE INDEX. New weights mean a new
embedding space, so the entire corpus must be re-embedded and re-indexed -- roughly three
hours for 2.1M chunks on this machine, on top of training itself. Budget for that before
starting, not after.

It is also gated on evidence. The query set is split by QUERY (never by passage, which would
leak a query's positives into both halves), the model is scored on held-out queries it never
trained on, and the result is only worth shipping if it beats the base model there. A model
that improves on its training queries and not on held-out ones has memorised, and shipping it
would make retrieval worse in exactly the case that matters -- a question nobody has asked yet.

Requires the optional training extra:  uv pip install "sentence-transformers>=3.0"

Usage:
    python scripts/finetune_embedder.py --docs data/docs14.jsonl --epochs 1
    python scripts/finetune_embedder.py --evaluate-only --model data/models/e5-voxrag
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

DEFAULT_OUT = Path("data/models/e5-voxrag")
#: e5 requires these prefixes at train time too. Training without them and serving with them
#: is a silent train/serve skew that produces a model that is worse than the one you started
#: with, for reasons that look like a bug in retrieval.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def build_pairs(
    docs_path: Path, *, max_per_lang: int | None, seed: int = 0
) -> tuple[list[dict], list[dict]]:
    """(query, positive, hard_negative) triples, split by QUERY ID into train and held-out.

    The split key is `meta.query_id`, not the query text and not (lang, text). MSMARCO-XI is
    ONE corpus translated fourteen ways: the same MS MARCO query carries the same query_id in
    every language. Splitting on text puts the Hindi rendering of a question in train and the
    Tamil rendering of the SAME question in held-out, which is cross-lingual leakage -- the
    model has already seen that question, just spelled differently, and the held-out score
    stops meaning "queries it never saw". Splitting on query_id keeps all fourteen
    translations on the same side of the wall.

    (It also fixes the narrower bug that surfaced it: a query whose translation happens to be
    byte-identical in two languages appeared verbatim in both halves.)
    """
    grouped: dict[int | str, dict[str, list]] = defaultdict(
        lambda: {"pos": [], "neg": [], "langs": []}
    )
    with docs_path.open(encoding="utf-8") as fh:
        for line in fh:
            doc = json.loads(line)
            query = (doc.get("source_query") or "").strip()
            text = (doc.get("text") or "").strip()
            if not query or not text:
                continue
            # Fall back to (lang, text) only if query_id is absent, so the function still
            # works on a docs file produced by an older ingest.
            key = (doc.get("meta") or {}).get("query_id")
            if key is None:
                key = f"{doc['lang']}:{query}"
            bucket = grouped[key]
            bucket["pos" if doc.get("is_selected") else "neg"].append(
                {"lang": doc["lang"], "query": query, "text": text}
            )

    usable = [
        (key, group) for key, group in grouped.items() if group["pos"] and group["neg"]
    ]
    # Split before any per-language capping, so the cap cannot pull one language's copy of a
    # held-out query back into training.
    rng = random.Random(seed)
    rng.shuffle(usable)
    cut = int(len(usable) * 0.9)

    def to_rows(items: Sequence[tuple[object, dict[str, list]]], cap: int | None) -> list[dict]:
        per_lang: dict[str, int] = defaultdict(int)
        rows: list[dict] = []
        for _key, group in items:
            # Every language present for this query becomes its own training row: the point of
            # this corpus is that the same fact exists in fourteen scripts, and the retriever
            # has to work in all of them.
            negatives_by_lang: dict[str, list[str]] = defaultdict(list)
            for neg in group["neg"]:
                negatives_by_lang[neg["lang"]].append(neg["text"])

            for pos in group["pos"]:
                lang = pos["lang"]
                if cap is not None and per_lang[lang] >= cap:
                    continue
                # Same-language negative. A cross-language negative is trivially separable --
                # different script -- so it teaches the model to detect scripts, not meaning.
                pool = negatives_by_lang.get(lang)
                if not pool:
                    continue
                per_lang[lang] += 1
                rows.append(
                    {
                        "lang": lang,
                        "anchor": QUERY_PREFIX + pos["query"],
                        "positive": PASSAGE_PREFIX + pos["text"],
                        "negative": PASSAGE_PREFIX + rng.choice(pool),
                    }
                )
        return rows

    # Held-out is capped proportionally, so it stays a ~10% evaluation set rather than being
    # inflated to the same size as train by an independently-applied per-language cap.
    held_cap = max(1, max_per_lang // 9) if max_per_lang is not None else None
    return to_rows(usable[:cut], max_per_lang), to_rows(usable[cut:], held_cap)


def evaluate_model(model, rows: Sequence[dict]) -> dict[str, float]:
    """Triplet accuracy: how often the positive outranks its hard negative.

    Deliberately simple and self-contained -- the real judgement is `voxrag.eval.retrieval`
    against the live index. This is the cheap signal that says whether re-indexing 2.1M
    chunks is even worth starting.
    """
    import numpy as np

    anchors = model.encode([r["anchor"] for r in rows], normalize_embeddings=True)
    positives = model.encode([r["positive"] for r in rows], normalize_embeddings=True)
    negatives = model.encode([r["negative"] for r in rows], normalize_embeddings=True)

    pos_sim = np.sum(anchors * positives, axis=1)
    neg_sim = np.sum(anchors * negatives, axis=1)
    return {
        "triplet_accuracy": float(np.mean(pos_sim > neg_sim)),
        "mean_positive_sim": float(np.mean(pos_sim)),
        "mean_negative_sim": float(np.mean(neg_sim)),
        "mean_margin": float(np.mean(pos_sim - neg_sim)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=Path("data/docs14.jsonl"))
    parser.add_argument("--base-model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-per-lang", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--model", type=Path, default=None, help="model to evaluate")
    args = parser.parse_args(argv)

    try:
        from sentence_transformers import SentenceTransformer, losses
        from sentence_transformers import SentenceTransformerTrainer as Trainer
        from sentence_transformers import SentenceTransformerTrainingArguments as TrainingArgs
    except ImportError:
        print(
            'sentence-transformers is not installed.\n'
            '  uv pip install "sentence-transformers>=3.0"\n'
            "It pulls in torch (~2GB), which is why it is an optional extra rather than a "
            "dependency of the serving path -- inference runs on ONNX and never needs it."
        )
        return 2

    train_rows, held_out = build_pairs(args.docs, max_per_lang=args.max_per_lang)
    print(f"train {len(train_rows):,} triples | held-out {len(held_out):,} (split by query)")
    if not train_rows or not held_out:
        print("not enough labelled data to train")
        return 1

    if args.evaluate_only:
        model = SentenceTransformer(str(args.model or args.out))
        print(json.dumps(evaluate_model(model, held_out), indent=2))
        return 0

    base = SentenceTransformer(args.base_model)
    before = evaluate_model(base, held_out)
    print(f"BASE  held-out triplet accuracy = {before['triplet_accuracy']:.4f} "
          f"(margin {before['mean_margin']:+.4f})")

    from datasets import Dataset

    dataset = Dataset.from_list(
        [{k: r[k] for k in ("anchor", "positive", "negative")} for r in train_rows]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=base,
        args=TrainingArgs(
            output_dir=str(args.out / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            fp16=False,  # MPS does not support fp16 autocast reliably
            logging_steps=50,
            save_strategy="no",
            report_to=[],
        ),
        train_dataset=dataset,
        # In-batch negatives on top of the supplied hard negative: every other positive in the
        # batch is also a negative for this anchor, which is what makes this loss efficient.
        loss=losses.MultipleNegativesRankingLoss(base),
    )
    trainer.train()

    base.save(str(args.out))
    after = evaluate_model(base, held_out)
    delta = after["triplet_accuracy"] - before["triplet_accuracy"]
    print(
        f"TUNED held-out triplet accuracy = {after['triplet_accuracy']:.4f} "
        f"(margin {after['mean_margin']:+.4f})   delta {delta:+.4f}"
    )

    (args.out / "eval.json").write_text(
        json.dumps({"before": before, "after": after, "delta": delta}, indent=2),
        encoding="utf-8",
    )

    if delta <= 0:
        print(
            "\nDO NOT SHIP: the tuned model is not better on queries it never saw. "
            "Keep the base model."
        )
        return 1
    print(
        f"\nsaved to {args.out}. Shipping it requires re-embedding and re-indexing the whole "
        "corpus (~3h for 2.1M chunks), then re-running python -m voxrag.eval.retrieval to "
        "confirm the gain survives on the live index."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
