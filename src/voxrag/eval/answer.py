"""Answer quality against MSMARCO-XI's gold `Answer` field.

Reports three numbers that must be read together, because any one alone is misleading:

    ceiling   -- is the gold answer's content present in what retrieval returned at all?
    F1        -- token overlap between our answer and the gold answer
    refusal   -- how often we declined a query the corpus CAN answer

The split matters for diagnosis. A low F1 under a high ceiling means the generator is
picking badly from good evidence; a low ceiling means retrieval never gave it a chance, and
tuning the generator would be wasted effort. Refusal rate is the counterweight to both: a
system that refuses everything scores perfect groundedness and is useless, so it has to be
reported next to the quality numbers rather than buried.

Read the F1 with its caveat, which the CLI prints alongside it. The FAST path is EXTRACTIVE:
it returns spans copied verbatim from a passage. The gold answers are ABSTRACTIVE: a human
rewrote the fact in their own words. Token F1 therefore systematically understates extractive
quality -- "the most common form of business organization" and "a corporation is the most
common form of business organisation, chartered by a state" mean the same thing and share
only some tokens. It is still a useful RELATIVE measure between configurations, which is what
it is used for here; it is not an absolute score to boast about.

CLI: python -m voxrag.eval.answer --per-lang 20 --mode fast
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voxrag.contracts import GenerationMode, Lang
from voxrag.eval.qrels import EvalQuery, balanced_sample, load_qrels


@dataclass
class AnswerScore:
    query: str
    lang: str
    f1: float
    exact: bool
    #: Share of the gold answer's content tokens present anywhere in the retrieved context.
    ceiling: float
    refused: bool
    groundedness: float
    latency_ms: float


def _content_tokens(text: str) -> list[str]:
    """Indic-aware content tokens, shared with the guardrails' groundedness measure.

    A whitespace/`\\w` tokeniser scores Devanagari, Tamil and Bengali near zero -- it splits
    combining marks off their base characters -- so an F1 built on it would report that the
    system fails at exactly the languages this corpus is made of.
    """
    from voxrag.guardrails import lexicons

    normalised = (
        text.translate(lexicons.NORMALIZE_FOLD).translate(lexicons.DIGIT_FOLD).casefold()
    )
    return [
        token
        for token in lexicons.TOKEN_RE.findall(normalised)
        if token not in lexicons.ALL_STOPWORDS and (len(token) > 1 or token.isdigit())
    ]


def token_f1(predicted: str, gold: str) -> tuple[float, bool]:
    """SQuAD-style token F1 and exact match over content tokens.

    Multiset (Counter) intersection rather than set intersection: repeating a gold word three
    times should not count as three matches, which a set-based version silently allows.
    """
    pred_tokens, gold_tokens = _content_tokens(predicted), _content_tokens(gold)
    if not pred_tokens or not gold_tokens:
        return (0.0, False)

    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return (0.0, False)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return (f1, pred_tokens == gold_tokens)


def context_ceiling(gold: str, context_texts: Sequence[str]) -> float:
    """Share of the gold answer's content tokens present in the retrieved context."""
    gold_tokens = _content_tokens(gold)
    if not gold_tokens:
        return 0.0
    available: set[str] = set()
    for text in context_texts:
        available.update(_content_tokens(text))
    return sum(1 for token in gold_tokens if token in available) / len(gold_tokens)


async def evaluate(
    queries: Sequence[EvalQuery],
    *,
    mode: GenerationMode = GenerationMode.FAST,
    concurrency: int = 4,
) -> dict[str, Any]:
    """Run the full harness over graded queries and score the answers."""
    from voxrag.harness.orchestrator import get_harness

    harness = get_harness()
    await harness.warmup()
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_one(query: EvalQuery) -> AnswerScore | None:
        if not query.gold_answer:
            return None
        async with semaphore:
            started = time.perf_counter()
            result = await harness.run(text=query.query, lang=query.lang, mode=mode)
            elapsed = (time.perf_counter() - started) * 1000.0

        f1, exact = token_f1(result.answer.text, query.gold_answer)
        return AnswerScore(
            query=query.query,
            lang=query.lang.value,
            # A refusal is not a wrong answer, it is a non-answer: scoring its text against
            # the gold would credit whatever words the refusal message happens to share.
            f1=0.0 if result.answer.refused else f1,
            exact=(not result.answer.refused) and exact,
            ceiling=context_ceiling(
                query.gold_answer, [s.chunk.generation_text for s in result.retrieved]
            ),
            refused=result.answer.refused,
            groundedness=result.answer.groundedness,
            latency_ms=elapsed,
        )

    scored = [s for s in await asyncio.gather(*(run_one(q) for q in queries)) if s is not None]
    return summarise(scored, mode=mode)


def summarise(scores: Sequence[AnswerScore], *, mode: GenerationMode) -> dict[str, Any]:
    def agg(rows: Sequence[AnswerScore]) -> dict[str, Any]:
        if not rows:
            return {"n": 0}
        n = len(rows)
        answered = [r for r in rows if not r.refused]
        return {
            "n": n,
            "f1": sum(r.f1 for r in rows) / n,
            "f1_when_answered": (sum(r.f1 for r in answered) / len(answered)) if answered else 0.0,
            "exact_match": sum(1 for r in rows if r.exact) / n,
            "ceiling": sum(r.ceiling for r in rows) / n,
            "refusal_rate": sum(1 for r in rows if r.refused) / n,
            "groundedness": (
                sum(r.groundedness for r in answered) / len(answered) if answered else 0.0
            ),
            "mean_latency_ms": sum(r.latency_ms for r in rows) / n,
        }

    langs = sorted({s.lang for s in scores})
    return {
        "mode": mode.value,
        "overall": agg(scores),
        "per_lang": {lang: agg([s for s in scores if s.lang == lang]) for lang in langs},
        "caveat": (
            "FAST is extractive (verbatim spans); gold answers are abstractive (rewritten). "
            "Token F1 systematically understates extractive quality and is meaningful as a "
            "relative measure between configurations, not as an absolute score."
        ),
    }


def _report(result: dict[str, Any]) -> None:
    from rich.console import Console
    from rich.table import Table

    o = result["overall"]
    table = Table(title=f"answer quality vs gold Answer ({result['mode']} mode)")
    for col in ("lang", "n", "ceiling", "F1", "F1|answered", "EM", "refusal", "grounded", "ms"):
        table.add_column(col, justify="right" if col != "lang" else "left")

    for lang, row in [*result["per_lang"].items(), ("ALL", o)]:
        if not row.get("n"):
            continue
        table.add_row(
            lang,
            str(row["n"]),
            f"{row['ceiling']:.3f}",
            f"{row['f1']:.3f}",
            f"{row['f1_when_answered']:.3f}",
            f"{row['exact_match']:.3f}",
            f"{row['refusal_rate']:.3f}",
            f"{row['groundedness']:.3f}",
            f"{row['mean_latency_ms']:.0f}",
        )

    console = Console()
    console.print(table)
    console.print(f"[dim]{result['caveat']}[/dim]")
    if o.get("n"):
        # State the diagnosis rather than leaving the reader to infer it.
        if o["ceiling"] < 0.5:
            console.print(
                "[yellow]diagnosis:[/yellow] low ceiling -- retrieval is the bottleneck, "
                "the generator is rarely given the evidence it needs"
            )
        elif o["f1_when_answered"] < o["ceiling"] * 0.6:
            console.print(
                "[yellow]diagnosis:[/yellow] ceiling is high but F1 is not -- the evidence is "
                "being retrieved and then chosen from badly; tune the generator, not retrieval"
            )


async def _main_async(args: argparse.Namespace) -> int:
    langs = [Lang(x) for x in args.langs.split(",")] if args.langs else None
    queries, _stats = load_qrels(args.docs, langs=langs)
    sample = balanced_sample(queries, args.per_lang)
    print(f"scoring {len(sample):,} queries ({args.per_lang}/language) in {args.mode} mode\n")

    result = await evaluate(
        sample, mode=GenerationMode(args.mode), concurrency=args.concurrency
    )
    _report(result)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="data/docs14.jsonl")
    parser.add_argument("--langs", default=None)
    parser.add_argument("--per-lang", type=int, default=20)
    parser.add_argument("--mode", default="fast", choices=[m.value for m in GenerationMode])
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", default="data/eval_answer.json")
    return asyncio.run(_main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
