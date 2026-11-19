"""The ship gate for `CrossLanguageRail`, on pairs nobody authored.

`scripts/eval_grounding.py` exists because this project twice produced a perfect-looking
guardrail number by writing the rule and its cases together (`data/eval_guardrail.json`
reports F1 1.000 and means nothing). The cross-language rail arrived with the same defect in
a newer costume: fifteen unit tests, every one of them injecting a `FakeVerifier`, so they
prove the rail's control flow and say nothing at all about whether gemma4 can tell an
answerable English/Indic pair from an unanswerable one. This script is the instrument that
can say that.

Both populations are built from the corpus, and the labels come from the index rather than
from a judgement call:

  ON-TOPIC       `Eng_Query` -- the English original that sits beside every translated row
                 in the parquet, so not written by me -- for a row that IS indexed, kept only
                 when a passage from that row's own query_id is actually retrieved. The
                 evidence demonstrably contains the answer, so a refusal is a FALSE refusal.
  OFF-TOPIC      `Eng_Query` for a row that is NOT indexed. Only 10,136 of the ~98k parquet
  (live)         query_ids were ingested (100k docs per language, ~10 passages per query), so
                 the other ~88k are queries this corpus was never given an answer to. They are
                 retrieved live, which means each negative's evidence is the index's own best
                 match out of 2,268,974 vectors -- the same selection process that produced
                 the live failure. A refusal is CORRECT.
  OFF-TOPIC      Ten genuinely off-domain questions, and these ARE authored by me. They are
  (hard)         reported separately and never folded into the headline catch rate, because a
                 set I wrote is not evidence about a rail I also wrote.

The caveat that makes this a LOWER bound, and it belongs in any write-up quoting the number:
MS MARCO is topically dense, so an unindexed query ("what is a checking account") may well be
answerable from an indexed passage about bank accounts. Those are mislabelled negatives, and
they can only push the measured catch rate DOWN. That is the safe direction.

    .venv/bin/python scripts/eval_crosslang_rail.py --out data/eval_crosslang.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from voxrag.config import DATA_DIR, settings
from voxrag.contracts import GenerationMode, Lang, ScoredChunk
from voxrag.guardrails import post_retrieval_rails, run_rails
from voxrag.guardrails.cross_language import CROSS_LANGUAGE_RAIL
from voxrag.harness.orchestrator import _post_retrieval_timeout_ms

BY_LANG = DATA_DIR / "by_lang"
PARQUET = DATA_DIR / "parquet"

#: doc_id and query_id, pulled with a regex rather than json.loads: this reads 1.4M lines
#: across 14 files and the parse is the whole cost.
_DOC_ID_RE = re.compile(rb'"doc_id":"([0-9a-f]+)"')
_QUERY_ID_RE = re.compile(rb'"query_id":(\d+)')

#: The only queries in this file I wrote. Kept apart from the measured sets on purpose --
#: see the module docstring. They are here because a rail that passes the mechanical gate and
#: still answers "what is the price of bitcoin right now" has not solved the reported problem.
HARD_OFF_TOPIC: list[str] = [
    "what is the price of bitcoin right now",
    "who won the football match last night",
    "book me a table for tonight",
    "what is the weather in Mumbai right now",
    "send an email to my manager saying I am sick",
    "what is my account balance",
    "who is the current president of the united states",
    "what time does the next train to Goa leave",
    "translate this sentence into french for me",
    "write me a poem about the sea",
]


#: Wall time of every gemma4 entailment call the rail actually made, filled by
#: `time_entailment`. The rail-set timing below is a superset (it also contains OffTopicRail
#: and EntityGroundingRail), and the ship gate asks for the entailment call specifically, so
#: the two are reported separately rather than one standing in for the other.
_ENTAILMENT_MS: list[float] = []


def time_entailment() -> None:
    """Wrap the live rail's verifier in a stopwatch, changing nothing about what runs.

    `post_retrieval_rails` hands back one `CrossLanguageRail` instance per mode, built at
    import, so wrapping it once here covers every later call. It is wrapped in place, on the
    private attribute, rather than by constructing a substitute rail with a `verifier=`
    argument: a substitute would be a rail this script built, and the whole point of the file
    is that the decisions come from the production object. The wrapper awaits the original and
    returns its verdict untouched -- it can only add timing, never a verdict.
    """
    rail = next(
        r for r in post_retrieval_rails(GenerationMode.QUALITY) if r.name == CROSS_LANGUAGE_RAIL
    )
    inner = rail._verifier

    async def timed(query: str, context: Sequence[ScoredChunk]) -> bool | None:
        started = time.perf_counter()
        try:
            return await inner(query, context)
        finally:
            _ENTAILMENT_MS.append((time.perf_counter() - started) * 1000.0)

    rail._verifier = timed


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sortable list. Empty -> 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def wilson(successes: int, total: int) -> tuple[float, float]:
    """95% Wilson interval, because n=150 makes a 14.0% reading thinner than it looks.

    A gate at 15% decided on a point estimate 1 point below it is a coin flip dressed as a
    measurement; the interval is what says whether the run actually cleared the bar. Wilson
    rather than normal-approximation because these rates sit near 0 and 1, where the normal
    interval runs off the end of the scale.
    """
    if total == 0:
        return (0.0, 0.0)
    z = 1.96
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def indexed_ids() -> tuple[dict[str, int], set[int]]:
    """(doc_id -> query_id) for everything ingested, and the set of indexed query_ids."""
    doc_to_query: dict[str, int] = {}
    query_ids: set[int] = set()
    for path in sorted(BY_LANG.glob("*.jsonl")):
        with path.open("rb") as handle:
            for line in handle:
                doc = _DOC_ID_RE.search(line)
                query = _QUERY_ID_RE.search(line)
                if doc is None or query is None:
                    continue
                qid = int(query.group(1))
                doc_to_query[doc.group(1).decode()] = qid
                query_ids.add(qid)
    return doc_to_query, query_ids


def english_queries(indexed: set[int], seed: int) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """(query_id, Eng_Query) for indexed rows and for rows outside the indexed slice.

    One parquet is enough: the 14 files are translations of the same MS MARCO rows, so
    `Eng_Query` is the same English string in all of them.
    """
    import pandas as pd

    frame = pd.read_parquet(PARQUET / "hinval.parquet", columns=["query_id", "Eng_Query"])
    inside: dict[int, str] = {}
    outside: dict[int, str] = {}
    for qid, query in zip(frame["query_id"], frame["Eng_Query"], strict=True):
        # The parquet's Eng_Query carries MS MARCO's leading ". " on some rows.
        text = str(query).lstrip(". ").strip()
        if len(text) < 8:
            continue
        (inside if int(qid) in indexed else outside)[int(qid)] = text

    rng = random.Random(seed)
    on = sorted(inside.items())
    off = sorted(outside.items())
    rng.shuffle(on)
    rng.shuffle(off)
    return on, off


async def retrieve(query: str) -> list[ScoredChunk]:
    """Exactly the call `orchestrator._retrieve` makes, retry policy included.

    No `lang` filter, because the orchestrator passes one only when the index holds that
    language and it never holds English -- so this is the production shape for this path.
    Two attempts with no backoff for the same reason the orchestrator uses them: measured
    here, a retrieve costs 27-451ms against a 485ms tool budget, so the occasional overrun is
    a timing edge rather than a failure, and dropping the query would bias the sample toward
    whatever retrieves fastest. A query that fails both attempts is skipped, not counted.
    """
    from voxrag.harness.retry import with_retry
    from voxrag.harness.tools import call_tool
    from voxrag.telemetry import Trace

    trace = Trace(request_id="crosslang-eval")
    payload = {"query": query[: settings.max_query_chars], "lang": None, "top_k": settings.fuse_top_k}
    try:
        result = await with_retry(
            lambda: call_tool("retrieve", payload, trace, ctx=None),
            attempts=2,
            backoff_ms=0.0,
            trace=trace,
        )
    except Exception:
        return []
    return list(result.chunks)


async def judge(query: str, chunks: list[ScoredChunk]) -> dict[str, Any]:
    """Run the real post-retrieval set and report both the rail and the pipeline decision.

    The gate is about `CrossLanguageRail`, but a false refusal that `OffTopicRail` would have
    issued anyway is not this rail's fault and not a cost it added, so both are recorded.
    """
    started = time.perf_counter()
    verdicts, blocking = await run_rails(
        post_retrieval_rails(GenerationMode.QUALITY),
        query=query,
        context=chunks,
        lang=Lang.ENG,
        # The production budget, not the stage default. Passing None here runs the set on the
        # 25ms input-guardrail ceiling, `_run_one` converts every entailment call into "rail
        # timed out and was skipped", and the script measures 0% false refusals / 0% caught
        # while believing it measured the rail. That is finding 9 of the review, and it
        # happened in this file first.
        timeout_ms=_post_retrieval_timeout_ms(GenerationMode.QUALITY),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    rail = next(v for v in verdicts if v.name == CROSS_LANGUAGE_RAIL)
    return {
        "query": query,
        "rail_allowed": rail.allowed,
        "rail_detail": rail.detail,
        "blocked_by": blocking.name if blocking is not None else None,
        "langs": sorted({hit.chunk.lang.value for hit in chunks}),
        "ms": round(elapsed_ms, 1),
    }


async def warm_verifier() -> float:
    """Load gemma4 before measuring, and report what the cold call cost.

    Without this the first `check()` eats its whole 1500ms ceiling on a model load and the
    cancellation aborts that load, so every later call is cold too -- the rail measures as
    100% refusal and the measurement is about ollama's residency, not about entailment.
    """
    from voxrag.contracts import Chunk
    from voxrag.generate import get_generator

    generator = get_generator(GenerationMode.QUALITY)
    entails = getattr(generator, "entails", None)
    if entails is None:
        return 0.0
    probe = [
        ScoredChunk(
            chunk=Chunk(
                chunk_id="warm", doc_id="warm", text="A corporation is a legal entity.",
                lang=Lang.ENG, strategy=settings.active_strategy, ordinal=0,
            ),
            score=1.0,
            source="dense",
        )
    ]
    started = time.perf_counter()
    await entails("what is a corporation", probe)
    return (time.perf_counter() - started) * 1000.0


async def collect(
    queries: list[tuple[int, str]],
    limit: int,
    *,
    doc_to_query: dict[str, int] | None,
    label: str,
) -> tuple[list[dict[str, Any]], int]:
    """Retrieve and judge until `limit` usable cases are collected. Returns (rows, tried).

    `doc_to_query` is passed for the on-topic set only: a positive is kept only when a chunk
    from the query's OWN row came back, which is what makes a refusal on it a false refusal
    rather than a retrieval failure being blamed on the rail.

    `tried` is reported because that filter is survivorship: the on-topic population is not
    "English queries" but "English queries whose own passage this index retrieves", and the
    false-refusal rate below is a statement about that population only. The ratio says how
    much of a subset it is.
    """
    out: list[dict[str, Any]] = []
    tried = 0
    for qid, query in queries:
        if len(out) >= limit:
            break
        tried += 1
        chunks = await retrieve(query)
        if not chunks:
            continue
        # The gold passage must be in the slice the VERIFIER SEES, not merely somewhere in the
        # fused list. `entails` judges `settings.context_top_k` passages (4) while retrieval
        # returns `fuse_top_k` (12), so checking the wider list asks the verifier "does this
        # answer it?" about four passages while scoring it against a fifth it was never shown.
        # Measured cost of getting this wrong: it reported 16.2% false refusals, and every one
        # of the 73 came back "the verifier says these passages do not answer it" over evidence
        # spanning eight or nine languages -- correct answers to a question this script asked
        # badly. A rail can only be judged on the evidence it was given.
        judged = chunks[: settings.context_top_k]
        if doc_to_query is not None and not any(
            doc_to_query.get(hit.chunk.doc_id) == qid for hit in judged
        ):
            continue  # its own passage is not in front of the verifier; this pair proves nothing
        row = await judge(query, chunks)
        row["query_id"] = qid
        out.append(row)
        if len(out) % 25 == 0:
            print(f"  {label}: {len(out)}/{limit}", flush=True)
    return out, tried


def refusal_rate(rows: list[dict[str, Any]]) -> float:
    """Fraction the rail refused. On the on-topic set that is the false-refusal rate; on an
    off-topic set it is the catch rate. One direction, so the two cannot be swapped."""
    if not rows:
        return 0.0
    return sum(not row["rail_allowed"] for row in rows) / len(rows)


def skipped(rows: list[dict[str, Any]]) -> int:
    """Cases where `run_rails` timed the rail out and allowed it -- measured as an allow but
    evidence of nothing. A non-zero count invalidates the run rather than lowering a score."""
    return sum("was skipped" in row["rail_detail"] for row in rows)


async def main_async(args: argparse.Namespace) -> int:
    print("reading the indexed slice ...", flush=True)
    doc_to_query, indexed = indexed_ids()
    on_pool, off_pool = english_queries(indexed, args.seed)
    print(
        f"{len(indexed)} indexed query_ids, {len(doc_to_query)} docs; "
        f"{len(on_pool)} indexed / {len(off_pool)} unindexed English queries available",
        flush=True,
    )

    from voxrag.store.hybrid import get_hybrid_retriever

    await get_hybrid_retriever().warmup()
    cold_ms = await warm_verifier()
    print(f"verifier warm-up call: {cold_ms:.0f}ms", flush=True)
    time_entailment()

    on_topic, on_tried = await collect(
        on_pool, args.on_topic, doc_to_query=doc_to_query, label="on-topic"
    )
    off_topic, _ = await collect(off_pool, args.off_topic, doc_to_query=None, label="off-topic")
    hard, _ = await collect(
        [(-1, query) for query in HARD_OFF_TOPIC], len(HARD_OFF_TOPIC), doc_to_query=None, label="hard"
    )

    false_refusals = refusal_rate(on_topic)
    caught = refusal_rate(off_topic)
    caught_hard = refusal_rate(hard)
    n_skipped = skipped(on_topic) + skipped(off_topic) + skipped(hard)
    latencies = [row["ms"] for row in on_topic + off_topic]
    p50 = percentile(latencies, 0.5)
    p90 = percentile(latencies, 0.9)
    ent_p50 = percentile(_ENTAILMENT_MS, 0.5)
    ent_p90 = percentile(_ENTAILMENT_MS, 0.9)
    ent_max = max(_ENTAILMENT_MS) if _ENTAILMENT_MS else 0.0

    fr_lo, fr_hi = wilson(sum(not r["rail_allowed"] for r in on_topic), len(on_topic))
    ca_lo, ca_hi = wilson(sum(not r["rail_allowed"] for r in off_topic), len(off_topic))

    gate = {
        "off_topic_caught>=0.80": caught >= 0.80,
        "false_refusals<=0.15": false_refusals <= 0.15,
        "no_rail_timed_out": n_skipped == 0,
    }
    report: dict[str, Any] = {
        "n_on_topic": len(on_topic),
        "n_on_topic_tried": on_tried,
        "n_off_topic_live": len(off_topic),
        "n_off_topic_hard": len(hard),
        "n_rail_skipped_by_set_timeout": n_skipped,
        "false_refusal_rate": round(false_refusals, 4),
        "false_refusal_ci95": [round(fr_lo, 4), round(fr_hi, 4)],
        "off_topic_caught": round(caught, 4),
        "off_topic_caught_ci95": [round(ca_lo, 4), round(ca_hi, 4)],
        "off_topic_hard_caught": round(caught_hard, 4),
        "post_retrieval_ms_p50": p50,
        "post_retrieval_ms_p90": p90,
        "entailment_calls": len(_ENTAILMENT_MS),
        "entailment_ms_p50": round(ent_p50, 1),
        "entailment_ms_p90": round(ent_p90, 1),
        "entailment_ms_max": round(ent_max, 1),
        "verifier_warmup_ms": round(cold_ms, 1),
        "caveat": (
            "MS MARCO is topically dense, so an unindexed query can still be answerable from "
            "an indexed passage. Those cases are labelled off-topic here and can only push the "
            "measured catch rate DOWN, which makes off_topic_caught a LOWER bound."
        ),
        "gate": gate,
        "passed": all(gate.values()),
        "settings": {
            "cross_language_verification": settings.cross_language_verification,
            "cross_language_timeout_ms": settings.cross_language_timeout_ms,
            "context_top_k": settings.context_top_k,
            "fuse_top_k": settings.fuse_top_k,
            "ollama_model": settings.ollama_model,
            "seed": args.seed,
        },
        "on_topic": on_topic,
        "off_topic_live": off_topic,
        "off_topic_hard": hard,
    }

    print()
    print(f"{'set':22s} {'n':>5s} {'rail refuses':>13s}   {'95% CI':>16s}")
    print(
        f"{'on-topic (must ANSWER)':22s} {len(on_topic):5d} {false_refusals:12.1%}"
        f"   [{fr_lo:5.1%}, {fr_hi:5.1%}]  <- false refusals"
    )
    print(
        f"{'':22s} {'':5s} {'':12s}    kept {len(on_topic)} of {on_tried} tried; "
        "the rest did not retrieve their own passage"
    )
    print(
        f"{'off-topic live':22s} {len(off_topic):5d} {caught:12.1%}"
        f"   [{ca_lo:5.1%}, {ca_hi:5.1%}]  <- caught (LOWER bound, see caveat)"
    )
    print(f"{'off-topic hard (mine)':22s} {len(hard):5d} {caught_hard:12.1%}"
          f"   {'':16s}  <- caught, reported apart: I wrote these")
    print(f"\nentailment call (QUALITY): p50 {ent_p50:.0f}ms  p90 {ent_p90:.0f}ms  "
          f"max {ent_max:.0f}ms  over {len(_ENTAILMENT_MS)} calls, cold call {cold_ms:.0f}ms")
    print(f"post-retrieval rail set:   p50 {p50:.0f}ms  p90 {p90:.0f}ms")
    print(f"rail timed out and was skipped by run_rails: {n_skipped}")
    print(
        "\ncaveat: MS MARCO is topically dense, so some unindexed queries ARE answerable from\n"
        "an indexed passage. Those are mislabelled negatives here, and they can only push the\n"
        "off-topic catch rate DOWN -- the measured number is a LOWER bound, the safe direction."
    )
    for name, ok in gate.items():
        print(f"gate {name:28s} {'PASS' if ok else 'FAIL'}")

    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwritten to {args.out}")
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on-topic", type=int, default=150)
    parser.add_argument("--off-topic", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--out", type=Path, default=None)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
