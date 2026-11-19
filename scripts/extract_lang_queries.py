"""Extract per-language query pools and a language-balanced docs sample from docs14.jsonl.

Two artefacts, one pass over the 1.9GB corpus:

  * `data/queries_by_lang.json` -- unique `source_query` strings per language. Training and
    held-out data for the language-ID classifier, and the query pool every per-language
    measurement in this repo should be drawn from.
  * `data/docs_bench14.jsonl` -- a small, language-balanced slice of the corpus in the same
    Document shape. `voxrag.bench` stops scanning at MAX_SCAN_DOCS and docs14.jsonl is
    blocked by language, so pointing `--docs` at the full file can never sample the tail
    languages. Pointing it at this file can.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voxrag.config import DATA_DIR

MIN_QUERY_CHARS = 8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DATA_DIR / "docs14.jsonl")
    parser.add_argument("--queries-per-lang", type=int, default=20_000)
    parser.add_argument("--docs-per-lang", type=int, default=4_000)
    parser.add_argument("--queries-out", type=Path, default=DATA_DIR / "queries_by_lang.json")
    parser.add_argument("--docs-out", type=Path, default=DATA_DIR / "docs_bench14.jsonl")
    args = parser.parse_args()

    queries: dict[str, dict[str, None]] = {}
    docs_written: dict[str, int] = {}
    scanned = 0

    with args.source.open(encoding="utf-8") as src, args.docs_out.open("w", encoding="utf-8") as out:
        for line in src:
            scanned += 1
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            lang = doc.get("lang")
            query = (doc.get("source_query") or "").strip()
            if not lang or len(query) < MIN_QUERY_CHARS:
                continue
            pool = queries.setdefault(lang, {})
            fresh = query not in pool
            if fresh and len(pool) < args.queries_per_lang:
                pool[query] = None
            if fresh and docs_written.get(lang, 0) < args.docs_per_lang:
                out.write(line)
                docs_written[lang] = docs_written.get(lang, 0) + 1

    payload = {lang: list(pool) for lang, pool in sorted(queries.items())}
    args.queries_out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    print(f"scanned {scanned:,} documents")
    for lang in sorted(payload):
        print(f"  {lang:3s} queries={len(payload[lang]):6,d} docs={docs_written.get(lang, 0):5,d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
