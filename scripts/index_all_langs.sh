#!/usr/bin/env bash
# Build the 14-language index one language at a time, resumably.
#
# Two problems this solves, both learned the hard way:
#
#   1. A single 2.5-hour `voxrag.index` run over all 1.4M documents has no checkpoint. It was
#      killed at 6% by an unrelated process teardown and there was nothing to resume from.
#      One run per language means a kill costs at most one language, and the completion
#      markers make re-running a no-op for everything already done.
#   2. A job started from an interactive tool dies with that tool's process group. `setsid`
#      (see the launcher at the bottom) puts this in its own session so it survives.
#
# Re-running is always safe: chunk_ids are deterministic, so Qdrant upserts overwrite rather
# than duplicate, and a half-finished language simply redoes its own work.
#
# Usage: setsid nohup scripts/index_all_langs.sh > data/index_all.log 2>&1 &

set -uo pipefail

cd "$(dirname "$0")/.."

DOCS="${DOCS:-data/docs14.jsonl}"
SPLIT_DIR="${SPLIT_DIR:-data/by_lang}"
MARKER_DIR="${MARKER_DIR:-data/.index_markers}"
STRATEGY="${STRATEGY:-parent_child}"
LANGS="${LANGS:-as bn gu hi kn ml mr ne or pa sa ta te ur}"

export VOXRAG_QDRANT_COLLECTION="${VOXRAG_QDRANT_COLLECTION:-msmarco_xi14}"
export VOXRAG_EMBED_THREADS="${VOXRAG_EMBED_THREADS:-10}"

mkdir -p "${SPLIT_DIR}" "${MARKER_DIR}"

# Split once. Guarded by its own marker so a restart does not re-read 1.4M lines.
if [[ ! -f "${SPLIT_DIR}/.done" ]]; then
    echo "[split] $(date +%T) splitting ${DOCS} by language"
    .venv/bin/python - "$DOCS" "$SPLIT_DIR" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path

docs_path, out_dir = sys.argv[1], Path(sys.argv[2])
handles, counts = {}, Counter()
with open(docs_path, encoding="utf-8") as fh:
    for line in fh:
        lang = json.loads(line)["lang"]
        if lang not in handles:
            handles[lang] = (out_dir / f"{lang}.jsonl").open("w", encoding="utf-8")
        handles[lang].write(line)
        counts[lang] += 1
for h in handles.values():
    h.close()
print("  " + "  ".join(f"{k}={v:,}" for k, v in sorted(counts.items())))
PY
    touch "${SPLIT_DIR}/.done"
fi

started=$(date +%s)
for lang in ${LANGS}; do
    marker="${MARKER_DIR}/${lang}.${STRATEGY}.done"
    shard="${SPLIT_DIR}/${lang}.jsonl"

    if [[ -f "${marker}" ]]; then
        echo "[skip ] ${lang}: already indexed"
        continue
    fi
    if [[ ! -s "${shard}" ]]; then
        echo "[warn ] ${lang}: no shard at ${shard}, skipping"
        continue
    fi

    echo "[index] $(date +%T) ${lang} -> ${VOXRAG_QDRANT_COLLECTION} (${STRATEGY})"
    # --no-lexical is REQUIRED here, not an optimisation. BM25Index loads every language
    # already on disk when it opens, so building lexically once per language re-reads the
    # whole growing index each time: O(n^2) I/O that reached 738MB of mmap'd .npy per run
    # and drove the process into uninterruptible disk wait (0.6% CPU, ~1M pageouts) partway
    # through the first language. Vectors are built per language here; the lexical index is
    # built once afterwards by scripts/build_bm25.py in a single pass.
    if .venv/bin/python -u -m voxrag.index \
        --docs "${shard}" --strategy "${STRATEGY}" --batch-size 128 --no-lexical; then
        touch "${marker}"
        echo "[done ] $(date +%T) ${lang}"
    else
        # Do not abort the whole run: one language failing (an OOM, a transient Qdrant
        # hiccup) should not cost the thirteen that would have succeeded after it.
        echo "[FAIL ] ${lang} exited non-zero; continuing with the rest"
    fi
done

echo "[all  ] finished in $(( ($(date +%s) - started) / 60 )) min"
