#!/usr/bin/env bash
# Container entrypoint. Serves immediately; builds the index on first boot only.
#
# Railway gives a container a short window to start listening before the deploy is marked
# failed, and ingesting + embedding a corpus takes minutes. So the server starts FIRST and
# the index is built in the background. /api/health reports the index as empty until the
# build finishes, and the harness degrades to a NO_RELEVANT_CONTEXT refusal rather than
# crashing on an empty index in the meantime.

set -euo pipefail

DATA_DIR="${VOXRAG_DATA_DIR:-/app/data}"
DOCS="${DATA_DIR}/docs.jsonl"
MARKER="${DATA_DIR}/.index_built"
LANGS="${VOXRAG_INGEST_LANGS_CSV:-hi,ta,bn}"
LIMIT="${VOXRAG_INGEST_LIMIT:-8000}"
STRATEGY="${VOXRAG_INDEX_STRATEGY:-parent_child}"

mkdir -p "${DATA_DIR}"

build_index() {
    if [[ -f "${MARKER}" ]]; then
        echo "[entrypoint] index marker present, skipping build"
        return 0
    fi
    echo "[entrypoint] building index: langs=${LANGS} limit=${LIMIT} strategy=${STRATEGY}"
    if [[ ! -f "${DOCS}" ]]; then
        python -m voxrag.ingest.msmarco \
            --langs "${LANGS}" --limit "${LIMIT}" --split validation --out "${DOCS}"
    fi
    python -m voxrag.index --docs "${DOCS}" --strategy "${STRATEGY}"
    touch "${MARKER}"
    echo "[entrypoint] index build complete"
}

# Background the build so the port opens immediately. Failures are logged, not fatal:
# a running server that refuses every query is far more debuggable than a crash loop.
( build_index || echo "[entrypoint] WARNING: index build failed; serving with empty index" ) &

exec python -m uvicorn voxrag.api.app:app \
    --factory=false \
    --host "${VOXRAG_HOST:-0.0.0.0}" \
    --port "${VOXRAG_PORT:-8000}" \
    --log-level "$(echo "${VOXRAG_LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')"
