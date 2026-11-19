# API image: FastAPI + the ONNX embedder + BM25 + the demo UI. Qdrant is NOT in here.
#
# On Railway this is one of two services; the other runs the stock qdrant/qdrant image and
# is reached over the private network via VOXRAG_QDRANT_URL. Bundling Qdrant into this
# image would mean the embedded client, which silently performs brute-force exact search
# and ignores both hnsw_ef_search and payload indexes (see config.qdrant_url) -- so the
# deployed box would not be running the configuration any latency number was measured on.
#
# The 9.6GB gemma4:e4b generator is NOT in here either -- the container serves the FAST
# path (which is what the <200ms budget is measured on) and points at a hosted generator
# for QUALITY mode via VOXRAG_HOSTED_GENERATOR_URL (see Deploy in the README).

# ---------------------------------------------------------------------------------------
# UI build stage. Node exists ONLY here -- the runtime image below copies the compiled
# bundle and never sees a Node process, which is why the app can be served as static files.
# ---------------------------------------------------------------------------------------
FROM node:26-alpine AS ui

WORKDIR /ui

# Manifests first: `npm ci` is the expensive layer and only package*.json should bust it.
COPY web-ui/package.json web-ui/package-lock.json ./
RUN npm ci

# Copied file-by-file rather than `COPY web-ui/ ./`, which would drag in the host's
# node_modules -- darwin-native binaries that break `vite build` inside a linux image.
COPY web-ui/tsconfig.json web-ui/tsconfig.app.json web-ui/tsconfig.node.json ./
COPY web-ui/vite.config.ts web-ui/index.html ./
COPY web-ui/src/ ./src/
COPY web-ui/public/ ./public/

# vite.config.ts sets outDir to '../web/dist', so this writes /web/dist (not /ui/dist).
RUN npm run build


FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/app/data/hf

# libgomp1 is required by onnxruntime; curl is used by the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer first so application edits do not invalidate the wheel cache.
COPY pyproject.toml README.md ./
RUN uv venv /app/.venv && uv pip install --python /app/.venv/bin/python -r pyproject.toml

ENV PATH="/app/.venv/bin:$PATH"

COPY src/ ./src/
COPY scripts/ ./scripts/

# The UI comes from the build stage, never from the host: copying web/ in would let a stale
# locally-built bundle ship instead of the one built from the sources in this commit.
# FastAPI serves web/dist/index.html at GET / and the assets at /static/dist/.
COPY --from=ui /web/dist/ ./web/dist/
RUN uv pip install --python /app/.venv/bin/python --no-deps -e .

# docs.jsonl, the BM25 index, the HuggingFace cache and the build marker all live here, on a
# Railway volume, so a redeploy does not re-download the corpus or rebuild the vector index.
# The parquet cache is the bulk of it: 419-494MB per language, 6.6GB for all fourteen. Size
# the volume for it, or point HF_HOME elsewhere to keep that cache off it.
VOLUME ["/app/data"]

# VOXRAG_INGEST_LANGS_CSV / VOXRAG_INGEST_LIMIT are read by scripts/entrypoint.sh, not by
# Settings (pydantic ignores them -- the matching fields are `ingest_langs` and
# `ingest_limit_per_lang`). They are pinned here because the entrypoint's own fallbacks are
# hi,ta,bn at 8000 docs, which would deploy 3 of the 14 languages while `settings.ingest_langs`
# still advertises all 14. LanguageRail reads that list, so the other eleven would pass the
# door and then refuse with NO_RELEVANT_CONTEXT on every query -- the failure looks like a
# retrieval bug and is actually a missing index.
#
# 20000 docs/language is the deploy cut: all 14 languages kept, depth traded, ~430k chunks
# = ~660MB of vectors, which fits in a small box's RAM. The vectors have to be resident;
# memory-mapped they measured p100 302ms against 9.4ms resident. Same cut as
# scripts/make_deploy_corpus.py --per-lang 20000.
ENV VOXRAG_HOST=0.0.0.0 \
    VOXRAG_PORT=8000 \
    VOXRAG_VECTOR_STORE=qdrant \
    VOXRAG_DEFAULT_MODE=fast \
    VOXRAG_INGEST_LANGS_CSV=as,bn,gu,hi,kn,ml,mr,ne,or,pa,sa,ta,te,ur \
    VOXRAG_INGEST_LIMIT=20000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${VOXRAG_PORT}/api/health || exit 1

RUN chmod +x /app/scripts/entrypoint.sh
CMD ["/app/scripts/entrypoint.sh"]
