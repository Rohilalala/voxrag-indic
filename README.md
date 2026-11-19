# VoxRAG

Ask a question out loud in any of **14 Indic languages** and get a grounded, cited answer back, with the core pipeline answering inside **200 ms**.

```
voice ─▶ Sarvam STT ─▶ input guardrails ─▶ hybrid retrieval (dense + BM25, RRF) ─▶ grounding rails ─▶ cited answer
```

Built on [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI), MS MARCO translated into Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Sanskrit, Tamil, Telugu and Urdu. The served index holds **2,268,974 vectors** from 1.4M documents in all 14 languages.

## Results

All numbers come from this code, measured on the full 14-language index.

| | |
|---|---|
| **Latency** (500 language-balanced queries, everything after speech-to-text) | p50 **56.5 ms** · p70 71.2 ms · p90 94.1 ms · p100 165.2 ms. **100% inside 200 ms**, worst case included |
| **Retrieval** (350 graded queries, document-level) | hybrid nDCG@10 **0.377** · dense 0.351 · BM25 0.275 · hybrid Recall@20 0.651 |
| **Fusion tuning** (560 held-out queries) | +0.0291 nDCG@10, paired tests p = 0.0006 / 0.0011, three repeats |
| **Language routing** (40 real queries per language, no hint) | detection **99.6%** · refusal rate 5.7% |
| **Grounding rail** (474 on-topic + 474 off-topic pairs) | catches **82.3%** of off-topic evidence at 4.4% false refusals |
| **Answers, fast mode** (n = 168) | groundedness **1.000** · token F1 0.208 against an extractive ceiling of 0.684 |
| **Tests** | 611 passing, hermetic: no network, no Qdrant, no model needed |

Retrieval is modest in absolute terms because MS MARCO grades one passage per query. Answer F1 is limited by the extractive generator, not by retrieval: perfect span selection tops out at 0.684 on this data.

## What's inside

- **Speech to text.** Sarvam `saaras:v3` (primary, verified live in Hindi, Bengali and Tamil), ElevenLabs as an alternate and faster-whisper as an offline fallback. Sarvam runs in `transcribe` mode, because `translate` silently returns English and wrecks retrieval against an Indic index.
- **Language routing.** A character-trigram classifier separates languages that share a script (Marathi, Nepali and Sanskrit from Hindi; Assamese from Bengali), so each query searches its own language's partition.
- **Hybrid retrieval.** ONNX `multilingual-e5-small` embeddings in Qdrant (HNSW in RAM) plus an Indic-aware BM25 index, run concurrently and merged with reciprocal-rank fusion. A startup check names any configured language the collection cannot serve.
- **Eight chunking strategies,** benchmarked against MSMARCO-XI's own relevance labels and scored at document level, so fragmenting a document cannot inflate recall. `metadata_aware` is locked off by a test because it leaked the evaluation query into the index.
- **Guardrails.** Input rails for empty queries, unsupported languages, unsafe requests, prompt injection, personal data and off-topic intent; entity grounding after retrieval; output rails for groundedness, citations and refusal consistency. Refusals name the rail that fired.
- **Two answer modes.** `fast` copies cited spans from retrieved passages and stays inside the budget. `quality` uses `gemma4:e4b` through Ollama with `[n]` citations (about 0.9 s per answer on CPU) and falls back to `fast` if the model is down.
- **A real harness.** Typed tools validated with pydantic, per-stage timeouts, jittered retries, circuit breakers and graceful degradation: a dead vector store returns a refusal with timings intact, a dead BM25 index runs dense-only. Every request carries a per-stage trace.
- **Evaluation.** Recall@k, MRR, nDCG, answer F1 against its ceiling, guardrail precision and recall, and paired significance tests, so a tuning change ships only when it beats run-to-run noise.

## Quickstart

Requires Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/), Docker and Node 22+.

```bash
uv sync --extra dev
docker compose up -d                 # Qdrant; embedded Qdrant brute-forces and ignores HNSW
cp .env.example .env                 # add VOXRAG_SARVAM_API_KEY for voice input
export VOXRAG_INGEST_LANGS='["hi","ta","bn"]'   # serve only what this slice indexes

uv run voxrag ingest --langs hi,ta,bn --limit 40000 --local-dir data/parquet --out data/docs.jsonl
uv run voxrag index --docs data/docs.jsonl --strategy parent_child

(cd web-ui && npm ci && npm run build)   # builds the UI into web/dist
uv run voxrag serve                      # http://127.0.0.1:8000
```

For all 14 languages, pass `--langs as,bn,gu,hi,kn,ml,mr,ne,or,pa,sa,ta,te,ur` and drop the `VOXRAG_INGEST_LANGS` override. Every setting lives in `src/voxrag/config.py` and can be overridden with a `VOXRAG_` environment variable.

API: `POST /api/query` · `POST /api/voice` · `GET /api/analytics` · `GET /api/health` · `GET /api/strategies` · WebSocket streaming.

## Evaluate

```bash
uv run pytest -q                                              # hermetic test suite
uv run python -m voxrag.eval                                  # guardrails + retrieval + answers
uv run python -m voxrag.eval.retrieval --strategies all --modes dense,lexical,hybrid
uv run voxrag bench --n 500                                   # p50 / p70 / p100 against the budget
uv run voxrag bench --n 500 --compare                         # fast vs quality on identical queries
uv run voxrag strategies                                      # 8-way chunking comparison
scripts/verify_sarvam.sh                                      # live STT round trip
```

## Deploy

The `Dockerfile` builds the UI and serves the fast path on port 8000; mount a volume at `/app/data` so the corpus and index survive redeploys. The container does not ship the 9.6 GB generator: point `VOXRAG_HOSTED_GENERATOR_URL` at a hosted model for quality mode. `scripts/railway_setup.sh` sets up a Railway service end to end.

## Layout

```
src/voxrag/
  stt/         Sarvam, ElevenLabs, faster-whisper
  ingest/      MSMARCO-XI parquet → documents (streaming, dedup)
  chunking/    8 strategies + comparison harness
  embed/       ONNX multilingual-e5-small, length-bucketed batching, query cache
  store/       Qdrant, HNSW, Indic-aware BM25, RRF fusion
  guardrails/  input, post-retrieval and output rails
  generate/    extractive (fast) and gemma4 (quality), both cited
  harness/     orchestrator, typed tools, retries, breakers, degradation
  bench/       latency percentiles
  eval/        retrieval, answer, guardrail and significance evaluation
  api/         FastAPI + WebSocket
web-ui/        React + Vite front end: mic, latency waterfall, citations, refusal panel
scripts/       data fetch, evaluation and deploy helpers
tests/         hermetic test suite
```

## Set it up with a coding agent

Paste this into your coding agent from the repo root:

```text
Set up VoxRAG locally. Install uv, Docker and Node 22+ if missing. Run `uv sync --extra dev`,
start Qdrant with `docker compose up -d`, and copy .env.example to .env (ask me for a Sarvam
API key, or leave it empty for text-only queries) and export
VOXRAG_INGEST_LANGS='["hi","ta","bn"]'. Ingest a small slice with
`uv run voxrag ingest --langs hi,ta,bn --limit 40000 --local-dir data/parquet --out data/docs.jsonl`,
index it with `uv run voxrag index --docs data/docs.jsonl --strategy parent_child`, build the UI
with `cd web-ui && npm ci && npm run build`, then start `uv run voxrag serve`. Confirm
`uv run pytest -q` passes and `curl http://127.0.0.1:8000/api/health` is healthy, then tell me
the URL.
```

## License

MIT
