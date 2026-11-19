"""REST endpoints.

    POST /api/query       {"text", "lang", "mode"}            -> PipelineResult
    POST /api/voice       multipart: file, lang, mode         -> PipelineResult
    GET  /api/analytics   telemetry.report(TRACES.all()) + n_traces
    GET  /api/health      per-component probes
    GET  /api/strategies  chunking strategies and what is indexed

Failure policy: no endpoint here lets an exception reach the client as a bare 500. Every
error path returns the structured envelope from `voxrag.api.error_body`, which always
carries the `request_id` -- the same id the response's `X-Request-ID` header holds and the
same one the harness stamps on its trace, so a user reporting "it broke" hands over one
string that finds the request in the log, in the trace ring and in the analytics report.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from voxrag.api import HarnessUnavailable, error_body, run_audio, run_text
from voxrag.config import settings
from voxrag.contracts import ChunkStrategyName, GenerationMode, Lang
from voxrag.telemetry import TRACES, report

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

#: Hard ceiling on an uploaded clip. A memory guard, not a product limit: a browser
#: recording of a spoken question is tens of KB, and every STT vendor rejects far smaller
#: payloads than this. Without it, one large POST is an OOM.
MAX_AUDIO_BYTES = 25 * 1024 * 1024

#: Upload read granularity. Streaming rather than `await file.read()` is what makes the
#: cap above enforceable before the bytes are already resident.
_UPLOAD_CHUNK = 1 << 20

#: Container magic numbers, checked before the filename extension: a browser blob saved as
#: `recording.wav` is still webm inside, and the STT provider parses bytes, not names.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x1a\x45\xdf\xa3", "audio/webm"),  # EBML -- what MediaRecorder actually produces
    (b"OggS", "audio/ogg"),
    (b"RIFF", "audio/wav"),
    (b"fLaC", "audio/flac"),
    (b"ID3", "audio/mpeg"),
    (b"FORM", "audio/aiff"),
)

_EXT_MIME: dict[str, str] = {
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".flac": "audio/flac",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
}

#: Last resort. MediaRecorder's default container, which is where nearly all traffic on
#: this endpoint comes from.
_DEFAULT_MIME = "audio/webm"

#: One line per strategy, for the UI's strategy panel. Mirrors the enum documentation in
#: contracts.py, which is comments rather than runtime-readable docstrings.
STRATEGY_DESCRIPTIONS: dict[ChunkStrategyName, str] = {
    ChunkStrategyName.FIXED: "Naive token-count windows, no overlap. The baseline others are measured against.",
    ChunkStrategyName.FIXED_OVERLAP: "Fixed windows sharing an overlap, so a fact split across a boundary survives in one piece.",
    ChunkStrategyName.SENTENCE_WINDOW: "Sentence-aligned chunks, widened with neighbouring sentences for the generator.",
    ChunkStrategyName.SEMANTIC: "Splits where consecutive-sentence embedding distance spikes, i.e. at topic shifts.",
    ChunkStrategyName.RECURSIVE: "Recursive descent through paragraph, sentence and clause separators until the budget fits.",
    ChunkStrategyName.PASSAGE_NATIVE: "One chunk per MS MARCO passage: the corpus is already human-segmented.",
    ChunkStrategyName.PARENT_CHILD: "Small children embedded for retrieval precision, whole parents handed to the generator.",
    ChunkStrategyName.METADATA_AWARE: "Title/language/url folded into the embedded text, making metadata semantically searchable.",
}

#: Ceiling per health probe. /api/health is polled by the container healthcheck and by the
#: UI, so one wedged dependency must not make it hang.
HEALTH_PROBE_TIMEOUT_S = 5.0

#: Components whose failure means the service cannot answer at all. Everything else is
#: reported but degrades: no lexical arm still retrieves densely, no QUALITY generator
#: still answers extractively.
_REQUIRED_COMPONENTS = ("harness", "embedder", "vector_store")


# --------------------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------------------


def coerce_lang(value: Any) -> Lang | None:
    """Accept a language code, or any of the several ways a UI spells "auto-detect"."""
    if value is None or isinstance(value, Lang):
        return value
    text = str(value).strip().lower()
    if text in ("", "auto", "none", "null", "undefined"):
        return None
    try:
        return Lang(text)
    except ValueError as exc:
        known = ", ".join(lang.value for lang in Lang)
        raise ValueError(f"unknown language {text!r}; supported: {known}") from exc


def coerce_mode(value: Any) -> GenerationMode:
    if value is None or (isinstance(value, str) and not value.strip()):
        return settings.default_mode
    if isinstance(value, GenerationMode):
        return value
    try:
        return GenerationMode(str(value).strip().lower())
    except ValueError as exc:
        known = ", ".join(m.value for m in GenerationMode)
        raise ValueError(f"unknown mode {value!r}; expected one of: {known}") from exc


class QueryRequest(BaseModel):
    #: The upper bound is deliberately far above `settings.max_query_chars`. Over-length
    #: input is a *guardrail* case -- the input rails refuse it and the refusal is part of
    #: the demo -- so rejecting it here as a 422 would hide the behaviour being graded.
    #: This limit only stops a multi-megabyte body from reaching the tokeniser.
    text: str = Field(min_length=1, max_length=64_000)
    lang: Lang | None = None
    mode: GenerationMode = settings.default_mode

    @field_validator("lang", mode="before")
    @classmethod
    def _lang(cls, value: Any) -> Lang | None:
        return coerce_lang(value)

    @field_validator("mode", mode="before")
    @classmethod
    def _mode(cls, value: Any) -> GenerationMode:
        return coerce_mode(value)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def origin_allowed(origin: str | None) -> bool:
    """Browsers send Origin on cross-site calls and websocket handshakes, so a page that is
    not allowlisted is refused here. No Origin means a non-browser client, which the rate
    limit covers instead. Origin == Host is deliberately not accepted: under DNS rebinding
    a hostile page's Origin and Host both name the attacker's domain."""
    if not origin:
        return True
    return "*" in settings.cors_origins or origin in settings.cors_origins


_HITS: dict[str, deque[float]] = {}


def within_rate_limit(client: str, now: float | None = None) -> bool:
    """Sliding one-minute window per client. Records the call when it is allowed."""
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return True
    now = time.monotonic() if now is None else now
    hits = _HITS.setdefault(client, deque())
    while hits and now - hits[0] >= 60.0:
        hits.popleft()
    if len(hits) >= limit:
        return False
    hits.append(now)
    # ponytail: in-process and per-worker; move to Redis if this ever runs multi-replica.
    if len(_HITS) > 10_000:
        for key in [k for k, v in _HITS.items() if now - v[-1] >= 60.0]:
            del _HITS[key]
    return True


def client_key(conn: Any) -> str:
    """Who to count a call against: the address the nearest trusted proxy saw, else the peer."""
    hops = settings.trusted_proxy_hops
    if hops > 0:
        chain = [h.strip() for h in conn.headers.get("x-forwarded-for", "").split(",") if h.strip()]
        if len(chain) >= hops:
            return chain[-hops]
    return getattr(conn.client, "host", None) or "unknown"


def _paid_call_refused(request: Request) -> JSONResponse | None:
    if not origin_allowed(request.headers.get("origin")):
        return JSONResponse(
            status_code=403,
            content=error_body(
                _rid(request),
                PermissionError("origin not allowed; add it to VOXRAG_CORS_ORIGINS"),
                error="Forbidden",
            ),
        )
    if not within_rate_limit(client_key(request)):
        return JSONResponse(
            status_code=429,
            content=error_body(
                _rid(request),
                RuntimeError("too many requests; try again in a minute"),
                error="RateLimited",
            ),
        )
    return None


def _harness_or_error(request: Request) -> tuple[Any, JSONResponse | None]:
    harness = getattr(request.app.state, "harness", None)
    if harness is not None:
        return harness, None
    detail = getattr(request.app.state, "harness_error", None) or "harness not initialised"
    return None, JSONResponse(
        status_code=503,
        content=error_body(
            _rid(request),
            HarnessUnavailable(detail),
            message=f"the pipeline is not available: {detail}",
        ),
    )


def sniff_mime(declared: str | None, filename: str | None, head: bytes) -> str:
    """Best available content type for an uploaded clip.

    A browser's declared type wins when it is specific, parameters included: MediaRecorder
    sends `audio/webm;codecs=opus`, and the codec parameter is genuine information the STT
    provider can use. Generic types (`application/octet-stream`, an empty string) are
    ignored in favour of the container's magic bytes, then the extension.
    """
    if declared:
        primary = declared.split(";", 1)[0].strip().lower()
        if primary.startswith("audio/") or primary in ("video/webm", "video/ogg", "video/mp4"):
            return declared.strip()
    for prefix, mime in _MAGIC:
        if head.startswith(prefix):
            return mime
    if len(head) >= 12 and head[4:8] == b"ftyp":  # ISO-BMFF: m4a/mp4 audio
        return "audio/mp4"
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:  # MPEG frame sync
        return "audio/mpeg"
    if filename:
        suffix = filename[filename.rfind(".") :].lower() if "." in filename else ""
        if suffix in _EXT_MIME:
            return _EXT_MIME[suffix]
    return _DEFAULT_MIME


async def _read_upload(file: UploadFile) -> bytes:
    """Read an upload with a hard size cap, raising ValueError past the limit."""
    parts: list[bytes] = []
    total = 0
    while block := await file.read(_UPLOAD_CHUNK):
        total += len(block)
        if total > MAX_AUDIO_BYTES:
            raise ValueError(
                f"audio exceeds {MAX_AUDIO_BYTES // (1024 * 1024)}MB; record a shorter clip"
            )
        parts.append(block)
    return b"".join(parts)


# --------------------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------------------


@router.post("/query")
async def post_query(body: QueryRequest, request: Request) -> JSONResponse:
    """Text query through the full pipeline. Returns a PipelineResult."""
    if (refused := _paid_call_refused(request)) is not None:
        return refused
    rid = _rid(request)
    harness, failure = _harness_or_error(request)
    if failure is not None:
        return failure
    try:
        result = await run_text(
            harness, body.text, lang=body.lang, mode=body.mode, request_id=rid
        )
    except Exception as exc:
        log.exception("query failed [%s]", rid)
        return JSONResponse(status_code=500, content=error_body(rid, exc, stage="pipeline"))
    # JSONResponse rather than a response_model: FastAPI would otherwise re-validate the
    # whole PipelineResult (every chunk, every citation) on the way out, which is pure
    # duplicated work inside the graded latency path.
    return JSONResponse(content=result.model_dump(mode="json"))


@router.post("/voice")
async def post_voice(
    request: Request,
    file: Annotated[UploadFile, File(description="audio clip: webm/opus, wav, mp3, m4a, ogg")],
    lang: Annotated[str | None, Form()] = None,
    mode: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Voice query: speech-to-text, then the same pipeline as /api/query."""
    if (refused := _paid_call_refused(request)) is not None:
        return refused
    rid = _rid(request)
    harness, failure = _harness_or_error(request)
    if failure is not None:
        return failure

    try:
        resolved_lang = coerce_lang(lang)
        resolved_mode = coerce_mode(mode)
        audio = await _read_upload(file)
    except ValueError as exc:
        return JSONResponse(status_code=422, content=error_body(rid, exc, error="ValidationError"))

    if not audio:
        return JSONResponse(
            status_code=422,
            content=error_body(
                rid, ValueError("empty audio upload"), error="ValidationError"
            ),
        )

    mime = sniff_mime(file.content_type, file.filename, audio[:16])
    try:
        result = await run_audio(
            harness,
            audio,
            mime_type=mime,
            lang=resolved_lang,
            mode=resolved_mode,
            request_id=rid,
        )
    except Exception as exc:
        log.exception("voice query failed [%s] mime=%s bytes=%d", rid, mime, len(audio))
        return JSONResponse(
            status_code=500,
            content=error_body(rid, exc, stage="pipeline", mime=mime, bytes=len(audio)),
        )
    return JSONResponse(content=result.model_dump(mode="json"))


# --------------------------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------------------------


@router.get("/analytics")
async def get_analytics() -> dict[str, Any]:
    """Per-stage latency percentiles over every request this process has served.

    Read straight from the process-wide trace ring, so the UI's P50/P70/P100 panel fills in
    live as queries run -- there is no precomputed benchmark artefact behind this.
    """
    traces = TRACES.all()
    payload = report(traces)
    payload["n_traces"] = len(traces)
    return payload


# --------------------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------------------


async def _probe(name: str, coro: Any) -> tuple[str, dict[str, Any]]:
    """Run one health probe under a deadline; a failure is a report, not an exception."""
    try:
        ok, detail = await asyncio.wait_for(coro, timeout=HEALTH_PROBE_TIMEOUT_S)
    except TimeoutError:
        return name, {"ok": False, "detail": f"probe timed out after {HEALTH_PROBE_TIMEOUT_S:.0f}s"}
    except Exception as exc:
        return name, {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    return name, {"ok": bool(ok), "detail": detail}


async def _probe_embedder() -> tuple[bool, str]:
    from voxrag.embed import get_embedder

    # In a warmed process this is a dict lookup; the thread hop only matters on the path
    # where warmup failed and this call is the one paying the model load.
    encoder = await asyncio.to_thread(get_embedder)
    return encoder.dim > 0, f"{encoder.name} dim={encoder.dim}"


async def _probe_vector_store() -> tuple[bool, str]:
    from voxrag.store import get_vector_store

    store = await asyncio.to_thread(get_vector_store)
    healthy = await store.health()
    count = await store.count() if healthy else 0
    info = store.info() if hasattr(store, "info") else {}
    where = info.get("location") or info.get("path") or "in-process"
    # Non-empty is part of healthy on purpose: a reachable but empty index answers every
    # question with a refusal, which is a broken demo, not a healthy service.
    return healthy and count > 0, f"{store.name} @ {where}: {count} vectors"


async def _probe_lexical() -> tuple[bool, str]:
    if not settings.enable_lexical:
        return True, "disabled (settings.enable_lexical=false); dense-only retrieval"
    from voxrag.store.bm25 import get_bm25_index

    index = await asyncio.to_thread(get_bm25_index)
    count = await asyncio.to_thread(index.count)
    langs = ",".join((await asyncio.to_thread(index.info))["langs"]) or "none"
    return count > 0, f"bm25: {count} chunks over [{langs}]"


async def _probe_generator() -> tuple[bool, str]:
    from voxrag.generate import get_generator

    generator = get_generator(settings.default_mode)
    healthy = await generator.health()
    suffix = "" if healthy else " (QUALITY degrades to the extractive FAST path)"
    return healthy, f"{generator.name} mode={generator.mode.value}{suffix}"


async def _probe_stt() -> tuple[bool, str]:
    """Key presence, not a vendor round trip: /api/health is polled every 30s by the
    container healthcheck, and burning STT credits on a liveness probe would be absurd.
    The chain is genuinely exercised once, at startup, by `voxrag.stt.warmup`."""
    keys = {"sarvam": settings.sarvam_api_key, "elevenlabs": settings.elevenlabs_api_key}
    provider = settings.stt_provider
    fallback = settings.stt_fallback_provider
    needed = keys.get(provider, "")
    if provider == "local" or needed:
        return True, f"provider={provider} fallback={fallback or 'none'}"
    # Keyless local Whisper still answers, so this is degraded rather than dead.
    ok = fallback == "local"
    return ok, (
        f"provider={provider} has no API key; "
        + ("serving from the keyless local fallback" if ok else "and no local fallback configured")
    )


@router.get("/health")
async def get_health(request: Request) -> JSONResponse:
    """Probe every component. Always HTTP 200 -- the payload carries the verdict.

    Deliberate: the container healthcheck restarts the service on a non-2xx, and the boot
    sequence serves traffic *while* the index builds in the background. Restart-looping a
    process that is correctly reporting "index still empty" would turn a slow start into a
    permanent outage.
    """
    harness = getattr(request.app.state, "harness", None)
    harness_error = getattr(request.app.state, "harness_error", None)

    async def _harness_state() -> tuple[bool, str]:
        if harness is None:
            return False, harness_error or "not initialised"
        return True, type(harness).__name__

    results = await asyncio.gather(
        _probe("harness", _harness_state()),
        _probe("embedder", _probe_embedder()),
        _probe("vector_store", _probe_vector_store()),
        _probe("lexical", _probe_lexical()),
        _probe("generator", _probe_generator()),
        _probe("stt", _probe_stt()),
    )
    components = dict(results)
    ok = all(components[name]["ok"] for name in _REQUIRED_COMPONENTS)
    return JSONResponse(
        content={
            "ok": ok,
            "components": components,
            "required": list(_REQUIRED_COMPONENTS),
            "warmup": getattr(request.app.state, "warmup", {}),
        }
    )


# --------------------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------------------


@router.get("/strategies")
async def get_strategies() -> list[dict[str, Any]]:
    """Every chunking strategy, and how much of the live index each one contributed.

    Counts come from the manifest the index builder writes, which is the only place that
    knows them: neither store's `count()` breaks down by strategy, and issuing eight
    filtered counts against Qdrant to answer a UI panel would be silly.
    """
    from voxrag.index import load_manifest

    manifest = load_manifest() or {}
    built: dict[str, Any] = manifest.get("strategies") or {}
    return [
        {
            "name": name.value,
            "description": STRATEGY_DESCRIPTIONS[name],
            "n_chunks": (built.get(name.value) or {}).get("chunks"),
            "indexed": name.value in built,
            "active": name is settings.active_strategy,
        }
        for name in ChunkStrategyName
    ]
