"""ElevenLabs Scribe speech-to-text -- the secondary hosted provider.

Contract read off elevenlabs.io/docs/api-reference/speech-to-text/convert on 2026-08-15
(NOT live-verified -- we have no ElevenLabs key; see the module report):

    POST {base}/v1/speech-to-text          multipart/form-data
    header  xi-api-key: <key>
    form    file           the audio blob (exactly one of file | source_url)
            model_id       "scribe_v2" (batch SOTA) | "scribe_v1" | "scribe_v1_experimental"
            language_code  ISO-639-1 ("hi") or ISO-639-3 ("hin"); omit to auto-detect
            plus optional diarize / timestamps_granularity / tag_audio_events / keyterms
    200     {"text", "language_code", "language_probability", "words": [...],
             "transcription_id", "audio_duration_secs"}
            multichannel variant nests the above under "transcripts": [...]
    202     async/webhook acknowledgement -- only if webhook=true, which we never send

Note the difference from Sarvam that actually matters: `language_code` here is plain ISO,
so Odia is "or" and not "od-IN". Do not copy Sarvam's mapping into this file.

The realtime WebSocket endpoint (wss://.../v1/speech-to-text/realtime) exists and would
cut perceived latency, but it is a streaming contract; this pipeline hands STT one
complete clip, so the batch endpoint is the correct shape.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Final

import httpx

from voxrag.config import settings
from voxrag.contracts import LANG_FILE_STEM, Lang, Transcript

# Pure MIME->filename helper, shared rather than duplicated. Nothing else is shared.
from voxrag.stt.sarvam import filename_for

#: ISO-639-3 reverse map -- Scribe may answer in either ISO form. LANG_FILE_STEM is
#: already ISO-639-3 for the 14 Indic languages ("hi" -> "hin"), so it doubles as one.
_ISO3_TO_LANG: Final[dict[str, Lang]] = {
    **{stem: lang for lang, stem in LANG_FILE_STEM.items()},
    "eng": Lang.ENG,
}
_ISO1_TO_LANG: Final[dict[str, Lang]] = {lang.value: lang for lang in Lang}

_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_RETRY_BASE_S: Final[float] = 0.25
_RETRY_MAX_S: Final[float] = 2.0
_MIN_AUDIO_BYTES: Final[int] = 512
_HEALTH_TIMEOUT_S: Final[float] = 2.0


class ElevenLabsAuthError(RuntimeError):
    """Missing or rejected credentials. Never retried."""


class ElevenLabsAPIError(RuntimeError):
    """ElevenLabs returned a non-2xx that survived the retry budget."""


def to_elevenlabs_code(lang: Lang | None) -> str | None:
    """ISO-639-1 code, or None meaning "auto-detect" (the field is simply omitted)."""
    return None if lang is None else lang.value


def parse_elevenlabs_lang(code: str | None) -> Lang | None:
    """Parse `language_code`, which arrives as either ISO-639-1 or ISO-639-3."""
    if not code:
        return None
    norm = code.strip().lower().replace("_", "-").split("-", 1)[0]
    if norm in ("unknown", "auto", "und"):
        return None
    return _ISO1_TO_LANG.get(norm) or _ISO3_TO_LANG.get(norm)


_client: httpx.AsyncClient | None = None
_client_loop: asyncio.AbstractEventLoop | None = None


def _get_client() -> httpx.AsyncClient:
    """Process-wide client; see the same note in stt/sarvam.py for the loop-rebind rule."""
    global _client, _client_loop
    loop = asyncio.get_running_loop()
    if _client is None or _client.is_closed or _client_loop is not loop:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.stt_timeout_s, connect=min(5.0, settings.stt_timeout_s)),
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
            follow_redirects=True,
        )
        _client_loop = loop
    return _client


async def aclose() -> None:
    """Release the pooled connections. For server shutdown and tests."""
    global _client, _client_loop
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
    _client_loop = None


def _backoff_s(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(_RETRY_MAX_S, max(0.0, float(retry_after)))
        except ValueError:
            pass  # Retry-After may be an HTTP-date
    return min(_RETRY_MAX_S, _RETRY_BASE_S * (2 ** (attempt - 1))) * (1.0 + random.random() * 0.25)


class ElevenLabsSTT:
    """contracts.STTProvider backed by ElevenLabs Scribe."""

    name: str = "elevenlabs"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.elevenlabs_api_key
        self.model = model or settings.elevenlabs_model
        self.base_url = (base_url or settings.elevenlabs_base_url).rstrip("/")
        self.endpoint = f"{self.base_url}/v1/speech-to-text"

    async def transcribe(
        self, audio: bytes, *, mime_type: str = "audio/wav", lang_hint: Lang | None = None
    ) -> Transcript:
        if len(audio) < _MIN_AUDIO_BYTES:
            return Transcript(
                text="",
                lang=lang_hint,
                provider=self.name,
                raw={"detail": f"audio too short: {len(audio)} bytes", "_voxrag": {"attempts": 0}},
            )
        if not self.api_key:
            raise ElevenLabsAuthError(
                "ElevenLabs API key is not set -- set VOXRAG_ELEVENLABS_API_KEY in .env "
                "(settings.elevenlabs_api_key), or set VOXRAG_STT_PROVIDER=sarvam."
            )

        code = to_elevenlabs_code(lang_hint)
        files = {"file": (filename_for(mime_type), audio, mime_type.split(";", 1)[0] or "audio/wav")}
        headers = {"xi-api-key": self.api_key}

        client = _get_client()
        started = time.perf_counter()
        last_exc: Exception | None = None
        attempt = 0

        # `attempt` is not the loop variable because a language-unsupported retry must not
        # consume the transport-failure budget: they are different failure modes.
        while attempt <= settings.max_retries:
            attempt += 1
            data = {"model_id": self.model}
            if code is not None:
                data["language_code"] = code
            try:
                resp = await client.post(
                    self.endpoint, headers=headers, files=files, data=data,
                    timeout=settings.stt_timeout_s,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt > settings.max_retries:
                    break
                await asyncio.sleep(_backoff_s(attempt, None))
                continue

            if resp.status_code in (401, 403):
                raise ElevenLabsAuthError(
                    f"ElevenLabs rejected the credentials (HTTP {resp.status_code}). Check "
                    "VOXRAG_ELEVENLABS_API_KEY in .env (settings.elevenlabs_api_key)."
                )
            if resp.status_code in _RETRYABLE_STATUS and attempt <= settings.max_retries:
                last_exc = ElevenLabsAPIError(f"HTTP {resp.status_code}")
                await asyncio.sleep(_backoff_s(attempt, resp.headers.get("retry-after")))
                continue
            if resp.status_code == 422 and code is not None and "language" in resp.text.lower():
                # Scribe covers 90+ languages but not necessarily all 14 of ours (Odia and
                # Sanskrit are the likely gaps). An unsupported hint is recoverable: drop it
                # and let Scribe auto-detect rather than failing the whole request.
                code = None
                attempt -= 1  # this retry is a contract fix, not a reliability retry
                continue
            if resp.status_code >= 400:
                raise ElevenLabsAPIError(
                    f"ElevenLabs HTTP {resp.status_code} for model_id={self.model} "
                    f"language_code={code}: {resp.text[:200]}"
                )

            return self._to_transcript(
                resp, code=code, lang_hint=lang_hint, attempts=attempt, started=started
            )

        raise ElevenLabsAPIError(
            f"ElevenLabs unreachable after {settings.max_retries + 1} attempts: {last_exc!r}"
        ) from last_exc

    def _to_transcript(
        self,
        resp: httpx.Response,
        *,
        code: str | None,
        lang_hint: Lang | None,
        attempts: int,
        started: float,
    ) -> Transcript:
        try:
            payload: Any = resp.json()
        except ValueError as exc:
            raise ElevenLabsAPIError(f"ElevenLabs returned non-JSON: {resp.text[:200]}") from exc
        if not isinstance(payload, dict):
            raise ElevenLabsAPIError(f"Unexpected JSON type: {type(payload).__name__}")

        body = payload
        if "text" not in body:
            channels = body.get("transcripts")
            if isinstance(channels, list) and channels and isinstance(channels[0], dict):
                # Multichannel response: our input is a single mic, so channel 0 is the whole
                # of it. Anything else would be a caller bug, not something to merge blindly.
                body = channels[0]
            elif "transcription_id" in body:
                raise ElevenLabsAPIError(
                    "ElevenLabs accepted the job asynchronously (202); this pipeline needs a "
                    "synchronous transcript. Do not enable the `webhook` field."
                )

        text = str(body.get("text") or "").strip()
        lang = parse_elevenlabs_lang(body.get("language_code")) or lang_hint

        prob = body.get("language_probability")
        confidence: float | None = None
        if isinstance(prob, (int, float)):
            confidence = min(1.0, max(0.0, float(prob)))

        duration = payload.get("audio_duration_secs") or body.get("audio_duration_secs")
        duration_s = float(duration) if isinstance(duration, (int, float)) else None

        payload["_voxrag"] = {
            "attempts": attempts,
            "retries": max(0, attempts - 1),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "language_code_sent": code,
            "model": self.model,
        }
        return Transcript(
            text=text,
            lang=lang,
            confidence=confidence,
            provider=self.name,
            duration_s=duration_s,
            raw=payload,
        )

    async def health(self) -> bool:
        """Validate the key against /v1/user -- free, fast, and actually proves authz.

        A reachability-only probe would report healthy with a dead key, which is exactly
        the failure the circuit breaker is meant to route around.
        """
        if not self.api_key:
            return False
        try:
            resp = await _get_client().get(
                f"{self.base_url}/v1/user",
                headers={"xi-api-key": self.api_key},
                timeout=_HEALTH_TIMEOUT_S,
            )
        except Exception:
            return False
        if resp.status_code in (401, 403):
            return False
        # A key scoped without user_read still 4xxs here while transcribing fine, so treat
        # any non-auth, non-server response as reachable.
        return resp.status_code < 500


__all__ = [
    "ElevenLabsAPIError",
    "ElevenLabsAuthError",
    "ElevenLabsSTT",
    "aclose",
    "parse_elevenlabs_lang",
    "to_elevenlabs_code",
]
