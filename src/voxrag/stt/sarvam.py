"""Sarvam AI speech-to-text -- the primary provider.

Sarvam is primary because the corpus is Indic: it covers all 14 MSMARCO-XI languages
natively and returns text in the source script, which is what the index is built on.

The wire contract below was verified against the live API on 2026-08-15 (see
scripts/verify_sarvam.sh, which is the executable ground truth):

    POST {base}/speech-to-text
    header  api-subscription-key: <key>        <- NOT Authorization, NOT Bearer
    form    file, model, language_code, mode
    200     {"request_id", "transcript", "language_code", "timestamps",
             "language_probability"}

Measured round trip is 1.5-2.7s. That is why StageName.CORE (total minus STT) is the
number the 200ms budget is graded on -- a third-party network hop is not optimisable.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Final

import httpx

from voxrag.config import settings
from voxrag.contracts import LANG_FILE_STEM, Lang, Transcript

# --------------------------------------------------------------------------------------
# Language codes
# --------------------------------------------------------------------------------------

#: Sarvam speaks BCP-47-ish region-tagged codes, NOT the ISO-639-1 codes in contracts.Lang.
#: The dangerous entry is Odia: Sarvam's code is `od-IN` while ISO (and Lang.ORI) says `or`.
#: Sending `or-IN` does not error -- it is accepted and silently transcribed as some other
#: language, which is far worse than a 400. Every code below is from the documented valid
#: set: hi bn kn ml mr od pa ta te en gu as ur ne sa, all suffixed `-IN`.
SARVAM_LANG: Final[dict[Lang, str]] = {
    Lang.ASM: "as-IN",
    Lang.BEN: "bn-IN",
    Lang.GUJ: "gu-IN",
    Lang.HIN: "hi-IN",
    Lang.KAN: "kn-IN",
    Lang.MAL: "ml-IN",
    Lang.MAR: "mr-IN",
    Lang.NEP: "ne-IN",
    Lang.ORI: "od-IN",  # <- the trap; ISO-639-1 for Odia is "or"
    Lang.PAN: "pa-IN",
    Lang.SAN: "sa-IN",
    Lang.TAM: "ta-IN",
    Lang.TEL: "te-IN",
    Lang.URD: "ur-IN",
    Lang.ENG: "en-IN",
}

#: Exact reverse map, lowercase-keyed.
_SARVAM_TO_LANG: Final[dict[str, Lang]] = {v.lower(): k for k, v in SARVAM_LANG.items()}

#: Primary-subtag reverse map, so a bare "hi" or an unexpected "hi-Latn" still parses.
#: "or" is registered alongside "od" because the response field is not contractually
#: guaranteed to echo the request form.
_PRIMARY_TO_LANG: Final[dict[str, Lang]] = {
    **{code.split("-", 1)[0]: lang for lang, code in SARVAM_LANG.items()},
    "or": Lang.ORI,
}

#: ISO-639-3 reverse map. LANG_FILE_STEM already is ISO-639-3 for the 14 Indic languages.
_ISO3_TO_LANG: Final[dict[str, Lang]] = {
    **{stem: lang for lang, stem in LANG_FILE_STEM.items()},
    "eng": Lang.ENG,
}


def to_sarvam_code(lang: Lang | None) -> str:
    """Language code to send. Falls back to auto-detect when there is no hint."""
    if lang is None:
        return settings.sarvam_auto_language
    return SARVAM_LANG.get(lang, settings.sarvam_auto_language)


def parse_sarvam_lang(code: str | None) -> Lang | None:
    """Parse Sarvam's `language_code` back into a Lang, tolerantly.

    Returns None rather than guessing when the code is absent or unrecognised, so the
    caller can fall back to its own hint instead of indexing against a wrong language.
    """
    if not code:
        return None
    norm = code.strip().lower()
    if norm in ("unknown", "auto", "und"):
        return None
    hit = _SARVAM_TO_LANG.get(norm)
    if hit is not None:
        return hit
    primary = norm.replace("_", "-").split("-", 1)[0]
    return _PRIMARY_TO_LANG.get(primary) or _ISO3_TO_LANG.get(primary)


# --------------------------------------------------------------------------------------
# Module constants (not in config.py -- they are Sarvam wire details, not tunables)
# --------------------------------------------------------------------------------------

#: Statuses worth a second attempt. 401/403 are deliberately absent: retrying an auth
#: failure only multiplies the user's wait for an outcome that cannot change.
_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
#: Backoff base. settings.retry_backoff_ms (40ms) is sized for in-process stage retries;
#: 40ms of backoff against a rate-limited vendor is indistinguishable from no backoff.
_RETRY_BASE_S: Final[float] = 0.25
#: Backoff ceiling, including any server-supplied Retry-After. A user is waiting on this
#: request and the keyless local fallback is cheaper than honouring a 60s Retry-After.
_RETRY_MAX_S: Final[float] = 2.0
#: Below this there is no speech, only a container header -- skip the round trip.
_MIN_AUDIO_BYTES: Final[int] = 512
#: Reachability probe only; never gated on the (slow) transcribe path.
_HEALTH_TIMEOUT_S: Final[float] = 2.0

_MIME_EXT: Final[dict[str, str]] = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/vnd.wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "audio/aac": "aac",
    "audio/aiff": "aiff",
    "audio/x-aiff": "aiff",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/ogg": "ogg",
    "audio/opus": "opus",
    "audio/webm": "webm",
    "video/webm": "webm",
}


class SarvamAuthError(RuntimeError):
    """Missing or rejected credentials. Never retried."""


class SarvamAPIError(RuntimeError):
    """Sarvam returned a non-2xx that survived the retry budget."""


def filename_for(mime_type: str) -> str:
    """Multipart filename with a truthful extension.

    Sniffing is done on the extension by more than one audio stack, and a browser sends
    `audio/webm;codecs=opus` -- the parameters have to be stripped before lookup.
    """
    base = (mime_type or "").split(";", 1)[0].strip().lower()
    ext = _MIME_EXT.get(base)
    if ext is None:
        sub = base.rsplit("/", 1)[-1]
        ext = sub if sub.isalnum() and 0 < len(sub) <= 5 else "wav"
    return f"audio.{ext}"


# --------------------------------------------------------------------------------------
# Shared client
# --------------------------------------------------------------------------------------

_client: httpx.AsyncClient | None = None
_client_loop: asyncio.AbstractEventLoop | None = None


def _get_client() -> httpx.AsyncClient:
    """One AsyncClient for the process, rebuilt only if the event loop changed.

    A per-request client pays DNS + TCP + TLS on every call, which on a 1.5s budget is a
    visible fraction of it. The loop check exists because the connection pool holds
    sockets registered with the loop that created them: the benchmark runs several
    `asyncio.run()` calls, and reusing a pool across loops raises at request time. The
    orphaned client is left to GC -- closing it would need an await we cannot do here,
    and its loop is already dead.
    """
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
    """Exponential backoff with jitter; jitter stops N concurrent requests re-colliding."""
    if retry_after:
        try:
            return min(_RETRY_MAX_S, max(0.0, float(retry_after)))
        except ValueError:
            pass  # Retry-After may be an HTTP-date; fall through to exponential
    return min(_RETRY_MAX_S, _RETRY_BASE_S * (2 ** (attempt - 1))) * (1.0 + random.random() * 0.25)


# --------------------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------------------


class SarvamSTT:
    """contracts.STTProvider backed by Sarvam's saaras models."""

    name: str = "sarvam"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.sarvam_api_key
        self.model = model or settings.sarvam_model
        self.base_url = (base_url or settings.sarvam_base_url).rstrip("/")
        self.endpoint = f"{self.base_url}/speech-to-text"

    async def transcribe(
        self, audio: bytes, *, mime_type: str = "audio/wav", lang_hint: Lang | None = None
    ) -> Transcript:
        if len(audio) < _MIN_AUDIO_BYTES:
            # Not an error: an empty clip is a legitimate outcome of a mis-fired mic. The
            # EMPTY_TRANSCRIPT guardrail turns this into a refusal downstream.
            return Transcript(
                text="",
                lang=lang_hint,
                provider=self.name,
                raw={"detail": f"audio too short: {len(audio)} bytes", "_voxrag": {"attempts": 0}},
            )
        if not self.api_key:
            raise SarvamAuthError(
                "Sarvam API key is not set -- set VOXRAG_SARVAM_API_KEY in .env "
                "(settings.sarvam_api_key), or set VOXRAG_STT_PROVIDER=local."
            )

        code = to_sarvam_code(lang_hint)
        files = {"file": (filename_for(mime_type), audio, mime_type.split(";", 1)[0] or "audio/wav")}
        # `mode` comes from config: "transcribe" keeps the source script. "translate" would
        # return English and quietly destroy retrieval against an Indic index.
        data = {"model": self.model, "language_code": code, "mode": settings.sarvam_mode}
        headers = {"api-subscription-key": self.api_key}

        client = _get_client()
        started = time.perf_counter()
        last_exc: Exception | None = None

        for attempt in range(1, settings.max_retries + 2):
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
                raise SarvamAuthError(
                    f"Sarvam rejected the credentials (HTTP {resp.status_code}). Check "
                    "VOXRAG_SARVAM_API_KEY in .env (settings.sarvam_api_key)."
                )
            if resp.status_code in _RETRYABLE_STATUS and attempt <= settings.max_retries:
                last_exc = SarvamAPIError(f"HTTP {resp.status_code}")
                await asyncio.sleep(_backoff_s(attempt, resp.headers.get("retry-after")))
                continue
            if resp.status_code >= 400:
                # Truncate: an error body can be long, and it must never echo the key back.
                raise SarvamAPIError(
                    f"Sarvam HTTP {resp.status_code} for model={self.model} "
                    f"language_code={code}: {resp.text[:200]}"
                )

            return self._to_transcript(
                resp, code=code, lang_hint=lang_hint, attempts=attempt, started=started
            )

        raise SarvamAPIError(
            f"Sarvam unreachable after {settings.max_retries + 1} attempts: {last_exc!r}"
        ) from last_exc

    def _to_transcript(
        self,
        resp: httpx.Response,
        *,
        code: str,
        lang_hint: Lang | None,
        attempts: int,
        started: float,
    ) -> Transcript:
        try:
            payload: Any = resp.json()
        except ValueError as exc:
            raise SarvamAPIError(f"Sarvam returned non-JSON: {resp.text[:200]}") from exc
        if not isinstance(payload, dict):
            raise SarvamAPIError(f"Sarvam returned unexpected JSON type: {type(payload).__name__}")

        text = str(payload.get("transcript") or "").strip()
        # Trust the detected language over the hint, but keep the hint when detection is
        # absent (it is null whenever an explicit language_code was sent).
        lang = parse_sarvam_lang(payload.get("language_code")) or lang_hint

        prob = payload.get("language_probability")
        confidence: float | None = None
        if isinstance(prob, (int, float)):
            # Note this is Sarvam's confidence in the *language*, not in the words. It is
            # the only per-request confidence the API exposes, so it is what Transcript
            # carries; the groundedness guardrail is what actually protects the answer.
            confidence = min(1.0, max(0.0, float(prob)))

        # Namespaced so it cannot collide with a vendor field. transcribe_with_fallback
        # folds `retries` into Trace.retries, which the analytics report sums.
        payload["_voxrag"] = {
            "attempts": attempts,
            "retries": attempts - 1,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "language_code_sent": code,
            "model": self.model,
        }
        return Transcript(
            text=text, lang=lang, confidence=confidence, provider=self.name, raw=payload
        )

    async def health(self) -> bool:
        """Key present and host reachable.

        Deliberately does not transcribe: a probe clip costs 1.5s and real credits on
        every health poll. Any HTTP response below 500 proves DNS/TLS/routing are fine.
        """
        if not self.api_key:
            return False
        try:
            resp = await _get_client().get(self.base_url, timeout=_HEALTH_TIMEOUT_S)
        except Exception:
            return False
        return resp.status_code < 500


__all__ = [
    "SARVAM_LANG",
    "SarvamAPIError",
    "SarvamAuthError",
    "SarvamSTT",
    "aclose",
    "filename_for",
    "parse_sarvam_lang",
    "to_sarvam_code",
]
