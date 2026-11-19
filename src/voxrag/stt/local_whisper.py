"""Keyless local speech-to-text -- the fallback that makes the system demoable offline.

Its job is to never be the reason a request dies. It therefore has no hard dependency of
any kind: every backend is imported inside the call, and if none of them exist it returns
an empty Transcript carrying install instructions instead of raising.

Backends, in preference order:
  1. faster-whisper  -- CTranslate2 int8, ~4x realtime on CPU, decodes bytes in-process.
  2. openai-whisper  -- reference implementation; needs ffmpeg on PATH and a temp file.
  3. ollama          -- settings.ollama_model (gemma4:e4b) is multimodal and takes audio,
                        so a box already running the quality generator gets STT for free.
  4. nothing         -- empty transcript + a detail string naming what to install.

None of these are in pyproject.toml deliberately: faster-whisper drags in PyAV and
openai-whisper drags in torch, and neither belongs in the image that has to hit a 200ms
retrieval budget. This module upgrades itself if they happen to be present.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import os
import tempfile
import threading
import time
from typing import Any, Final

import httpx

from voxrag.config import settings
from voxrag.contracts import Lang, Transcript
from voxrag.stt.sarvam import filename_for

#: "small" is the smallest multilingual Whisper checkpoint that is usable on Indic audio;
#: "base" hallucinates badly outside Latin scripts. int8 keeps it ~250MB resident.
_WHISPER_MODEL_SIZE: Final[str] = "small"
_WHISPER_COMPUTE_TYPE: Final[str] = "int8"
#: Greedy decoding. This is a fallback path -- a 5x beam search buys a little accuracy for
#: several seconds of CPU, and the user is already waiting because a vendor died.
_WHISPER_BEAM_SIZE: Final[int] = 1
_MIN_AUDIO_BYTES: Final[int] = 512
_HEALTH_TIMEOUT_S: Final[float] = 2.0

#: Whisper's language table has no Odia. Passing "or" raises ValueError, so ORI (and any
#: future gap) must degrade to auto-detect rather than to an exception.
_WHISPER_UNSUPPORTED: Final[frozenset[Lang]] = frozenset({Lang.ORI})

_OLLAMA_PROMPT: Final[str] = (
    "Transcribe the speech in this audio verbatim. Keep the original language and its "
    "native script -- do not translate, do not transliterate, do not summarise. "
    "Reply with the transcription only and nothing else."
)

_INSTALL_HINT: Final[str] = (
    "No local speech-to-text backend available. Install one of: "
    "`uv pip install faster-whisper` (recommended), `uv pip install openai-whisper` "
    "(also needs ffmpeg on PATH), or run `ollama pull " + settings.ollama_model + "` "
    "with the Ollama server up. Alternatively set VOXRAG_SARVAM_API_KEY and use the "
    "hosted provider."
)

# --------------------------------------------------------------------------------------
# Model cache
# --------------------------------------------------------------------------------------

_LOAD_LOCK = threading.Lock()
_BACKEND: str | None = None  # None = not probed yet; "none" = probed, nothing available
_MODEL: Any = None


def _load_model() -> tuple[str, Any] | None:
    """Import and construct a Whisper backend once, in a worker thread.

    Runs under a threading.Lock rather than an asyncio.Lock because the caller already
    pushes it through asyncio.to_thread -- model construction reads weights off disk and
    would otherwise stall the event loop for seconds. The `"none"` sentinel is important:
    without it every request re-pays a failing import, which is not free.
    """
    global _BACKEND, _MODEL
    with _LOAD_LOCK:
        if _BACKEND is not None:
            return None if _BACKEND == "none" else (_BACKEND, _MODEL)

        try:
            from faster_whisper import WhisperModel

            _MODEL = WhisperModel(
                _WHISPER_MODEL_SIZE, device="cpu", compute_type=_WHISPER_COMPUTE_TYPE
            )
            _BACKEND = "faster-whisper"
            return (_BACKEND, _MODEL)
        except Exception:  # ImportError, or a weights download that failed
            pass

        try:
            import whisper

            _MODEL = whisper.load_model(_WHISPER_MODEL_SIZE)
            _BACKEND = "whisper"
            return (_BACKEND, _MODEL)
        except Exception:
            pass

        _BACKEND = "none"
        _MODEL = None
        return None


def reset_model_cache() -> None:
    """Drop the cached backend so a newly installed one is picked up. For tests."""
    global _BACKEND, _MODEL
    with _LOAD_LOCK:
        _BACKEND = None
        _MODEL = None


def _whisper_lang(lang: Lang | None) -> str | None:
    if lang is None or lang in _WHISPER_UNSUPPORTED:
        return None
    return lang.value


def _run_faster_whisper(model: Any, audio: bytes, lang: Lang | None) -> dict[str, Any]:
    # faster-whisper decodes a file-like object via PyAV, so no temp file and no ffmpeg.
    segments, info = model.transcribe(
        io.BytesIO(audio), language=_whisper_lang(lang), beam_size=_WHISPER_BEAM_SIZE
    )
    # `segments` is a generator: nothing is actually decoded until it is drained.
    text = "".join(seg.text for seg in segments).strip()
    return {
        "text": text,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
    }


def _run_openai_whisper(model: Any, audio: bytes, mime_type: str, lang: Lang | None) -> dict[str, Any]:
    # openai-whisper only accepts a path or a decoded array; it shells out to ffmpeg.
    suffix = os.path.splitext(filename_for(mime_type))[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(audio)
        try:
            result = model.transcribe(path, language=_whisper_lang(lang), fp16=False)
        except ValueError:
            # Unsupported language code slipped through -- retry with auto-detection.
            result = model.transcribe(path, fp16=False)
    finally:
        os.unlink(path)
    return {
        "text": str(result.get("text", "")).strip(),
        "language": result.get("language"),
        "language_probability": None,  # openai-whisper reports no utterance confidence
        "duration": None,
    }


def _clean_llm_transcript(raw: str) -> str:
    """Strip the preamble an instruction-tuned model adds despite being told not to.

    Only removes a leading line that ends in a colon (e.g. "Here is the transcription:")
    and surrounding quotes. Anything more aggressive risks eating real transcript text.
    """
    text = raw.strip()
    if "\n" in text:
        head, _, tail = text.partition("\n")
        if head.rstrip().endswith(":") and tail.strip():
            text = tail.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


async def _ollama_transcribe(
    audio: bytes, mime_type: str, lang_hint: Lang | None, started: float
) -> Transcript | None:
    """Last resort before giving up: the local multimodal generator does the ASR.

    Ollama's field name for non-image modalities has moved between releases, so both the
    audio-specific and the legacy generic field are attempted before declaring failure.
    """
    encoded = base64.b64encode(audio).decode("ascii")
    url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
    base_body: dict[str, Any] = {
        "model": settings.ollama_model,
        "prompt": _OLLAMA_PROMPT,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 512},
    }

    async with httpx.AsyncClient(timeout=settings.ollama_timeout_s) as client:
        for field in ("audio", "images"):
            try:
                resp = await client.post(url, json={**base_body, field: [encoded]})
            except Exception:
                return None  # server down; no point trying the second field shape
            if resp.status_code >= 400:
                continue  # wrong field name for this Ollama build -- try the other
            try:
                payload = resp.json()
            except ValueError:
                continue
            text = _clean_llm_transcript(str(payload.get("response") or ""))
            if not text:
                continue
            payload["_voxrag"] = {
                "attempts": 1,
                "retries": 0,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "backend": "ollama",
                "audio_field": field,
                "mime_type": mime_type,
            }
            return Transcript(
                text=text,
                lang=lang_hint,  # the model returns no language id; trust the caller's hint
                provider="local",
                raw=payload,
            )
    return None


class LocalSTT:
    """contracts.STTProvider that needs no API key. The fallback of last resort."""

    name: str = "local"

    async def transcribe(
        self, audio: bytes, *, mime_type: str = "audio/wav", lang_hint: Lang | None = None
    ) -> Transcript:
        started = time.perf_counter()
        if len(audio) < _MIN_AUDIO_BYTES:
            return Transcript(
                text="",
                lang=lang_hint,
                provider=self.name,
                raw={"detail": f"audio too short: {len(audio)} bytes", "_voxrag": {"attempts": 0}},
            )

        loaded = await asyncio.to_thread(_load_model)
        if loaded is not None:
            backend, model = loaded
            try:
                if backend == "faster-whisper":
                    out = await asyncio.to_thread(_run_faster_whisper, model, audio, lang_hint)
                else:
                    out = await asyncio.to_thread(
                        _run_openai_whisper, model, audio, mime_type, lang_hint
                    )
            except Exception as exc:
                # A decode failure here must not raise: this provider is already the
                # fallback, so raising would surface a stack trace instead of a refusal.
                return self._empty(
                    lang_hint, f"{backend} failed to decode {mime_type}: {exc!r}", started
                )

            prob = out.get("language_probability")
            return Transcript(
                text=str(out.get("text") or ""),
                lang=_parse_whisper_lang(out.get("language")) or lang_hint,
                confidence=(
                    min(1.0, max(0.0, float(prob))) if isinstance(prob, (int, float)) else None
                ),
                provider=self.name,
                duration_s=(
                    float(out["duration"])
                    if isinstance(out.get("duration"), (int, float))
                    else None
                ),
                raw={
                    **out,
                    "_voxrag": {
                        "attempts": 1,
                        "retries": 0,
                        "latency_ms": (time.perf_counter() - started) * 1000.0,
                        "backend": backend,
                    },
                },
            )

        via_ollama = await _ollama_transcribe(audio, mime_type, lang_hint, started)
        if via_ollama is not None:
            return via_ollama
        return self._empty(lang_hint, _INSTALL_HINT, started)

    def _empty(self, lang_hint: Lang | None, detail: str, started: float) -> Transcript:
        return Transcript(
            text="",
            lang=lang_hint,
            provider=self.name,
            raw={
                "detail": detail,
                "_voxrag": {
                    "attempts": 1,
                    "retries": 0,
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                    "backend": "none",
                },
            },
        )

    async def health(self) -> bool:
        """True if any backend could serve a request.

        Uses find_spec rather than _load_model so a health poll never triggers a multi
        hundred-megabyte weights download.
        """
        if await asyncio.to_thread(available_backend) is not None:
            return True
        return await _ollama_reachable()


def available_backend() -> str | None:
    """Name of the Whisper backend that is importable, without importing it."""
    if _BACKEND not in (None, "none"):
        return _BACKEND
    for module, label in (("faster_whisper", "faster-whisper"), ("whisper", "whisper")):
        if importlib.util.find_spec(module) is not None:
            return label
    return None


async def _ollama_reachable() -> bool:
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT_S) as client:
            resp = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
    except Exception:
        return False
    return resp.status_code < 400


_WHISPER_LANG_TO_LANG: Final[dict[str, Lang]] = {lang.value: lang for lang in Lang}


def _parse_whisper_lang(code: Any) -> Lang | None:
    """Whisper reports ISO-639-1; anything unrecognised means "keep the caller's hint"."""
    if not isinstance(code, str) or not code:
        return None
    return _WHISPER_LANG_TO_LANG.get(code.strip().lower().split("-", 1)[0])


__all__ = ["LocalSTT", "available_backend", "reset_model_cache"]
