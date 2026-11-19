"""Speech-to-text providers and the failover policy that sits in front of them.

Selection is `settings.stt_provider` with `settings.stt_fallback_provider` behind it.
Two things make the failover cheap rather than merely present:

  * a per-provider circuit breaker, so once a vendor is confirmed dead every subsequent
    request skips straight to the fallback instead of re-paying a full stt_timeout_s to
    rediscover the same outage;
  * an outer wall-clock ceiling per hop, because a provider's own internal retry budget
    (attempts x stt_timeout_s + backoff) is longer than anyone should wait before the
    keyless local backend is tried.

Nothing in here raises on a provider failure. A dead STT stage degrades to an empty
Transcript, which the EMPTY_TRANSCRIPT guardrail turns into a clean refusal -- a refusal
is a correct answer, a stack trace is not.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Final

from voxrag.config import settings
from voxrag.contracts import Lang, StageName, STTProvider, Transcript
from voxrag.stt.elevenlabs import ElevenLabsSTT
from voxrag.stt.local_whisper import LocalSTT
from voxrag.stt.sarvam import SARVAM_LANG, SarvamSTT, parse_sarvam_lang, to_sarvam_code
from voxrag.telemetry import Trace

#: Canonical name -> factory. Aliases are folded in `_canonical`.
_FACTORIES: Final[dict[str, Any]] = {
    "sarvam": SarvamSTT,
    "elevenlabs": ElevenLabsSTT,
    "local": LocalSTT,
}

_ALIASES: Final[dict[str, str]] = {
    "sarvam": "sarvam",
    "sarvam-ai": "sarvam",
    "saaras": "sarvam",
    "elevenlabs": "elevenlabs",
    "eleven": "elevenlabs",
    "11labs": "elevenlabs",
    "scribe": "elevenlabs",
    "local": "local",
    "whisper": "local",
    "local_whisper": "local",
    "faster-whisper": "local",
}

#: The local backend may download several hundred MB of Whisper weights on its very first
#: call. Killing that at stt_timeout_s would mean the fallback can never become usable, so
#: it gets its own generous ceiling; every later call returns in seconds.
_LOCAL_BUDGET_S: Final[float] = 180.0
#: Slack over the provider's own retry budget, covering backoff sleeps and TLS setup.
_HOP_SLACK_S: Final[float] = 2.0


class CircuitBreaker:
    """Consecutive-failure breaker: closed -> open for `reset_s` -> one trial request.

    Thread-safe because the API server, the benchmark and any to_thread worker can all
    reach it, and a torn counter would either trip early or never trip at all.
    """

    def __init__(self, name: str, threshold: int | None = None, reset_s: float | None = None):
        self.name = name
        self.threshold = threshold if threshold is not None else settings.circuit_breaker_threshold
        self.reset_s = reset_s if reset_s is not None else settings.circuit_breaker_reset_s
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None

    def allow(self) -> bool:
        """False while the breaker is open. Lets exactly one trial through after reset_s."""
        with self._lock:
            if self._opened_at is None:
                return True
            if time.monotonic() - self._opened_at >= self.reset_s:
                # Half-open: clear the timer but keep the failure count, so a single
                # further failure re-opens immediately instead of needing `threshold` more.
                self._opened_at = None
                self._failures = self.threshold - 1
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.threshold and self._opened_at is None:
                self._opened_at = time.monotonic()

    def state(self) -> dict[str, Any]:
        with self._lock:
            open_for = None if self._opened_at is None else time.monotonic() - self._opened_at
            return {
                "provider": self.name,
                "open": self._opened_at is not None,
                "consecutive_failures": self._failures,
                "threshold": self.threshold,
                "open_for_s": open_for,
                "reset_s": self.reset_s,
            }


_registry_lock = threading.Lock()
_instances: dict[str, STTProvider] = {}
_breakers: dict[str, CircuitBreaker] = {}


def _canonical(name: str | None) -> str:
    key = (name or settings.stt_provider or "sarvam").strip().lower()
    resolved = _ALIASES.get(key)
    if resolved is None:
        raise ValueError(
            f"unknown STT provider {name!r}; valid names: {sorted(set(_ALIASES.values()))}"
        )
    return resolved


def get_stt(name: str | None = None) -> STTProvider:
    """Return the named provider, or `settings.stt_provider` when name is None.

    Instances are cached: a provider owns a pooled httpx client and, for `local`, a cache
    of loaded model weights. Rebuilding one per request would throw both away.
    """
    key = _canonical(name)
    provider = _instances.get(key)
    if provider is None:
        with _registry_lock:
            provider = _instances.get(key)  # another thread may have built it while we waited
            if provider is None:
                provider = _FACTORIES[key]()
                _instances[key] = provider
    return provider


def get_breaker(name: str | None = None) -> CircuitBreaker:
    key = _canonical(name)
    breaker = _breakers.get(key)
    if breaker is None:
        with _registry_lock:
            breaker = _breakers.get(key)
            if breaker is None:
                breaker = CircuitBreaker(key)
                _breakers[key] = breaker
    return breaker


def breaker_states() -> list[dict[str, Any]]:
    """Breaker snapshot for the health/analytics endpoint."""
    return [b.state() for b in list(_breakers.values())]


def reset_breakers() -> None:
    """Close every breaker. For tests and for a manual recovery poke from /health."""
    for breaker in list(_breakers.values()):
        breaker.record_success()


def _hop_budget_s(provider: str) -> float:
    """Wall-clock ceiling for one provider attempt, including its internal retries."""
    if provider == "local":
        return _LOCAL_BUDGET_S
    return settings.stt_timeout_s * (settings.max_retries + 1) + _HOP_SLACK_S


def _provider_chain(primary: str | None) -> list[str]:
    chain = [_canonical(primary)]
    fallback = settings.stt_fallback_provider
    if fallback:
        try:
            resolved = _canonical(fallback)
        except ValueError:
            resolved = None  # a typo'd fallback must not break the primary path
        if resolved is not None and resolved not in chain:
            chain.append(resolved)
    return chain


async def transcribe_with_fallback(
    audio: bytes,
    mime_type: str = "audio/wav",
    lang_hint: Lang | None = None,
    trace: Trace | None = None,
    *,
    provider: str | None = None,
    record_stage: bool = True,
) -> Transcript:
    """Transcribe via the configured provider, degrading to the fallback on failure.

    `record_stage` opens the StageName.STT span itself. That default matters: Trace.finish
    derives CORE (the number the 200ms budget is graded on) as TOTAL minus the STT span,
    so if nobody records STT then ~2s of vendor network time lands inside the budget. A
    caller that already wraps this call in its own STT span should pass False.
    """
    chain = _provider_chain(provider)
    # Annotated because the two arms are structurally identical but nominally unrelated;
    # both yield the detail dict that ends up in StageTiming.detail.
    span: AbstractContextManager[dict[str, Any]] = (
        trace.stage(StageName.STT) if (trace is not None and record_stage) else nullcontext({})
    )

    with span as detail:
        detail["chain"] = chain
        attempted: list[str] = []
        last_error: str = "no provider attempted"

        for hop, name in enumerate(chain):
            breaker = get_breaker(name)
            if not breaker.allow():
                last_error = f"{name}: circuit open"
                _degrade(trace, f"stt:{name}:circuit_open")
                continue

            attempted.append(name)
            try:
                transcript = await asyncio.wait_for(
                    get_stt(name).transcribe(audio, mime_type=mime_type, lang_hint=lang_hint),
                    timeout=_hop_budget_s(name),
                )
            except TimeoutError:  # asyncio.TimeoutError is an alias of this since 3.11
                breaker.record_failure()
                last_error = f"{name}: exceeded {_hop_budget_s(name):.1f}s hop budget"
                _degrade(trace, f"stt:{name}:timeout")
                _count_retry(trace)
                continue
            except Exception as exc:  # auth errors, API errors, transport, decode
                breaker.record_failure()
                last_error = f"{name}: {exc!r}"
                _degrade(trace, f"stt:{name}:{type(exc).__name__}")
                _count_retry(trace)
                continue

            breaker.record_success()
            transcript.from_fallback = hop > 0
            if transcript.from_fallback:
                _degrade(trace, f"stt:served_by_fallback:{name}")

            meta = transcript.raw.get("_voxrag") if isinstance(transcript.raw, dict) else None
            if isinstance(meta, dict):
                _count_retry(trace, int(meta.get("retries") or 0))

            detail.update(
                provider=transcript.provider,
                from_fallback=transcript.from_fallback,
                attempted=attempted,
                chars=len(transcript.text),
                lang=transcript.lang.value if transcript.lang else None,
            )
            return transcript

        # Everything failed. Return rather than raise: an empty transcript routes into the
        # EMPTY_TRANSCRIPT guardrail and the user gets a refusal instead of an HTTP 500.
        detail.update(provider="none", from_fallback=len(chain) > 1, attempted=attempted,
                      chars=0, error=last_error)
        _degrade(trace, f"stt:all_providers_failed:{last_error}")
        return Transcript(
            text="",
            lang=lang_hint,
            provider="none",
            from_fallback=len(chain) > 1,
            raw={"detail": f"all STT providers failed -- {last_error}", "attempted": attempted},
        )


def _degrade(trace: Trace | None, note: str) -> None:
    if trace is not None:
        trace.degrade(note)


def _count_retry(trace: Trace | None, n: int = 1) -> None:
    if trace is not None and n:
        trace.retries += n


async def warmup() -> dict[str, bool]:
    """Probe the whole chain concurrently at startup.

    Worth doing at boot rather than on the first request: it resolves DNS, completes the
    TLS handshake into the keep-alive pool, and answers "is the fallback actually usable"
    before a user is waiting on it.
    """
    chain = _provider_chain(None)
    results = await asyncio.gather(
        *(get_stt(name).health() for name in chain), return_exceptions=True
    )
    return {
        name: (result is True) for name, result in zip(chain, results, strict=False)
    }


async def aclose_all() -> None:
    """Close every pooled HTTP client. Call from the server's shutdown hook."""
    from voxrag.stt import elevenlabs as _elevenlabs
    from voxrag.stt import sarvam as _sarvam

    await asyncio.gather(_sarvam.aclose(), _elevenlabs.aclose(), return_exceptions=True)


__all__ = [
    "SARVAM_LANG",
    "CircuitBreaker",
    "ElevenLabsSTT",
    "LocalSTT",
    "SarvamSTT",
    "aclose_all",
    "breaker_states",
    "get_breaker",
    "get_stt",
    "parse_sarvam_lang",
    "reset_breakers",
    "to_sarvam_code",
    "transcribe_with_fallback",
    "warmup",
]
