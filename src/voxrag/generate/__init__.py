"""Answer generation: two generators, one shared fallback policy.

`ExtractiveGenerator` (FAST) is verbatim sentence extraction -- microseconds, grounded by
construction, and the only mode that fits the 200ms budget. `OllamaGenerator` (QUALITY) is
gemma4:e4b over /api/chat, reported with its own latency profile.

The rule that matters for the demo lives in `generate_with_fallback`: **QUALITY degrades to
FAST, it never fails.** A demo that dies because a 9.6GB model is not running on the machine
in front of the user is a failed demo, so a dead or slow Ollama produces a real, cited,
extractive answer plus a recorded degradation -- and the trace says so honestly, because the
returned Answer carries `mode=FAST`.

Repeated failures trip a circuit breaker (`settings.circuit_breaker_threshold` /
`circuit_breaker_reset_s`) so that once the daemon is known to be down, subsequent requests
go straight to the extractive path instead of each paying a connect timeout first.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence

from voxrag.config import settings
from voxrag.contracts import Answer, GenerationMode, Generator, Lang, ScoredChunk
from voxrag.generate.extractive import ExtractiveGenerator
from voxrag.generate.ollama_gemma import GenerationError, OllamaGenerator
from voxrag.generate.prompts import build_messages, system_prompt
from voxrag.telemetry import Trace

#: Fallback ceiling for the QUALITY stage when `settings.stage_timeout_ms` has no entry.
_DEFAULT_GENERATE_TIMEOUT_MS = 25_000.0


class _CircuitBreaker:
    """Consecutive-failure breaker with a half-open probe after the reset window.

    Half-open rather than fully closed on reset: exactly one request is allowed through to
    test the daemon, and it re-opens on the next failure. Closing outright would send the
    whole burst back into a still-dead dependency.
    """

    __slots__ = ("_failures", "_opened_at", "_reset_s", "_threshold")

    def __init__(self, threshold: int, reset_s: float) -> None:
        self._threshold = max(1, threshold)
        self._reset_s = reset_s
        self._failures = 0
        self._opened_at = 0.0

    @property
    def open(self) -> bool:
        if self._failures < self._threshold:
            return False
        if time.monotonic() - self._opened_at >= self._reset_s:
            self._failures = self._threshold - 1  # half-open: let one probe through
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = time.monotonic()


_EXTRACTIVE = ExtractiveGenerator()
_OLLAMA: OllamaGenerator | None = None
_BREAKER = _CircuitBreaker(settings.circuit_breaker_threshold, settings.circuit_breaker_reset_s)


def get_generator(mode: GenerationMode | str | None = None) -> Generator:
    """Resolve a mode to its generator. `None` means `settings.default_mode`.

    Both generators are process-wide singletons: the extractive one so its tokeniser cache is
    shared across requests, the Ollama one so its keep-alive connection pool is.
    """
    global _OLLAMA
    if mode is None:
        mode = settings.default_mode
    if not isinstance(mode, GenerationMode):
        try:
            mode = GenerationMode(str(mode))
        except ValueError as exc:
            known = ", ".join(m.value for m in GenerationMode)
            raise KeyError(f"unknown generation mode {mode!r}; known modes: {known}") from exc
    if mode is GenerationMode.FAST:
        return _EXTRACTIVE
    if _OLLAMA is None:
        _OLLAMA = OllamaGenerator()
    return _OLLAMA


def _generate_timeout_s() -> float:
    return settings.stage_timeout_ms.get("generate", _DEFAULT_GENERATE_TIMEOUT_MS) / 1000.0


async def generate_with_fallback(
    query: str,
    context: Sequence[ScoredChunk],
    *,
    lang: Lang | None = None,
    mode: GenerationMode | str | None = None,
    trace: Trace | None = None,
) -> Answer:
    """Generate in `mode`, degrading QUALITY -> FAST rather than failing the request.

    The degradation is recorded on `trace` when one is supplied, so the analytics report
    shows how often the quality path was unavailable instead of silently reporting FAST
    latencies as if they were the configured mode.
    """
    chunks = list(context)
    resolved = get_generator(mode)
    if resolved.mode is GenerationMode.FAST:
        return await resolved.generate(query, chunks, lang=lang)

    if _BREAKER.open:
        if trace is not None:
            trace.degrade(f"generate: {resolved.name} circuit open, served extractive")
        return await _EXTRACTIVE.generate(query, chunks, lang=lang)

    try:
        answer = await asyncio.wait_for(
            resolved.generate(query, chunks, lang=lang), timeout=_generate_timeout_s()
        )
        if not answer.text.strip():
            raise GenerationError("empty answer text")
        _BREAKER.record_success()
        return answer
    except (GenerationError, TimeoutError, OSError) as exc:
        _BREAKER.record_failure()
        if trace is not None:
            trace.degrade(f"generate: {resolved.name} failed ({type(exc).__name__}: {exc})")
        return await _EXTRACTIVE.generate(query, chunks, lang=lang)


async def stream_with_fallback(
    query: str,
    context: Sequence[ScoredChunk],
    *,
    lang: Lang | None = None,
    mode: GenerationMode | str | None = None,
    trace: Trace | None = None,
) -> AsyncIterator[str]:
    """Token deltas for the websocket route, with the same never-fail policy.

    A failure *before* the first token falls back to the extractive answer as a single
    delta. After the first token the answer prefix is already on the client's screen, so a
    silent switch would splice two different answers together; the stream ends there and the
    degradation is recorded instead.
    """
    chunks = list(context)
    resolved = get_generator(mode)
    if resolved.mode is GenerationMode.QUALITY and not _BREAKER.open:
        emitted = False
        try:
            async for piece in resolved.stream(query, chunks, lang):  # type: ignore[attr-defined]
                emitted = True
                yield piece
            _BREAKER.record_success()
            return
        except (GenerationError, OSError, TimeoutError) as exc:
            _BREAKER.record_failure()
            if trace is not None:
                trace.degrade(f"generate: {resolved.name} stream failed ({type(exc).__name__})")
            if emitted:
                return
    elif resolved.mode is GenerationMode.QUALITY and trace is not None:
        trace.degrade(f"generate: {resolved.name} circuit open, streamed extractive")

    answer = await _EXTRACTIVE.generate(query, chunks, lang=lang)
    yield answer.text


async def warmup() -> dict[str, bool]:
    """Pay the one-off costs before the first real request.

    Importing this package has already loaded the encoder tokeniser (via
    `voxrag.chunking.base`), which is the extractive path's only startup cost. What is left
    is the QUALITY probe: doing it here means the first user question does not also pay for
    discovering that Ollama is down.
    """
    quality = get_generator(GenerationMode.QUALITY)
    try:
        healthy = await quality.health()
    except Exception:  # a warmup probe must never take the process down
        healthy = False
    if not healthy:
        _BREAKER.record_failure()
    return {"fast": True, "quality": healthy}


async def warm_verifier() -> bool:
    """Load the model `guardrails.cross_language` verifies with. Returns whether it is ready.

    Separate from `warmup` above because it is a different kind of cost: `warmup` runs a
    /api/tags probe in milliseconds, this waits for a ~6s model load. It is skipped entirely
    when cross-language verification is off, since then no request ever calls the verifier.
    """
    if not settings.cross_language_verification:
        return False
    quality = get_generator(GenerationMode.QUALITY)
    warm = getattr(quality, "warm", None)
    if warm is None:  # the extractive fallback: nothing to load, nothing to verify with
        return False
    try:
        ready: bool = await warm()
    except Exception:  # a warmup probe must never take the process down
        return False
    return ready


async def aclose() -> None:
    """Release the QUALITY generator's connection pool. Called from the API shutdown hook."""
    if _OLLAMA is not None:
        await _OLLAMA.aclose()


__all__ = [
    "ExtractiveGenerator",
    "GenerationError",
    "OllamaGenerator",
    "aclose",
    "build_messages",
    "generate_with_fallback",
    "get_generator",
    "stream_with_fallback",
    "system_prompt",
    "warm_verifier",
    "warmup",
]
