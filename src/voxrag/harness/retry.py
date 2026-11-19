"""Retries, deadlines and circuit breaking -- the three things that keep one sick
dependency from becoming a sick service.

Three deliberate choices:

* **Full jitter, not fixed backoff.** `sleep(base * 2**n)` makes every client that failed
  at the same moment retry at the same moment, so a dependency that is recovering gets hit
  by the entire fleet at once and falls over again. Sleeping a uniform random draw from
  `[0, base * 2**n]` spreads the herd across the window, which is measurably the best of
  the common strategies (AWS's "Exponential Backoff and Jitter"). It also means two runs of
  the same failing request take different amounts of time -- expected, not a bug.
* **Factories, not coroutines.** `with_retry` takes a zero-argument callable. A coroutine
  object can only be awaited once, so a retry loop over a coroutine raises
  "cannot reuse already awaited coroutine" on attempt two, which is the classic version of
  this bug.
* **Timeouts are per stage, not per request.** One hung dependency should cost its own
  slice of the 200ms budget and nothing more, so the stage that follows it still gets to run.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any, Final, TypeVar

from voxrag.config import settings
from voxrag.contracts import StageName
from voxrag.harness.errors import (
    CircuitOpenError,
    StageError,
    StageTimeout,
    VoxRAGError,
    is_recoverable,
)
from voxrag.telemetry import Trace

_LOG = logging.getLogger(__name__)

T = TypeVar("T")

#: Anything the helpers accept as "work to run": an awaitable, or a factory producing one.
Work = Callable[[], Awaitable[T]] | Awaitable[T]

#: Ceiling on a single backoff sleep. Without it, `base * 2**n` grows past any request
#: budget within a handful of attempts and the retry becomes an outage of its own.
MAX_BACKOFF_MS: Final[float] = 2_000.0

#: Used when a stage has no entry in `settings.stage_timeout_ms`. Generous on purpose: an
#: unlisted stage is one nobody has profiled, and guessing tight would break it.
DEFAULT_STAGE_BUDGET_MS: Final[float] = 1_000.0


def stage_budget_ms(stage: StageName | str, default: float | None = None) -> float:
    """The configured wall-clock ceiling for `stage`, in milliseconds."""
    key = stage.value if isinstance(stage, StageName) else str(stage)
    return settings.stage_timeout_ms.get(key, DEFAULT_STAGE_BUDGET_MS if default is None else default)


async def _run(work: Work[T]) -> T:
    """Await either a factory's result or a bare awaitable."""
    result = work() if callable(work) else work
    if not inspect.isawaitable(result):  # a sync callable is still usable work
        return result
    return await result


def backoff_delay_s(attempt: int, base_ms: float, jitter: bool = True) -> float:
    """Seconds to sleep before attempt `attempt + 1` (0-based). Full jitter by default."""
    window = min(base_ms * (2.0**attempt), MAX_BACKOFF_MS)
    if window <= 0.0:
        return 0.0
    return (random.uniform(0.0, window) if jitter else window) / 1000.0


async def with_retry(
    fn: Work[T],
    *,
    attempts: int | None = None,
    backoff_ms: float | None = None,
    jitter: bool = True,
    recoverable_only: bool = True,
    trace: Trace | None = None,
    stage: StageName | str | None = None,
) -> T:
    """Run `fn`, retrying recoverable failures with exponentially backed-off full jitter.

    `attempts` counts total tries, not extra ones: `attempts=1` disables retrying. Each
    retry increments `trace.retries`, which is what the analytics report sums -- a retry
    that nobody counted is a latency spike nobody can explain.

    Raises the *last* failure. Non-VoxRAG exceptions are wrapped in `StageError` when a
    `stage` is given, so callers upstack keep the stage attribution instead of receiving a
    bare `httpx.ConnectError` with no idea which arm produced it.
    """
    total = max(1, settings.max_retries + 1 if attempts is None else attempts)
    base = settings.retry_backoff_ms if backoff_ms is None else backoff_ms

    for attempt in range(total):
        try:
            return await _run(fn)
        except asyncio.CancelledError:
            # Cancellation is our own deadline firing, not a dependency failing. Retrying
            # it would keep working past a budget the caller has already given up on.
            raise
        except Exception as exc:
            last_attempt = attempt + 1 >= total
            if last_attempt or (recoverable_only and not is_recoverable(exc)):
                raise _attributed(exc, stage) from exc
            if trace is not None:
                trace.retries += 1
            delay = backoff_delay_s(attempt, base, jitter)
            _LOG.debug(
                "retry %d/%d for %s after %r (sleeping %.0fms)",
                attempt + 1,
                total - 1,
                stage or "call",
                exc,
                delay * 1000.0,
            )
            if delay > 0.0:
                await asyncio.sleep(delay)

    raise AssertionError("unreachable: the loop either returns or raises")  # pragma: no cover


def _attributed(exc: Exception, stage: StageName | str | None) -> Exception:
    if stage is None or isinstance(exc, VoxRAGError):
        return exc
    return StageError(stage, exc)


async def with_timeout(coro: Work[T], ms: float, stage: StageName | str) -> T:
    """Bound `coro` to `ms` milliseconds, raising `StageTimeout` when it overruns.

    A non-positive `ms` means "no ceiling" -- used by stages whose budget is deliberately
    unset (the QUALITY generator, whose own client owns the deadline).

    Note that `asyncio.wait_for` cancels the *await*, not any worker thread behind an
    `asyncio.to_thread` call: the orphaned thread finishes into nothing. That is the right
    trade against a hard budget, but it means a stage that times out repeatedly will leak
    threads until its work completes, which is why every to_thread payload here is bounded
    work (one ONNX forward pass, one BM25 scan) rather than an open-ended loop.
    """
    if ms <= 0:
        return await _run(coro)
    try:
        return await asyncio.wait_for(_run(coro), timeout=ms / 1000.0)
    except TimeoutError as exc:
        raise StageTimeout(stage, ms) from exc


class CircuitBreaker:
    """Consecutive-failure breaker with a real half-open state.

    closed -> (threshold consecutive failures) -> open
    open   -> (reset_s elapsed)                -> half_open, exactly one probe admitted
    half_open -> probe succeeds                -> closed
              -> probe fails                   -> open, timer restarted

    The single-probe half-open state is the part that matters under load: if every caller
    were admitted the instant `reset_s` elapsed, a dependency that is still down would take
    the full concurrent request volume as its welcome-back present, and the breaker would
    have bought nothing.

    Thread-safe -- the API server, the benchmark and `to_thread` workers can all reach one
    breaker, and a torn counter either trips early or never trips at all.
    """

    def __init__(
        self,
        name: str,
        *,
        threshold: int | None = None,
        reset_s: float | None = None,
    ) -> None:
        self.name = name
        self.threshold = max(1, threshold if threshold is not None else settings.circuit_breaker_threshold)
        self.reset_s = reset_s if reset_s is not None else settings.circuit_breaker_reset_s
        self._lock = threading.Lock()
        self._failures = 0
        self._state = "closed"
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def state(self) -> str:
        """"closed" | "open" | "half_open". Reads the clock, so an expired open reports open
        until someone calls `allow()` -- state is only advanced by an actual admission."""
        with self._lock:
            return self._state

    def allow(self) -> bool:
        with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                if self._opened_at is None or time.monotonic() - self._opened_at < self.reset_s:
                    return False
                self._state = "half_open"
                self._probe_in_flight = True
                return True
            # half_open: one probe at a time, everyone else keeps failing fast.
            if self._probe_in_flight:
                return False
            self._probe_in_flight = True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "closed"
            self._opened_at = None
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._probe_in_flight = False
            # A failed probe re-opens immediately; it is direct evidence, not a data point.
            if self._state == "half_open" or self._failures >= self.threshold:
                self._state = "open"
                self._opened_at = time.monotonic()

    def reset(self) -> None:
        """Force closed. For tests and for a manual recovery poke from /health."""
        self.record_success()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            open_for = None if self._opened_at is None else time.monotonic() - self._opened_at
            return {
                "name": self.name,
                "state": self._state,
                "consecutive_failures": self._failures,
                "threshold": self.threshold,
                "open_for_s": open_for,
                "reset_s": self.reset_s,
            }

    async def call(self, fn: Work[T]) -> T:
        """Run `fn` under the breaker, raising `CircuitOpenError` instead of calling when open."""
        if not self.allow():
            snap = self.snapshot()
            raise CircuitOpenError(self.name, open_for_s=snap["open_for_s"])
        try:
            result = await _run(fn)
        except asyncio.CancelledError:
            # Our own deadline, not the provider's fault: recording it as a failure would
            # let an overloaded *caller* trip a perfectly healthy dependency.
            raise
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result


_BREAKERS: dict[str, CircuitBreaker] = {}
_BREAKER_LOCK = threading.Lock()


def get_breaker(name: str) -> CircuitBreaker:
    """Process-wide breaker for `name`. Per-request breakers would never trip."""
    breaker = _BREAKERS.get(name)
    if breaker is not None:
        return breaker
    with _BREAKER_LOCK:
        breaker = _BREAKERS.get(name)
        if breaker is None:
            breaker = CircuitBreaker(name)
            _BREAKERS[name] = breaker
    return breaker


def breaker_states() -> list[dict[str, Any]]:
    """Snapshot of every harness-owned breaker, for /health."""
    return [b.snapshot() for b in list(_BREAKERS.values())]


async def guarded(
    fn: Work[T],
    *,
    stage: StageName | str,
    budget_ms: float | None = None,
    attempts: int = 1,
    backoff_ms: float | None = None,
    fallback: T,
    trace: Trace | None = None,
    note: str | None = None,
) -> T:
    """Timeout + retry + degrade, the combination every optional stage needs.

    Returns `fallback` instead of raising, recording a degradation on the trace. This is
    the harness's default failure posture: a pipeline that returns a slightly worse answer
    beats one that returns a stack trace, and the trace states exactly what was lost.
    """
    budget = stage_budget_ms(stage) if budget_ms is None else budget_ms
    try:
        return await with_retry(
            lambda: with_timeout(fn, budget, stage),
            attempts=attempts,
            backoff_ms=backoff_ms,
            trace=trace,
            stage=stage,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        name = stage.value if isinstance(stage, StageName) else str(stage)
        message = note or f"{name} failed ({type(exc).__name__}: {exc}); degraded"
        _LOG.warning("%s", message)
        if trace is not None:
            trace.degrade(message)
        return fallback


__all__ = [
    "DEFAULT_STAGE_BUDGET_MS",
    "MAX_BACKOFF_MS",
    "CircuitBreaker",
    "backoff_delay_s",
    "breaker_states",
    "get_breaker",
    "guarded",
    "stage_budget_ms",
    "with_retry",
    "with_timeout",
]
