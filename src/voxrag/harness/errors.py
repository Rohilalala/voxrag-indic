"""Typed failures for the harness.

Every error carries one bit the retry layer actually acts on: `recoverable`. A 429 and a
401 are both "the provider said no", but retrying the first is correct and retrying the
second burns the request budget to receive the identical rejection three times. Deciding
that at the raise site -- where the status code is still in hand -- is the only place it
can be decided correctly; by the time the exception reaches `with_retry` the HTTP response
is long gone.

Nothing here is caught-and-swallowed by the harness. These exist so the orchestrator can
tell *which* recovery applies: a StageTimeout degrades to the other arm, a GuardrailBlocked
becomes a refusal (never a retry), a ToolInputError is a programming error that will fail
identically forever.
"""

from __future__ import annotations

import asyncio
from typing import Any

from voxrag.contracts import GuardrailVerdict, RefusalReason, StageName

#: Statuses worth a second attempt. 408/425 are the server asking us to resend, 429 is
#: backpressure, 5xx is the server admitting it is at fault. Everything else in 4xx is a
#: statement about the request itself, which a retry cannot change.
_RETRYABLE_STATUS: frozenset[int] = frozenset({408, 425, 429})

#: Exceptions that mean "the world was briefly unavailable" rather than "the code is wrong".
#: asyncio.TimeoutError is TimeoutError on 3.11+, kept explicit for readers on older docs.
_TRANSIENT: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError,
    TimeoutError,
    ConnectionError,
    OSError,
)

#: Deterministic failures. Retrying one of these produces the same exception, more slowly.
#: pydantic's ValidationError subclasses ValueError, so schema mismatches land here too.
_DETERMINISTIC: tuple[type[BaseException], ...] = (
    TypeError,
    ValueError,
    KeyError,
    IndexError,
    AttributeError,
    ImportError,
    NotImplementedError,
)


class VoxRAGError(Exception):
    """Base class. `recoverable` is the contract with `harness.retry.with_retry`."""

    #: Class-level default; instances may override it via the constructor.
    recoverable: bool = False

    def __init__(self, message: str, *, recoverable: bool | None = None, **detail: Any) -> None:
        super().__init__(message)
        self.message = message
        if recoverable is not None:
            self.recoverable = recoverable
        self.detail: dict[str, Any] = detail

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form, for the trace and the API error envelope."""
        return {
            "error": type(self).__name__,
            "message": self.message,
            "recoverable": self.recoverable,
            **self.detail,
        }


class StageError(VoxRAGError):
    """A named pipeline stage failed. `cause` is the original exception, kept for triage."""

    def __init__(
        self,
        stage: StageName | str,
        cause: BaseException | None = None,
        message: str | None = None,
        *,
        recoverable: bool | None = None,
        **detail: Any,
    ) -> None:
        self.stage = stage
        self.cause = cause
        name = stage.value if isinstance(stage, StageName) else str(stage)
        text = message or (f"{name} failed: {cause!r}" if cause is not None else f"{name} failed")
        # Inherit the cause's classification: a stage that failed because a socket dropped is
        # as retryable as the socket, and a stage that failed on a TypeError is not.
        if recoverable is None:
            recoverable = True if cause is None else is_recoverable(cause)
        super().__init__(text, recoverable=recoverable, stage=name, **detail)


class StageTimeout(StageError):
    """A stage exceeded its slice of the request budget.

    Recoverable in principle -- the dependency may simply have been slow once -- but the
    caller decides whether a second attempt still fits inside the remaining budget.
    """

    def __init__(self, stage: StageName | str, budget_ms: float, **detail: Any) -> None:
        name = stage.value if isinstance(stage, StageName) else str(stage)
        self.budget_ms = budget_ms
        super().__init__(
            stage,
            None,
            f"{name} exceeded its {budget_ms:.0f}ms budget",
            recoverable=True,
            budget_ms=budget_ms,
            **detail,
        )


class ProviderError(VoxRAGError):
    """An external dependency (STT vendor, Ollama, Qdrant) returned an error.

    `status` is the HTTP status when there was one; None means the call never got that far
    (DNS, connection refused, TLS), which is exactly the transient case worth retrying.
    """

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status: int | None = None,
        recoverable: bool | None = None,
        **detail: Any,
    ) -> None:
        self.provider = provider
        self.status = status
        if recoverable is None:
            recoverable = status is None or status in _RETRYABLE_STATUS or status >= 500
        super().__init__(
            f"{provider}: {message}",
            recoverable=recoverable,
            provider=provider,
            status=status,
            **detail,
        )


class CircuitOpenError(ProviderError):
    """The breaker for a provider is open, so the call was never attempted.

    Deliberately *not* recoverable: the whole point of an open breaker is that further
    attempts are known-futile until the reset window elapses. The caller should fail over
    to another provider or degrade, not retry into a wall.
    """

    def __init__(self, provider: str, *, open_for_s: float | None = None) -> None:
        super().__init__(
            provider,
            "circuit open, call not attempted",
            status=None,
            recoverable=False,
            open_for_s=open_for_s,
        )


class GuardrailBlocked(VoxRAGError):
    """A rail refused the request. Carries the verdict so the refusal can cite which one."""

    def __init__(self, verdict: GuardrailVerdict) -> None:
        self.verdict = verdict
        super().__init__(
            f"blocked by {verdict.name}: {verdict.reason.value if verdict.reason else 'unspecified'}",
            recoverable=False,  # a rail decision is deterministic; retrying is a policy bypass
            rail=verdict.name,
            reason=verdict.reason.value if verdict.reason else None,
            detail_text=verdict.detail,
        )

    @property
    def reason(self) -> RefusalReason:
        return self.verdict.reason or RefusalReason.LOW_CONFIDENCE


class ToolError(VoxRAGError):
    """Base for the tool-call layer. None of these are retryable: they are contract bugs."""

    def __init__(self, tool: str, message: str, **detail: Any) -> None:
        self.tool = tool
        super().__init__(f"tool {tool!r}: {message}", recoverable=False, tool=tool, **detail)


class ToolNotFoundError(ToolError):
    def __init__(self, tool: str, available: list[str]) -> None:
        super().__init__(tool, f"not registered; available: {available}", available=available)


class ToolInputError(ToolError):
    """Payload failed validation against the tool's declared input model."""

    def __init__(self, tool: str, message: str, **detail: Any) -> None:
        super().__init__(tool, f"invalid input -- {message}", **detail)


class ToolOutputError(ToolError):
    """Handler returned something that is not the tool's declared output model.

    Raised rather than coerced: a handler that returns the wrong shape has a bug, and
    silently reshaping it would push the failure into whichever caller reads the field
    that is missing.
    """

    def __init__(self, tool: str, message: str, **detail: Any) -> None:
        super().__init__(tool, f"invalid output -- {message}", **detail)


def is_recoverable(exc: BaseException) -> bool:
    """Should `with_retry` spend another attempt on this?

    Unknown exception types default to True. That is deliberate: attempts are bounded and
    each one runs inside a stage budget, so the cost of retrying a genuine bug is one extra
    fast failure, while the cost of *not* retrying an unrecognised transport error is a
    refusal the user did not need to see.
    """
    if isinstance(exc, VoxRAGError):
        return exc.recoverable
    if isinstance(exc, _DETERMINISTIC):
        return False
    if isinstance(exc, _TRANSIENT):
        return True
    return True


__all__ = [
    "CircuitOpenError",
    "GuardrailBlocked",
    "ProviderError",
    "StageError",
    "StageTimeout",
    "ToolError",
    "ToolInputError",
    "ToolNotFoundError",
    "ToolOutputError",
    "VoxRAGError",
    "is_recoverable",
]
