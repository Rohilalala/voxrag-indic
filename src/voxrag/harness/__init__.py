"""The harness: structured orchestration around the model, not a raw prompt call.

    from voxrag.harness import get_harness

    harness = get_harness()
    await harness.warmup()                         # pay the cold starts up front
    result = await harness.run(audio=wav, mime="audio/wav")
    result.core_ms                                 # the number the 200ms budget is graded on

Four pieces:

* `orchestrator.Harness` -- the ten-stage pipeline, every stage timed into a
  `telemetry.Trace`, bounded by `settings.stage_timeout_ms`, and given a degraded path so a
  dead dependency costs answer quality instead of the request.
* `tools` -- the capabilities (`retrieve`, `answer`, `clarify`, `refuse`) behind pydantic
  input and output models, dispatched by `call_tool` and describable to a function-calling
  model via `tool_schemas()`.
* `retry` -- bounded retries with full-jitter backoff, per-stage deadlines, and a circuit
  breaker with a real single-probe half-open state.
* `errors` -- a typed hierarchy whose `recoverable` flag is what lets the retry layer tell a
  429 from a 401.
"""

from __future__ import annotations

from voxrag.harness.errors import (
    CircuitOpenError,
    GuardrailBlocked,
    ProviderError,
    StageError,
    StageTimeout,
    ToolError,
    ToolInputError,
    ToolNotFoundError,
    ToolOutputError,
    VoxRAGError,
    is_recoverable,
)
from voxrag.harness.orchestrator import (
    EventCallback,
    Harness,
    detect_lang,
    get_harness,
)
from voxrag.harness.retry import (
    CircuitBreaker,
    breaker_states,
    guarded,
    stage_budget_ms,
    with_retry,
    with_timeout,
)
from voxrag.harness.tools import (
    TOOLS,
    AnswerInput,
    ClarifyInput,
    RefuseInput,
    RetrievalResult,
    RetrieveInput,
    Tool,
    call_tool,
    tool_schemas,
)

__all__ = [
    "TOOLS",
    "AnswerInput",
    "CircuitBreaker",
    "CircuitOpenError",
    "ClarifyInput",
    "EventCallback",
    "GuardrailBlocked",
    "Harness",
    "ProviderError",
    "RefuseInput",
    "RetrievalResult",
    "RetrieveInput",
    "StageError",
    "StageTimeout",
    "Tool",
    "ToolError",
    "ToolInputError",
    "ToolNotFoundError",
    "ToolOutputError",
    "VoxRAGError",
    "breaker_states",
    "call_tool",
    "detect_lang",
    "get_harness",
    "guarded",
    "is_recoverable",
    "stage_budget_ms",
    "tool_schemas",
    "with_retry",
    "with_timeout",
]
