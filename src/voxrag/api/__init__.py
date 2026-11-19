"""HTTP surface for the harness.

    from voxrag.api import create_app
    uvicorn voxrag.api.app:app            # module-level instance, see app.py
    uvicorn voxrag.api.app:create_app --factory

This package module holds the two things *both* `app.py` and `routes.py`/`ws.py` need --
the harness adapter and the error envelope -- rather than putting them in `app.py` and
having the route modules import from it. `app.py` must import the routers (to mount them)
and exposes a module-level `app`, so if the routers also imported `app.py` the cycle would
break for whichever module was imported first. Keeping the shared pieces one level up means
the dependency only ever points inwards.

**The harness contract.** The orchestrator is a separately-owned module; this adapter binds
to it by name at startup instead of importing one fixed symbol, and passes only the keyword
arguments its signature actually declares. The expected shape is:

    harness = get_harness()                    # or Harness() / get_pipeline() / ...
    await harness.warmup()
    await harness.run(text=..., lang=..., mode=..., request_id=..., on_event=...)
    await harness.run(audio=..., mime=..., lang=..., mode=..., request_id=..., on_event=...)

Separate `run_text` / `run_audio` entry points are equally acceptable, as is a payload bound
positionally; `_invoke` works out which by inspecting the signature. `on_event` is called as
`on_event(kind, name, value)` -- the shape `voxrag.harness` uses -- or with a single
`{"type": ...}` dict, sync or async either way; `ws.py` accepts both. A harness that declares
no `on_event` parameter at all still works: it simply streams nothing, and `ws.py`
synthesises the stage frames from the finished trace instead.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Sequence
from typing import Any

from voxrag.contracts import GenerationMode, Lang, PipelineResult

log = logging.getLogger(__name__)

#: Modules searched for the orchestrator, in order of preference.
_HARNESS_MODULES: tuple[str, ...] = (
    "voxrag.harness",
    "voxrag.harness.pipeline",
    "voxrag.harness.orchestrator",
)
#: Zero-argument factories first, then classes, tried in this order on each module above.
_HARNESS_FACTORIES: tuple[str, ...] = (
    "get_harness",
    "get_pipeline",
    "get_orchestrator",
    "Harness",
    "Pipeline",
    "Orchestrator",
)
#: Entry points for a text query and for an audio query. Bare `run` is last in both lists
#: because it is the ambiguous name: a harness that has it *and* a specific one meant the
#: specific one.
_TEXT_ENTRYPOINTS: tuple[str, ...] = ("run_text", "run_query", "query", "answer", "run")
_AUDIO_ENTRYPOINTS: tuple[str, ...] = ("run_audio", "run_voice", "run_speech", "run")

#: Parameter names the payload may be bound to. `Harness.run` is keyword-only
#: (`run(*, audio=None, text=None, ...)`) because one entry point serves both input kinds,
#: so the payload cannot simply be passed positionally.
_TEXT_PAYLOAD_NAMES: tuple[str, ...] = ("text", "query", "question")
_AUDIO_PAYLOAD_NAMES: tuple[str, ...] = ("audio", "audio_bytes", "blob", "data")


class HarnessUnavailable(RuntimeError):
    """No usable orchestrator could be resolved. Surfaced as a 503, never as a 500."""


def resolve_harness() -> Any:
    """Build (or fetch) the process-wide orchestrator.

    An ImportError raised from *inside* the harness is re-raised untouched: a missing
    dependency in the pipeline is a real failure worth reading, and reporting it as
    "harness not found" would send someone hunting the wrong bug.
    """
    tried: list[str] = []
    for module_name in _HARNESS_MODULES:
        try:
            module = __import__(module_name, fromlist=["*"])
        except ImportError as exc:
            if getattr(exc, "name", None) not in (module_name, None):
                raise  # the module exists but its own imports are broken
            tried.append(f"{module_name}: {exc}")
            continue
        for attr in _HARNESS_FACTORIES:
            factory = getattr(module, attr, None)
            if callable(factory):
                log.info("harness resolved: %s.%s", module_name, attr)
                return factory()
        tried.append(f"{module_name}: none of {list(_HARNESS_FACTORIES)}")
    raise HarnessUnavailable("could not resolve a harness entry point. Tried " + "; ".join(tried))


def _bind(obj: Any, names: Sequence[str]) -> tuple[Callable[..., Any], set[str], bool]:
    """First callable of `names` on `obj`, plus which keyword arguments it accepts.

    Returning the accepted parameter set is the point: the caller passes the union of
    everything it *could* supply (lang, mode, request_id, on_event, mime_type) and this
    drops whatever the target does not declare, instead of raising TypeError on a keyword
    the orchestrator chose not to take.
    """
    for name in names:
        method = getattr(obj, name, None)
        if not callable(method):
            continue
        try:
            params = inspect.signature(method).parameters
        except (TypeError, ValueError):  # C callables expose no signature
            return method, set(), True
        accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        keywords = {
            n
            for n, p in params.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        return method, keywords, accepts_var_kw
    raise HarnessUnavailable(f"harness {type(obj).__name__} exposes none of {list(names)}")


async def _invoke(
    obj: Any,
    names: Sequence[str],
    payload: Any,
    payload_names: Sequence[str],
    **kwargs: Any,
) -> PipelineResult:
    """Call a harness entry point with whatever subset of `kwargs` it accepts."""
    method, keywords, var_kw = _bind(obj, names)
    filtered = {k: v for k, v in kwargs.items() if var_kw or k in keywords}
    # Keyword binding first: it is the only form that works for a keyword-only entry point,
    # and it is equally correct for `run_text(text, *, ...)`. Positional is the fallback for
    # a signature whose payload parameter is named something this module has never heard of.
    slot = next((name for name in payload_names if name in keywords), None)
    result = method(**{slot: payload, **filtered}) if slot else method(payload, **filtered)
    if inspect.isawaitable(result):
        result = await result
    if isinstance(result, PipelineResult):
        return result
    # A dict is accepted so the harness may serialise across a boundary later; a validation
    # failure here is a genuine contract break and should raise loudly rather than be coerced.
    return PipelineResult.model_validate(result)


async def run_text(
    harness: Any,
    text: str,
    *,
    lang: Lang | None,
    mode: GenerationMode,
    request_id: str,
    on_event: Callable[[dict[str, Any]], Any] | None = None,
) -> PipelineResult:
    return await _invoke(
        harness,
        _TEXT_ENTRYPOINTS,
        text,
        _TEXT_PAYLOAD_NAMES,
        lang=lang,
        mode=mode,
        request_id=request_id,
        on_event=on_event,
    )


async def run_audio(
    harness: Any,
    audio: bytes,
    *,
    mime_type: str,
    lang: Lang | None,
    mode: GenerationMode,
    request_id: str,
    on_event: Callable[[dict[str, Any]], Any] | None = None,
) -> PipelineResult:
    # `mime`/`lang_hint` are passed alongside `mime_type`/`lang` because both spellings are
    # in use across the codebase (STTProvider.transcribe takes mime_type + lang_hint); the
    # signature filter in `_invoke` keeps whichever pair the harness actually declares.
    return await _invoke(
        harness,
        _AUDIO_ENTRYPOINTS,
        audio,
        _AUDIO_PAYLOAD_NAMES,
        mime_type=mime_type,
        mime=mime_type,
        lang=lang,
        lang_hint=lang,
        mode=mode,
        request_id=request_id,
        on_event=on_event,
    )


def error_body(request_id: str, exc: BaseException, **extra: Any) -> dict[str, Any]:
    """Structured error payload. Always carries `request_id`, so a failure stays traceable.

    `VoxRAGError` already describes itself (including whether the failure was recoverable),
    so its own dict is used when the exception provides one.
    """
    body: dict[str, Any] = {"request_id": request_id, "ok": False}
    as_dict = getattr(exc, "as_dict", None)
    if callable(as_dict):
        try:
            body.update(as_dict())
        except Exception:  # a broken __str__ must not replace the error with another one
            body.update({"error": type(exc).__name__, "message": str(exc)})
    else:
        body.update({"error": type(exc).__name__, "message": str(exc) or type(exc).__name__})
    body.update(extra)
    return body


def __getattr__(name: str) -> Any:
    """Lazy `create_app`, so importing this package does not build a FastAPI app."""
    if name in ("create_app", "app"):
        from voxrag.api import app as module

        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HarnessUnavailable",
    "create_app",
    "error_body",
    "resolve_harness",
    "run_audio",
    "run_text",
]
