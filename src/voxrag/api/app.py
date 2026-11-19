"""FastAPI application: wiring and lifecycle.

Two things live here and nowhere else:

* **`create_app()`** -- an application factory, so uvicorn's `--reload` and the tests can
  each build a clean app with its own state instead of sharing one mutable global. A
  module-level `app` is also exported, because `scripts/entrypoint.sh` serves
  `voxrag.api.app:app --factory=false`.
* **The startup warmup.** Every expensive thing in this system is expensive exactly once:
  the ONNX session, the HNSW graph or the Qdrant connection pool, the BM25 score matrices,
  the STT TLS handshake. Paying for those on the first user request would put seconds inside
  a 200ms budget and land in the P100 the benchmark reports, so startup pays instead.

The harness adapter and the error envelope live in `voxrag.api.__init__`; see the note
there for why they are not in this module.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from voxrag.api import error_body, resolve_harness
from voxrag.config import REPO_ROOT, settings

log = logging.getLogger(__name__)

#: The UI bundle. Owned by the web module; mounted read-only at /static.
WEB_DIR = REPO_ROOT / "web"

#: The Vite build output, and the entry document `GET /` serves. Vite writes its asset URLs
#: absolute against `base = '/static/dist/'` (see web-ui/vite.config.ts), which is exactly
#: where the /static mount exposes this directory -- so the built document is served
#: verbatim from here with no rewriting and no second copy to fall out of date.
UI_DIST = WEB_DIR / "dist"
UI_INDEX = UI_DIST / "index.html"


# --------------------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------------------


async def _warmup(harness: Any | None) -> dict[str, Any]:
    """Pay every one-off cost before the first request.

    Prefers the harness's own `warmup()`, which knows the real dependency graph. Falls back
    to warming each component directly, so a harness without one still serves a warm first
    request instead of a multi-second outlier.
    """
    if harness is not None:
        warm = getattr(harness, "warmup", None)
        if callable(warm):
            result = warm()
            if inspect.isawaitable(result):
                result = await result
            return {"harness": result if isinstance(result, dict) else True}

    from voxrag import generate as generate_mod
    from voxrag import stt as stt_mod
    from voxrag.embed import get_embedder
    from voxrag.store.hybrid import get_hybrid_retriever

    async def _embedder() -> Any:
        encoder = await asyncio.to_thread(get_embedder)  # a blocking model load
        return {"model": encoder.name, "dim": encoder.dim}

    async def _retriever() -> Any:
        await get_hybrid_retriever().warmup()
        return True

    names = ("embedder", "retrieval", "stt", "generate")
    outcomes = await asyncio.gather(
        _embedder(),
        _retriever(),
        stt_mod.warmup(),
        generate_mod.warmup(),
        return_exceptions=True,
    )
    warmed: dict[str, Any] = {}
    for name, outcome in zip(names, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            # A cold component is a degraded service, not a failed boot: retrieval still
            # works without a generator, and /api/health reports the truth either way.
            log.warning("warmup failed for %s: %r", name, outcome)
            warmed[name] = f"failed: {outcome!r}"
        else:
            warmed[name] = outcome
    return warmed


async def _shutdown() -> None:
    """Release pooled clients. Each closes independently; one failure closes the rest."""
    from voxrag import generate as generate_mod
    from voxrag import stt as stt_mod

    tasks: list[Any] = [stt_mod.aclose_all(), generate_mod.aclose()]
    try:
        from voxrag.store import get_vector_store

        close = getattr(get_vector_store(), "close", None)
        if close is not None:
            tasks.append(close())
    except Exception as exc:  # the store may never have been built
        log.debug("no vector store to close: %r", exc)
    for outcome in await asyncio.gather(*tasks, return_exceptions=True):
        if isinstance(outcome, BaseException):
            log.warning("shutdown step failed: %r", outcome)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    app.state.started_at = time.time()
    app.state.harness = None
    app.state.harness_error = None
    app.state.warmup = {}

    try:
        app.state.harness = resolve_harness()
    except Exception as exc:
        # Boot anyway: /api/health has to be reachable in order to *report* that the harness
        # is the broken thing. A process that refuses to start explains nothing, and the
        # container healthcheck would restart-loop it before anyone read the log.
        app.state.harness_error = f"{type(exc).__name__}: {exc}"
        log.error("harness unavailable: %s", app.state.harness_error)

    started = time.perf_counter()
    try:
        app.state.warmup = await _warmup(app.state.harness)
    except Exception as exc:
        app.state.warmup = {"error": f"{type(exc).__name__}: {exc}"}
        log.error("warmup failed: %r", exc)
    log.info(
        "warmup finished in %.0fms: %s", (time.perf_counter() - started) * 1000.0, app.state.warmup
    )

    yield

    await _shutdown()


# --------------------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------------------


def create_app() -> FastAPI:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    app = FastAPI(
        title="voxrag",
        version="0.1.0",
        description="Voice-enabled multilingual RAG over ai4bharat/MSMARCO-XI.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        # allow_credentials with a wildcard origin is rejected outright by browsers, and
        # this API is cookieless anyway -- the flag would be a silent no-op.
        allow_credentials="*" not in settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def tag_request(request: Request, call_next: Callable[[Request], Any]) -> Any:
        # One id per request, echoed on the response and threaded into the harness, so a
        # slow or failed request in the browser can be found in the server log.
        request.state.request_id = uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    def _rid(request: Request) -> str:
        return getattr(request.state, "request_id", None) or uuid.uuid4().hex[:12]

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_body(
                _rid(request),
                exc,
                error="ValidationError",
                message="request failed validation",
                fields=[
                    {"loc": [str(p) for p in e.get("loc", ())], "msg": e.get("msg", "")}
                    for e in exc.errors()
                ],
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(
                _rid(request),
                exc,
                error="HTTPError",
                message=str(exc.detail),
                status=exc.status_code,
            ),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Nothing escapes as a bare 500: the client gets the request id, and the traceback
        # goes to the log where it belongs rather than into the browser.
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=error_body(_rid(request), exc, message=f"internal error: {exc}"),
        )

    from voxrag.api.routes import router
    from voxrag.api.ws import ws_router

    app.include_router(router)
    app.include_router(ws_router)

    # Created if absent: StaticFiles raises at mount time on a missing directory, which
    # would let an absent UI bundle take the API down with it.
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def index(request: Request) -> Any:
        if not UI_INDEX.exists():
            return JSONResponse(
                status_code=503,
                content={
                    "request_id": _rid(request),
                    "ok": False,
                    "error": "UIMissing",
                    "message": (
                        f"{UI_INDEX} not found; run `npm run build` in web-ui/. "
                        "The API itself is up at /api/health"
                    ),
                },
            )
        # no-store: the document names hash-fingerprinted assets, so a browser holding a
        # stale copy of it would request a bundle that no longer exists. The assets it
        # names are immutable and cached normally by the /static mount.
        return FileResponse(
            UI_INDEX, media_type="text/html", headers={"Cache-Control": "no-store"}
        )

    return app


#: Module-level instance for `uvicorn voxrag.api.app:app` (what the container entrypoint
#: runs). `create_app` remains the factory form for --reload and for tests.
app = create_app()
