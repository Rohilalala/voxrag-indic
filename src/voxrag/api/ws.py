"""The /ws/voice websocket.

    client -> {"type":"audio","mime":str,"data":<base64>,"lang":str|null,"mode":str}
    server -> {"type":"stage","stage":str,"ms":float}
              {"type":"transcript","text":str,"lang":str}
              {"type":"token","text":str}                (quality mode only)
              {"type":"result","result":<PipelineResult JSON>}
              {"type":"error","message":str}

The socket exists so the UI can show the pipeline *happening* -- stage bars filling in, the
transcript landing before retrieval starts, tokens arriving one at a time -- rather than a
spinner followed by a finished object. Frames therefore have to leave as the harness
produces them, which shapes the three moving parts below.

* **One reader task per connection.** It is the only thing that calls `receive`, and it
  publishes both inbound messages and the disconnect onto queues. Two coroutines reading one
  socket is undefined behaviour, and -- this is the part that actually matters -- a
  disconnect is *only* observable through `receive`. A closed socket does not make `send`
  fail: ASGI sends after a client vanishes are silently dropped, so a design that waits for
  a send to raise never notices the client left and generates an answer into a void.
* **A queue between the harness and the socket.** `on_event` only appends; the pump task
  does the awaiting. A client on a slow link cannot then apply backpressure to the pipeline
  and inflate the very latency numbers it is watching.
* **Cancellation, not abandonment.** When the reader reports a disconnect mid-run, the
  harness task is cancelled -- it holds a generator stream and retrieval work that would
  otherwise run to completion for nobody.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Final

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from voxrag.api import error_body, run_audio
from voxrag.api.routes import (
    MAX_AUDIO_BYTES,
    client_key,
    coerce_lang,
    coerce_mode,
    origin_allowed,
    sniff_mime,
    within_rate_limit,
)
from voxrag.config import settings
from voxrag.contracts import PipelineResult, StageName

log = logging.getLogger(__name__)

ws_router = APIRouter()

#: Frame types the server may emit mid-run. Anything else the harness reports is dropped
#: rather than forwarded, so a stray event cannot corrupt the client's state machine.
_STREAMABLE: Final[frozenset[str]] = frozenset({"stage", "transcript", "token"})

#: Queue sentinels. `_END` closes a queue; `_BAD_JSON` marks a frame that did not parse,
#: which is a client bug worth reporting rather than a reason to drop the connection.
_END: Final[object] = object()
_BAD_JSON: Final[object] = object()

#: Frames waiting while a turn runs. Bounded so a client that keeps sending cannot grow
#: server memory: the reader stops reading, and the socket pushes back on the sender.
_INBOUND_MAX: Final[int] = 2

_open_sockets = 0


def _normalize(*event: Any) -> dict[str, Any] | None:
    """Coerce one harness event into a wire frame, or None if it is not streamable.

    Two call shapes are accepted. `voxrag.harness` emits positionally as
    `on_event(kind, name, value)` -- stage/ms, transcript/lang, token/None -- and a single
    dict `{"type": ..., ...}` is accepted as well, since the frame vocabulary here is the
    websocket's and not the orchestrator's to define.

    Kinds outside the wire contract (`retrieval`, `guardrail`, `answer`, `error`) are
    dropped rather than forwarded: the finished result frame already carries all of them,
    and an `error` event on the harness's recovery path would flash a failure in the UI for
    a request that goes on to return a perfectly good refusal.
    """
    if len(event) == 1 and isinstance(event[0], dict):
        payload = event[0]
        kind = str(payload.get("type") or "")
        name: Any = payload.get("stage") if kind == "stage" else payload.get("text")
        value: Any = payload.get("ms") if kind == "stage" else payload.get("lang")
    elif len(event) >= 2:
        kind, name = str(event[0]), event[1]
        value = event[2] if len(event) > 2 else None
    else:
        return None

    if kind not in _STREAMABLE:
        return None
    if kind == "stage":
        stage = name.value if isinstance(name, StageName) else str(name)
        return {"type": "stage", "stage": stage, "ms": float(value or 0.0)}
    if kind == "transcript":
        return {
            "type": "transcript",
            "text": str(name or ""),
            "lang": str(getattr(value, "value", value) or ""),
        }
    return {"type": "token", "text": str(name or "")}


def _fallback_frames(result: PipelineResult, streamed: set[str]) -> list[dict[str, Any]]:
    """Frames the harness did not stream, reconstructed from the finished result.

    A harness that takes no `on_event` callback still has to drive the UI's stage waterfall
    and transcript panel, so those frames are synthesised from the trace after the fact.
    Tokens are never synthesised: a token frame means "this arrived incrementally", and
    faking that would be a lie about latency.
    """
    frames: list[dict[str, Any]] = []
    if "transcript" not in streamed and result.transcript is not None:
        lang = result.transcript.lang or result.detected_lang
        frames.append(
            {
                "type": "transcript",
                "text": result.transcript.text,
                "lang": lang.value if lang is not None else "",
            }
        )
    if "stage" not in streamed:
        frames.extend(
            {"type": "stage", "stage": timing.stage.value, "ms": timing.ms}
            for timing in result.timings
            if not timing.skipped
        )
    return frames


def _decode_audio(payload: Any) -> bytes:
    """Base64 (optionally a `data:` URL) -> bytes, under the same cap as the REST route."""
    if not isinstance(payload, str) or not payload:
        raise ValueError("missing base64 'data'")
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")  # data:audio/webm;base64,<...>
    # Checked before decoding: base64 inflates by 4/3, so this refuses the allocation
    # rather than performing it and then complaining about it.
    if len(payload) * 3 // 4 > MAX_AUDIO_BYTES:
        raise ValueError(f"audio exceeds {MAX_AUDIO_BYTES // (1024 * 1024)}MB")
    try:
        audio = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"'data' is not valid base64: {exc}") from exc
    if not audio:
        raise ValueError("empty audio payload")
    return audio


async def _reader(ws: WebSocket, inbound: asyncio.Queue[Any], closed: asyncio.Event) -> None:
    """The connection's only consumer of `receive`. Ends by publishing `_END`."""
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                return
            raw = message.get("text")
            if raw is None:
                blob = message.get("bytes") or b""
                raw = blob.decode("utf-8", "replace")
            try:
                frame = json.loads(raw)
            except ValueError:
                frame = _BAD_JSON
            await inbound.put(frame)
    except (WebSocketDisconnect, RuntimeError) as exc:
        log.debug("ws reader stopped: %r", exc)
    finally:
        closed.set()
        # Nothing still queued matters once the client is gone, so make room for _END
        # rather than block on a full queue.
        if inbound.full():
            inbound.get_nowait()
        inbound.put_nowait(_END)


async def _pump(ws: WebSocket, outbound: asyncio.Queue[Any]) -> None:
    """Drain queued frames onto the socket until the sentinel arrives."""
    while True:
        frame = await outbound.get()
        if frame is _END:
            return
        try:
            await ws.send_json(frame)
        except (WebSocketDisconnect, RuntimeError) as exc:
            # The reader owns disconnect detection; this only stops us from looping on a
            # socket that is already gone.
            log.debug("ws send failed: %r", exc)
            return


@ws_router.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket) -> None:
    global _open_sockets
    # Closing before accept refuses the handshake: a page on another site cannot drive
    # this socket, and a flood of sockets cannot hold unbounded buffers.
    if not origin_allowed(websocket.headers.get("origin"), websocket.headers.get("host")):
        await websocket.close(code=1008)
        return
    if _open_sockets >= settings.ws_max_connections:
        await websocket.close(code=1013)
        return
    _open_sockets += 1
    try:
        await _serve_voice(websocket)
    finally:
        _open_sockets -= 1


async def _serve_voice(websocket: WebSocket) -> None:
    await websocket.accept()
    # The HTTP request-id middleware does not run for websockets. This id identifies the
    # *connection* in the log; each audio turn mints its own request id, because each turn
    # is a separate PipelineResult with its own trace.
    conn_id = uuid.uuid4().hex[:12]
    harness = getattr(websocket.app.state, "harness", None)

    if harness is None:
        detail = getattr(websocket.app.state, "harness_error", None) or "harness not initialised"
        await websocket.send_json({"type": "error", "message": f"pipeline unavailable: {detail}"})
        await websocket.close(code=1011)
        return

    inbound: asyncio.Queue[Any] = asyncio.Queue(maxsize=_INBOUND_MAX)
    closed = asyncio.Event()
    reader = asyncio.create_task(_reader(websocket, inbound, closed))

    try:
        while True:
            message = await inbound.get()
            if message is _END:
                return  # client disconnected
            if message is _BAD_JSON:
                await _safe_send(websocket, {"type": "error", "message": "expected a JSON frame"})
                continue

            kind = str((message or {}).get("type") or "")
            if kind in ("ping", "pong"):
                continue  # keepalives are not in the contract, but must not raise an error
            if kind != "audio":
                await _safe_send(
                    websocket, {"type": "error", "message": f"unsupported message type {kind!r}"}
                )
                continue
            if not within_rate_limit(client_key(websocket.client)):
                await _safe_send(
                    websocket,
                    {"type": "error", "message": "too many requests; try again in a minute"},
                )
                continue

            await _handle_audio(websocket, harness, message, closed)
    except WebSocketDisconnect:
        return
    except Exception as exc:  # never let the socket die without saying why
        log.exception("websocket failed [%s]", conn_id)
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await websocket.send_json(
                {"type": "error", "message": f"{type(exc).__name__}: {exc}", "request_id": conn_id}
            )
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)


async def _handle_audio(
    websocket: WebSocket, harness: Any, message: dict[str, Any], closed: asyncio.Event
) -> None:
    """Run one audio turn, streaming frames as the harness produces them."""
    request_id = uuid.uuid4().hex[:12]
    try:
        audio = _decode_audio(message.get("data"))
        lang = coerce_lang(message.get("lang"))
        mode = coerce_mode(message.get("mode"))
    except ValueError as exc:
        await _safe_send(
            websocket, {"type": "error", "message": str(exc), "request_id": request_id}
        )
        return

    mime = sniff_mime(message.get("mime"), message.get("filename"), audio[:16])

    loop = asyncio.get_running_loop()
    outbound: asyncio.Queue[Any] = asyncio.Queue()
    streamed: set[str] = set()

    # One completed future, reused for every event: returning it makes `_emit` usable both
    # as `on_event(e)` and as `await on_event(e)`, without allocating a coroutine per event
    # that would raise "never awaited" warnings if the harness calls it synchronously.
    done: asyncio.Future[None] = loop.create_future()
    done.set_result(None)

    def _emit(*event: Any) -> Awaitable[None]:
        frame = _normalize(*event)
        if frame is not None:
            streamed.add(frame["type"])
            # call_soon_threadsafe, not a bare put_nowait: a harness stage running under
            # asyncio.to_thread may emit from a worker thread, and an off-loop put can leave
            # the pump asleep on an already-filled queue.
            loop.call_soon_threadsafe(outbound.put_nowait, frame)
        return done

    emit: Callable[..., Awaitable[None]] = _emit
    pump = asyncio.create_task(_pump(websocket, outbound))
    run = asyncio.create_task(
        run_audio(
            harness,
            audio,
            mime_type=mime,
            lang=lang,
            mode=mode,
            request_id=request_id,
            on_event=emit,
        )
    )
    gone = asyncio.create_task(closed.wait())

    try:
        await asyncio.wait({run, gone}, return_when=asyncio.FIRST_COMPLETED)
        if not run.done():
            run.cancel()  # client left mid-generation; do not leak the pipeline task
            await asyncio.gather(run, return_exceptions=True)
            raise WebSocketDisconnect(code=1001)
        result = run.result()
    except WebSocketDisconnect:
        raise
    except Exception as exc:
        log.exception("voice ws run failed [%s] mime=%s bytes=%d", request_id, mime, len(audio))
        await _safe_send(
            websocket, {"type": "error", **error_body(request_id, exc, stage="pipeline")}
        )
        return
    finally:
        gone.cancel()
        outbound.put_nowait(_END)
        await asyncio.gather(pump, gone, return_exceptions=True)

    if closed.is_set():
        raise WebSocketDisconnect(code=1001)

    for frame in _fallback_frames(result, streamed):
        await _safe_send(websocket, frame)
    await _safe_send(websocket, {"type": "result", "result": result.model_dump(mode="json")})


async def _safe_send(websocket: WebSocket, frame: dict[str, Any]) -> None:
    """Send, converting a dead socket into a disconnect the caller can unwind on."""
    try:
        await websocket.send_json(frame)
    except (WebSocketDisconnect, RuntimeError) as exc:
        raise WebSocketDisconnect(code=1001) from exc
