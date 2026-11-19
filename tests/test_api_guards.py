"""Abuse guards on the paid endpoints: origin check, per-client rate limit, bounded buffering.

Speech-to-text costs money per call, so a deployed instance must not be drivable by any web
page a visitor happens to open, and a client that keeps sending must not grow server memory.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from voxrag.api import routes, ws


@pytest.fixture(autouse=True)
def fresh_limits(patch_settings: Callable[..., None]) -> Iterator[None]:
    routes._HITS.clear()
    patch_settings(cors_origins=["http://localhost:5173"], rate_limit_per_minute=30)
    yield
    routes._HITS.clear()


def api(harness: Any = None) -> TestClient:
    """The real routers. With no harness, anything that gets past the guards is a 503; with
    a stand-in one, a websocket stays open waiting for audio."""
    app = FastAPI()
    app.include_router(routes.router)
    app.include_router(ws.ws_router)
    app.state.harness = harness
    return TestClient(app)


class TestOrigin:
    def test_no_origin_is_a_non_browser_client(self) -> None:
        assert routes.origin_allowed(None)

    def test_allowlisted_dev_server(self) -> None:
        assert routes.origin_allowed("http://localhost:5173")

    def test_foreign_site_refused(self) -> None:
        assert not routes.origin_allowed("https://evil.example")

    def test_sandboxed_null_origin_refused(self) -> None:
        assert not routes.origin_allowed("null")

    def test_wildcard_opens_it_to_every_site(self, patch_settings: Callable[..., None]) -> None:
        patch_settings(cors_origins=["*"])
        assert routes.origin_allowed("https://evil.example")


class TestRateLimit:
    def test_sliding_window_per_client(self, patch_settings: Callable[..., None]) -> None:
        patch_settings(rate_limit_per_minute=3)
        assert [routes.within_rate_limit("a", now=t) for t in (0, 1, 2, 3)] == [
            True,
            True,
            True,
            False,
        ]
        assert routes.within_rate_limit("b", now=3)  # another client has its own budget
        assert routes.within_rate_limit("a", now=60.5)  # the first hit has aged out

    def test_zero_disables(self, patch_settings: Callable[..., None]) -> None:
        patch_settings(rate_limit_per_minute=0)
        assert all(routes.within_rate_limit("a", now=0) for _ in range(100))


class TestHttpGuards:
    def test_foreign_origin_is_refused_before_any_work(self) -> None:
        response = api().post(
            "/api/query", json={"text": "hi"}, headers={"origin": "https://evil.example"}
        )
        assert response.status_code == 403

    def test_voice_upload_from_a_foreign_origin_is_refused(self) -> None:
        # multipart is a "simple" request: no preflight, so CORS alone would not stop it.
        response = api().post(
            "/api/voice",
            files={"file": ("q.webm", b"\x1a\x45\xdf\xa3", "audio/webm")},
            headers={"origin": "https://evil.example"},
        )
        assert response.status_code == 403

    def test_matching_host_is_not_trusted(self) -> None:
        # DNS rebinding: the hostile page's Origin and the Host it sends name the same domain.
        response = api().post(
            "/api/query",
            json={"text": "hi"},
            headers={"origin": "http://rebind.example:8000", "host": "rebind.example:8000"},
        )
        assert response.status_code == 403

    def test_over_the_limit_returns_429(self, patch_settings: Callable[..., None]) -> None:
        patch_settings(rate_limit_per_minute=1)
        client = api()
        assert client.post("/api/query", json={"text": "hi"}).status_code == 503
        assert client.post("/api/query", json={"text": "hi"}).status_code == 429


class TestWebsocketGuards:
    def test_foreign_origin_handshake_is_refused(self) -> None:
        with (
            pytest.raises(WebSocketDisconnect),
            api().websocket_connect("/ws/voice", headers={"origin": "https://evil.example"}),
        ):
            pass

    def test_allowlisted_handshake_is_accepted(self, patch_settings: Callable[..., None]) -> None:
        patch_settings(cors_origins=["http://testserver"])
        with api().websocket_connect("/ws/voice", headers={"origin": "http://testserver"}) as sock:
            assert "pipeline unavailable" in sock.receive_json()["message"]

    def test_one_client_cannot_take_every_slot(self, patch_settings: Callable[..., None]) -> None:
        patch_settings(ws_max_connections_per_client=1)
        client = api(harness=object())
        with (
            client.websocket_connect("/ws/voice"),
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect("/ws/voice"),
        ):
            pass
        assert ws._open_by_client == {}  # the slot is released on close

    def test_idle_socket_is_closed(self, patch_settings: Callable[..., None]) -> None:
        patch_settings(ws_idle_timeout_s=0.05)
        with api(harness=object()).websocket_connect("/ws/voice") as sock:
            sock.send_json({"type": "ping"})  # keepalives do not extend the deadline
            with pytest.raises(WebSocketDisconnect) as closed:
                sock.receive_json()
        assert closed.value.code == 1000

    def test_connection_cap(self, patch_settings: Callable[..., None]) -> None:
        patch_settings(ws_max_connections=0)
        with pytest.raises(WebSocketDisconnect), api().websocket_connect("/ws/voice"):
            pass


class FloodSocket:
    """A client that sends audio frames as fast as it is read."""

    def __init__(self) -> None:
        self.reads = 0

    async def receive(self) -> dict[str, Any]:
        self.reads += 1
        return {"type": "websocket.receive", "text": '{"type": "audio"}'}


async def test_reader_stops_reading_when_the_queue_is_full() -> None:
    sock = FloodSocket()
    inbound: asyncio.Queue[Any] = asyncio.Queue(maxsize=ws._INBOUND_MAX)
    closed = asyncio.Event()
    reader = asyncio.create_task(ws._reader(sock, inbound, closed))  # type: ignore[arg-type]
    for _ in range(50):
        await asyncio.sleep(0)

    assert inbound.qsize() == ws._INBOUND_MAX
    assert sock.reads <= ws._INBOUND_MAX + 1
    assert not reader.done()  # waiting for room, not crashed on a full queue

    reader.cancel()
    await asyncio.gather(reader, return_exceptions=True)
    queued = [inbound.get_nowait() for _ in range(inbound.qsize())]
    assert queued[-1] is ws._END  # the handler still learns the client is gone
    assert closed.is_set()
