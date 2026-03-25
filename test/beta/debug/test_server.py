# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from autogen.beta.debug.server import DebugServer, _Session, _Stream, _create_fastapi_app, _session_view


def _make_stream(stream_id: str = "stream-1", prompt: list[str] | None = None) -> _Stream:
    return _Stream(stream_id, prompt or ["Be helpful."])


def _make_session(session_id: str = "sess-1", name: str = "test", stream_id: str = "stream-1") -> _Session:
    return _Session(session_id, name, stream_id)


@pytest_asyncio.fixture()
async def client_and_state():  # type: ignore[misc]
    streams: dict[str, _Stream] = {}
    sessions: dict[str, _Session] = {}
    app = _create_fastapi_app(streams, sessions)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac, streams, sessions


@pytest.mark.asyncio()
async def test_list_sessions_empty(client_and_state) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = client_and_state
    resp = await client.get("/sessions")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio()
async def test_register_stream(client_and_state) -> None:  # type: ignore[no-untyped-def]
    client, streams, _ = client_and_state
    resp = await client.post("/streams/s1", json={"prompt": ["Be helpful."]})
    assert resp.status_code == 201
    assert "s1" in streams
    assert streams["s1"].prompt == ["Be helpful."]


@pytest.mark.asyncio()
async def test_register_session(client_and_state) -> None:  # type: ignore[no-untyped-def]
    client, streams, sessions = client_and_state
    streams["st1"] = _make_stream("st1")
    resp = await client.post("/sessions", json={"session_id": "s1", "name": "my-session", "stream_id": "st1"})
    assert resp.status_code == 201
    assert "s1" in sessions
    assert sessions["s1"].name == "my-session"
    assert sessions["s1"].stream_ids == ["st1"]


@pytest.mark.asyncio()
async def test_register_session_unknown_stream(client_and_state) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = client_and_state
    resp = await client.post("/sessions", json={"session_id": "s1", "name": "x", "stream_id": "nope"})
    assert resp.status_code == 404


@pytest.mark.asyncio()
async def test_get_session_not_found(client_and_state) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = client_and_state
    resp = await client.get("/sessions/nope")
    assert resp.status_code == 404


@pytest.mark.asyncio()
async def test_receive_and_get_events(client_and_state) -> None:  # type: ignore[no-untyped-def]
    client, streams, sessions = client_and_state
    streams["st2"] = _make_stream("st2")
    sessions["s2"] = _make_session("s2", "test", "st2")

    await client.post("/streams/st2/events", json={"event_type": "ModelRequest", "event_data": {"content": "hello"}})

    resp = await client.get("/sessions/s2")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["events"]) == 1
    assert data["events"][0]["event_type"] == "ModelRequest"
    assert data["events"][0]["event_data"]["content"] == "hello"
    assert data["stream_ids"] == ["st2"]
    assert data["name"] == "test"


@pytest.mark.asyncio()
async def test_get_context(client_and_state) -> None:  # type: ignore[no-untyped-def]
    client, streams, sessions = client_and_state
    streams["st6"] = _make_stream("st6", prompt=["Be helpful."])
    sessions["s6"] = _make_session("s6", "ctx-test", "st6")

    resp = await client.get("/sessions/s6/context")
    assert resp.status_code == 200
    assert resp.json()["prompt"] == ["Be helpful."]


@pytest.mark.asyncio()
async def test_re_register_stream_resets_events(client_and_state) -> None:  # type: ignore[no-untyped-def]
    """Re-registering an existing stream should reset its events."""
    client, streams, _ = client_and_state

    await client.post("/streams/re1", json={"prompt": ["Original."]})
    streams["re1"].events.append({"event_type": "old", "event_data": {}})

    resp = await client.post("/streams/re1", json={"prompt": ["Updated."]})
    assert resp.status_code == 201
    assert streams["re1"].prompt == ["Updated."]
    assert streams["re1"].events == []


@pytest.mark.asyncio()
async def test_receive_event_unknown_stream(client_and_state) -> None:  # type: ignore[no-untyped-def]
    """Posting events to a non-existent stream should return 404."""
    client, _, _ = client_and_state
    resp = await client.post("/streams/nope/events", json={"event_type": "X", "event_data": {}})
    assert resp.status_code == 404


@pytest.mark.asyncio()
async def test_list_sessions_with_data(client_and_state) -> None:  # type: ignore[no-untyped-def]
    client, streams, sessions = client_and_state
    streams["st"] = _make_stream("st")
    sessions["a"] = _make_session("a", "alpha", "st")
    sessions["b"] = _make_session("b", "beta", "st")

    resp = await client.get("/sessions")
    assert resp.status_code == 200
    items = resp.json()
    ids = {s["id"] for s in items}
    assert ids == {"a", "b"}
    names = {s["name"] for s in items}
    assert names == {"alpha", "beta"}


@pytest.mark.asyncio()
async def test_get_context_unknown_session(client_and_state) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = client_and_state
    resp = await client.get("/sessions/missing/context")
    assert resp.status_code == 404


@pytest.mark.asyncio()
async def test_ui_endpoint(client_and_state) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = client_and_state
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_session_view_helper() -> None:
    streams = {"st1": _make_stream("st1")}
    streams["st1"].events.append({"event_type": "X", "event_data": {}, "timestamp": "2026-01-01T00:00:00"})
    session = _Session("sv1", "view-test", "st1")
    view = _session_view(session, streams)
    assert view.id == "sv1"
    assert view.name == "view-test"
    assert view.stream_ids == ["st1"]
    assert view.prompt == ["Be helpful."]
    assert len(view.events) == 1
    assert view.status == "running"


def test_debug_server_init() -> None:
    srv = DebugServer()
    assert srv._streams == {}
    assert srv._sessions == {}
    assert srv._app is not None


@pytest.mark.asyncio()
async def test_start_debug_server() -> None:
    """start_debug_server should create a DebugServer and schedule uvicorn."""
    import uvicorn as _uvicorn

    mock_server = MagicMock()
    mock_server.serve = AsyncMock()

    with (
        patch.object(_uvicorn, "Config", return_value=MagicMock()) as mock_config,
        patch.object(_uvicorn, "Server", return_value=mock_server),
        patch("asyncio.create_task") as mock_create_task,
    ):
        from autogen.beta.debug.server import start_debug_server

        srv = await start_debug_server(host="0.0.0.0", port=9999)

        assert isinstance(srv, DebugServer)
        mock_config.assert_called_once()
        mock_create_task.assert_called_once()


def test_run_debug_server() -> None:
    """run_debug_server should call uvicorn.run with the app."""
    import uvicorn as _uvicorn

    with patch.object(_uvicorn, "run") as mock_run:
        from autogen.beta.debug.server import run_debug_server

        run_debug_server(host="0.0.0.0", port=9999)

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs[1]["host"] == "0.0.0.0"
        assert call_kwargs[1]["port"] == 9999


@pytest.mark.asyncio()
async def test_websocket_sends_initial_sessions() -> None:
    """WebSocket connection should receive initial sessions list."""
    from starlette.testclient import TestClient

    streams: dict[str, _Stream] = {}
    sessions: dict[str, _Session] = {}
    streams["st-ws"] = _Stream("st-ws", ["p"])
    sessions["ws1"] = _Session("ws1", "ws-test", "st-ws")
    app = _create_fastapi_app(streams, sessions)

    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        data = ws.receive_json()
        assert data["type"] == "sessions_list"
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["id"] == "ws1"
        assert data["sessions"][0]["name"] == "ws-test"


@pytest.mark.asyncio()
async def test_end_session_freezes_snapshot(client_and_state) -> None:  # type: ignore[no-untyped-def]
    """Ending a session should freeze its event snapshot."""
    client, streams, sessions = client_and_state
    streams["st"] = _make_stream("st")
    sessions["s1"] = _make_session("s1", "snap-test", "st")

    # Add an event while session is running
    await client.post("/streams/st/events", json={"event_type": "ModelRequest", "event_data": {"content": "hello"}})

    # End the session
    resp = await client.post("/sessions/s1/done")
    assert resp.status_code == 200
    assert sessions["s1"].status == "done"
    assert sessions["s1"].event_snapshot is not None
    assert len(sessions["s1"].event_snapshot) == 1

    # Add another event to the stream AFTER session ended
    await client.post("/streams/st/events", json={"event_type": "ModelResponse", "event_data": {"message": "bye"}})

    # The ended session should still show only 1 event (snapshot)
    resp = await client.get("/sessions/s1")
    data = resp.json()
    assert len(data["events"]) == 1
    assert data["status"] == "done"


@pytest.mark.asyncio()
async def test_two_sessions_same_stream(client_and_state) -> None:  # type: ignore[no-untyped-def]
    """Two running sessions on the same stream should see the same events."""
    client, streams, sessions = client_and_state
    streams["shared"] = _make_stream("shared")
    sessions["a"] = _make_session("a", "alpha", "shared")
    sessions["b"] = _make_session("b", "beta", "shared")

    await client.post("/streams/shared/events", json={"event_type": "ModelRequest", "event_data": {"content": "hi"}})

    resp_a = await client.get("/sessions/a")
    resp_b = await client.get("/sessions/b")
    assert len(resp_a.json()["events"]) == 1
    assert len(resp_b.json()["events"]) == 1


@pytest.mark.asyncio()
async def test_end_session_not_found(client_and_state) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = client_and_state
    resp = await client.post("/sessions/nope/done")
    assert resp.status_code == 404


@pytest.mark.asyncio()
async def test_add_stream_to_session(client_and_state) -> None:  # type: ignore[no-untyped-def]
    """Adding a second stream to a session should aggregate events from both."""
    client, streams, sessions = client_and_state
    streams["st-a"] = _make_stream("st-a")
    streams["st-b"] = _make_stream("st-b")
    sessions["s1"] = _make_session("s1", "multi", "st-a")

    await client.post("/streams/st-a/events", json={"event_type": "ModelRequest", "event_data": {"content": "a"}})

    # Add second stream
    resp = await client.post("/sessions/s1/streams", json={"stream_id": "st-b"})
    assert resp.status_code == 200
    assert sessions["s1"].stream_ids == ["st-a", "st-b"]

    await client.post("/streams/st-b/events", json={"event_type": "ModelRequest", "event_data": {"content": "b"}})

    # Session should see events from both streams
    resp = await client.get("/sessions/s1")
    data = resp.json()
    assert len(data["events"]) == 2
    assert data["stream_ids"] == ["st-a", "st-b"]


@pytest.mark.asyncio()
async def test_add_stream_to_session_not_found(client_and_state) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = client_and_state
    resp = await client.post("/sessions/nope/streams", json={"stream_id": "x"})
    assert resp.status_code == 404


@pytest.mark.asyncio()
async def test_add_unknown_stream_to_session(client_and_state) -> None:  # type: ignore[no-untyped-def]
    client, streams, sessions = client_and_state
    streams["st"] = _make_stream("st")
    sessions["s1"] = _make_session("s1", "test", "st")
    resp = await client.post("/sessions/s1/streams", json={"stream_id": "nope"})
    assert resp.status_code == 404


@pytest.mark.asyncio()
async def test_end_session_multi_stream_snapshot(client_and_state) -> None:  # type: ignore[no-untyped-def]
    """Ending a multi-stream session should freeze events from all streams."""
    client, streams, sessions = client_and_state
    streams["st-x"] = _make_stream("st-x")
    streams["st-y"] = _make_stream("st-y")
    sessions["s1"] = _make_session("s1", "multi-snap", "st-x")
    sessions["s1"].stream_ids.append("st-y")

    await client.post("/streams/st-x/events", json={"event_type": "ModelRequest", "event_data": {"content": "x"}})
    await client.post("/streams/st-y/events", json={"event_type": "ModelRequest", "event_data": {"content": "y"}})

    await client.post("/sessions/s1/done")

    # Add more events after close
    await client.post("/streams/st-x/events", json={"event_type": "ModelResponse", "event_data": {"content": "z"}})

    resp = await client.get("/sessions/s1")
    data = resp.json()
    assert len(data["events"]) == 2  # snapshot frozen at 2, not 3
