# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from autogen.beta.debug.server import DebugServer, _Session, _create_fastapi_app, _session_view


def _make_session(session_id: str = "sess-1", prompt: list[str] | None = None) -> _Session:
    return _Session(session_id, prompt or ["Be helpful."])


@pytest_asyncio.fixture()
async def client_and_sessions():  # type: ignore[misc]
    sessions: dict[str, _Session] = {}
    app = _create_fastapi_app(sessions)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac, sessions


@pytest.mark.asyncio()
async def test_list_sessions_empty(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_sessions
    resp = await client.get("/sessions")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio()
async def test_register_session(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    client, sessions = client_and_sessions
    resp = await client.post("/sessions/s1", json={"prompt": ["Be helpful."]})
    assert resp.status_code == 201
    assert "s1" in sessions
    assert sessions["s1"].prompt == ["Be helpful."]


@pytest.mark.asyncio()
async def test_get_session_not_found(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_sessions
    resp = await client.get("/sessions/nope")
    assert resp.status_code == 404


@pytest.mark.asyncio()
async def test_receive_and_get_events(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    client, sessions = client_and_sessions
    sessions["s2"] = _make_session("s2")

    await client.post("/sessions/s2/events", json={"event_type": "ModelRequest", "event_data": {"content": "hello"}})

    resp = await client.get("/sessions/s2")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["events"]) == 1
    assert data["events"][0]["event_type"] == "ModelRequest"
    assert data["events"][0]["event_data"]["content"] == "hello"
    assert data["pending_bp_id"] is None


@pytest.mark.asyncio()
async def test_hit_and_resume_breakpoint(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    client, sessions = client_and_sessions
    sessions["s3"] = _make_session("s3")

    done = asyncio.Event()
    mods_received: list[dict] = []  # type: ignore[type-arg]

    async def _hit() -> None:
        resp = await client.post(
            "/sessions/s3/breakpoints",
            json={"type": "TURN_START", "event": {"type": "ModelRequest", "data": {"content": "x"}}},
        )
        mods_received.append(resp.json())
        done.set()

    task = asyncio.create_task(_hit())
    await asyncio.sleep(0.05)

    bp_id = sessions["s3"].pending_bp_id
    assert bp_id is not None

    resp = await client.post(f"/sessions/s3/breakpoints/{bp_id}/resume", json={})
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    await asyncio.wait_for(task, timeout=1.0)
    assert done.is_set()
    # Empty ResumeRequest → empty mods dict returned to agent
    assert mods_received[0].get("event_modifications") == {}


@pytest.mark.asyncio()
async def test_resume_with_event_modifications(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    client, sessions = client_and_sessions
    sessions["s4"] = _make_session("s4")

    mods_received: list[dict] = []  # type: ignore[type-arg]

    async def _hit() -> None:
        resp = await client.post(
            "/sessions/s4/breakpoints",
            json={"type": "TURN_START", "event": {}},
        )
        mods_received.append(resp.json())

    task = asyncio.create_task(_hit())
    await asyncio.sleep(0.05)

    bp_id = sessions["s4"].pending_bp_id
    assert bp_id is not None

    await client.post(
        f"/sessions/s4/breakpoints/{bp_id}/resume",
        json={"event_modifications": {"content": "patched"}},
    )
    await asyncio.wait_for(task, timeout=1.0)
    assert mods_received[0]["event_modifications"] == {"content": "patched"}


@pytest.mark.asyncio()
async def test_resume_wrong_bp_id(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    client, sessions = client_and_sessions
    sessions["s5"] = _make_session("s5")

    task = asyncio.create_task(
        client.post("/sessions/s5/breakpoints", json={"type": "TURN_START", "event": {}})
    )
    await asyncio.sleep(0.05)

    resp = await client.post("/sessions/s5/breakpoints/bad-id/resume", json={})
    assert resp.json()["success"] is False

    # clean up
    bp_id = sessions["s5"].pending_bp_id
    assert bp_id is not None
    await client.post(f"/sessions/s5/breakpoints/{bp_id}/resume", json={})
    await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio()
async def test_get_context(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    client, sessions = client_and_sessions
    sessions["s6"] = _make_session("s6", prompt=["Be helpful."])

    resp = await client.get("/sessions/s6/context")
    assert resp.status_code == 200
    assert resp.json()["prompt"] == ["Be helpful."]


@pytest.mark.asyncio()
async def test_re_register_session_resets_events(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    """Re-registering an existing session should reset its events and status."""
    client, sessions = client_and_sessions

    # First registration
    await client.post("/sessions/re1", json={"prompt": ["Original."]})
    sessions["re1"].events.append({"event_type": "old", "event_data": {}})
    sessions["re1"].status = "done"

    # Re-register
    resp = await client.post("/sessions/re1", json={"prompt": ["Updated."]})
    assert resp.status_code == 201
    assert sessions["re1"].prompt == ["Updated."]
    assert sessions["re1"].status == "running"
    assert sessions["re1"].events == []


@pytest.mark.asyncio()
async def test_receive_event_unknown_session(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    """Posting events to a non-existent session should return 404."""
    client, _ = client_and_sessions
    resp = await client.post("/sessions/nope/events", json={"event_type": "X", "event_data": {}})
    assert resp.status_code == 404


@pytest.mark.asyncio()
async def test_list_sessions_with_data(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    client, sessions = client_and_sessions
    sessions["a"] = _make_session("a")
    sessions["b"] = _make_session("b")

    resp = await client.get("/sessions")
    assert resp.status_code == 200
    ids = {s["id"] for s in resp.json()}
    assert ids == {"a", "b"}


@pytest.mark.asyncio()
async def test_get_context_unknown_session(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_sessions
    resp = await client.get("/sessions/missing/context")
    assert resp.status_code == 404


@pytest.mark.asyncio()
async def test_ui_endpoint(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_sessions
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_session_view_helper() -> None:
    session = _Session("sv1", ["prompt1"])
    session.events.append({"event_type": "X", "event_data": {}})
    view = _session_view(session)
    assert view.id == "sv1"
    assert view.prompt == ["prompt1"]
    assert len(view.events) == 1
    assert view.status == "running"


def test_debug_server_init() -> None:
    srv = DebugServer()
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

    sessions: dict[str, _Session] = {}
    sessions["ws1"] = _Session("ws1", ["p"])
    app = _create_fastapi_app(sessions)

    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        data = ws.receive_json()
        assert data["type"] == "sessions_list"
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["id"] == "ws1"
