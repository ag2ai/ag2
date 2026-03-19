# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from autogen.beta.debug.server import _create_fastapi_app
from autogen.beta.debug.session import DebugSession
from autogen.beta.events.types import ModelRequest


def _make_session(session_id: str = "sess-1") -> DebugSession:
    stream = MagicMock()
    stream.send = AsyncMock()
    context = MagicMock()
    context.prompt = ["Be helpful."]
    context.variables = {}
    return DebugSession(session_id, stream=stream, context=context, prompt=["Be helpful."])


@pytest_asyncio.fixture()
async def client_and_sessions():  # type: ignore[misc]
    sessions: dict[str, DebugSession] = {}
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
async def test_get_session(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    client, sessions = client_and_sessions
    session = _make_session("s1")
    await session.record_event(ModelRequest(content="hello"))
    sessions["s1"] = session

    resp = await client.get("/sessions/s1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "s1"
    assert data["prompt"] == ["Be helpful."]
    assert len(data["events"]) == 1
    assert data["events"][0]["type"] == "ModelRequest"
    assert data["events"][0]["data"]["content"] == "hello"
    assert data["pending_bp_id"] is None


@pytest.mark.asyncio()
async def test_get_session_not_found(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_sessions
    resp = await client.get("/sessions/nope")
    assert resp.status_code == 404


@pytest.mark.asyncio()
async def test_resume_breakpoint(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    client, sessions = client_and_sessions
    session = _make_session("s2")
    sessions["s2"] = session

    done = asyncio.Event()

    async def _pause() -> None:
        await session.pause("TURN_START", ModelRequest(content="x"))
        done.set()

    task = asyncio.create_task(_pause())
    await asyncio.sleep(0.05)

    bp_id = session.pending_bp_id
    assert bp_id is not None

    resp = await client.post(f"/sessions/s2/breakpoints/{bp_id}/resume", json={})
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    await asyncio.wait_for(task, timeout=1.0)
    assert done.is_set()


@pytest.mark.asyncio()
async def test_resume_with_event_modifications(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    client, sessions = client_and_sessions
    session = _make_session("s3")
    sessions["s3"] = session
    event = ModelRequest(content="original")

    task = asyncio.create_task(session.pause("TURN_START", event))
    await asyncio.sleep(0.05)

    bp_id = session.pending_bp_id
    assert bp_id is not None

    await client.post(
        f"/sessions/s3/breakpoints/{bp_id}/resume",
        json={"event_modifications": {"content": "patched"}},
    )
    await asyncio.wait_for(task, timeout=1.0)
    assert event.content == "patched"


@pytest.mark.asyncio()
async def test_inject_known_event(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    client, sessions = client_and_sessions
    session = _make_session("s4")
    sessions["s4"] = session

    resp = await client.post(
        "/sessions/s4/inject",
        json={"event_type": "ModelRequest", "event_data": {"content": "injected"}},
    )
    assert resp.status_code == 200
    session.stream.send.assert_awaited_once()
    injected = session.stream.send.call_args[0][0]
    assert injected.content == "injected"


@pytest.mark.asyncio()
async def test_inject_unknown_event_returns_422(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    client, sessions = client_and_sessions
    sessions["s5"] = _make_session("s5")
    resp = await client.post("/sessions/s5/inject", json={"event_type": "FakeEvent"})
    assert resp.status_code == 422


@pytest.mark.asyncio()
async def test_get_and_patch_context(client_and_sessions) -> None:  # type: ignore[no-untyped-def]
    client, sessions = client_and_sessions
    session = _make_session("s6")
    sessions["s6"] = session

    resp = await client.get("/sessions/s6/context")
    assert resp.status_code == 200
    assert resp.json()["prompt"] == ["Be helpful."]

    resp = await client.patch("/sessions/s6/context", json={"variables": {"x": 42}})
    assert resp.status_code == 200
    assert session.context.variables["x"] == 42
