# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from autogen.beta.debug.server import create_app


@pytest_asyncio.fixture()
async def client() -> AsyncClient:  # type: ignore[misc]
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio()
async def test_create_and_get_session(client: AsyncClient) -> None:
    resp = await client.post("/sessions", json={"session_id": "sess-1"})
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "sess-1"

    resp = await client.get("/sessions/sess-1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "sess-1"
    assert data["events"] == []
    assert data["breakpoints"] == []
    assert data["status"] == "running"


@pytest.mark.asyncio()
async def test_list_sessions(client: AsyncClient) -> None:
    await client.post("/sessions", json={"session_id": "s1"})
    await client.post("/sessions", json={"session_id": "s2"})

    resp = await client.get("/sessions")
    assert resp.status_code == 200
    ids = {s["id"] for s in resp.json()}
    assert {"s1", "s2"}.issubset(ids)


@pytest.mark.asyncio()
async def test_add_event(client: AsyncClient) -> None:
    await client.post("/sessions", json={"session_id": "sess-ev"})

    resp = await client.post(
        "/sessions/sess-ev/events",
        json={"event_type": "ModelRequest", "event_data": {"content": "hello"}},
    )
    assert resp.status_code == 200
    record = resp.json()
    assert record["event_type"] == "ModelRequest"
    assert record["event_data"]["content"] == "hello"


@pytest.mark.asyncio()
async def test_breakpoint_blocks_until_resume(client: AsyncClient) -> None:
    await client.post("/sessions", json={"session_id": "sess-bp"})

    bp_result: list[dict] = []  # type: ignore[type-arg]
    bp_done = asyncio.Event()

    async def _hit_bp() -> None:
        resp = await client.post(
            "/sessions/sess-bp/breakpoints",
            json={"bp_type": "TURN_START", "event_type": "ModelRequest", "event_data": {}},
        )
        bp_result.append(resp.json())
        bp_done.set()

    task = asyncio.create_task(_hit_bp())

    # Wait for the breakpoint to be registered server-side
    await asyncio.sleep(0.05)
    assert not bp_done.is_set()

    # Get the session to find the pending breakpoint id
    sess_resp = await client.get("/sessions/sess-bp")
    bps = sess_resp.json()["breakpoints"]
    assert len(bps) == 1
    bp_id = bps[0]["id"]

    # Resume
    resume_resp = await client.post(f"/sessions/sess-bp/breakpoints/{bp_id}/resume")
    assert resume_resp.status_code == 200
    assert resume_resp.json()["success"] is True

    await asyncio.wait_for(task, timeout=2.0)
    assert bp_done.is_set()
    assert bp_result[0]["resumed"] is True


@pytest.mark.asyncio()
async def test_get_session_not_found(client: AsyncClient) -> None:
    resp = await client.get("/sessions/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.asyncio()
async def test_resume_wrong_bp_id(client: AsyncClient) -> None:
    await client.post("/sessions", json={"session_id": "sess-bad-bp"})

    # Create a hanging breakpoint
    task = asyncio.create_task(
        client.post(
            "/sessions/sess-bad-bp/breakpoints",
            json={"bp_type": "TURN_START", "event_type": "ModelRequest", "event_data": {}},
        )
    )
    await asyncio.sleep(0.05)

    resp = await client.post("/sessions/sess-bad-bp/breakpoints/wrong-id/resume")
    assert resp.status_code == 200
    assert resp.json()["success"] is False

    # Clean up
    sess_resp = await client.get("/sessions/sess-bad-bp")
    bp_id = sess_resp.json()["breakpoints"][0]["id"]
    await client.post(f"/sessions/sess-bad-bp/breakpoints/{bp_id}/resume")
    await asyncio.wait_for(task, timeout=2.0)
