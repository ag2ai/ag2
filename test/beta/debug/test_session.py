# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from autogen.beta.debug.session import DebugSession
from autogen.beta.events.types import ModelRequest


def _make_session() -> DebugSession:
    stream = MagicMock()
    stream.send = AsyncMock()
    context = MagicMock()
    context.prompt = ["You are helpful."]
    context.variables = {}
    return DebugSession("test-session", stream=stream, context=context, prompt=["You are helpful."])


@pytest.mark.asyncio()
async def test_record_event_stores_live_reference() -> None:
    session = _make_session()
    event = ModelRequest(content="hello")
    await session.record_event(event)
    assert session.events[0] is event  # live reference, not a copy


@pytest.mark.asyncio()
async def test_pause_blocks_until_resumed() -> None:
    session = _make_session()
    event = ModelRequest(content="hi")
    completed = asyncio.Event()

    async def _hit() -> None:
        await session.pause("TURN_START", event)
        completed.set()

    task = asyncio.create_task(_hit())
    await asyncio.sleep(0.01)
    assert not completed.is_set()

    bp_id = session.pending_bp_id
    assert bp_id is not None
    success = await session.resume(bp_id)
    assert success

    await asyncio.wait_for(task, timeout=1.0)
    assert completed.is_set()


@pytest.mark.asyncio()
async def test_resume_applies_event_modifications() -> None:
    session = _make_session()
    event = ModelRequest(content="original")

    task = asyncio.create_task(session.pause("TURN_START", event))
    await asyncio.sleep(0.01)

    bp_id = session.pending_bp_id
    assert bp_id is not None
    await session.resume(bp_id, event_modifications={"content": "modified"})

    returned_event = await asyncio.wait_for(task, timeout=1.0)
    assert returned_event.content == "modified"
    # Same object — mutation applied in-place
    assert returned_event is event


@pytest.mark.asyncio()
async def test_resume_applies_context_modifications() -> None:
    session = _make_session()
    event = ModelRequest(content="hi")

    task = asyncio.create_task(session.pause("TURN_START", event))
    await asyncio.sleep(0.01)

    bp_id = session.pending_bp_id
    assert bp_id is not None
    await session.resume(bp_id, prompt=["new prompt"], variables={"key": "val"})
    await asyncio.wait_for(task, timeout=1.0)

    assert session.context.prompt == ["new prompt"]
    assert session.context.variables["key"] == "val"


@pytest.mark.asyncio()
async def test_resume_wrong_id_returns_false() -> None:
    session = _make_session()
    task = asyncio.create_task(session.pause("TURN_START", ModelRequest(content="x")))
    await asyncio.sleep(0.01)

    assert await session.resume("bad-id") is False

    # Clean up
    bp_id = session.pending_bp_id
    assert bp_id is not None
    await session.resume(bp_id)
    await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio()
async def test_breakpoint_marked_resumed_after_resume() -> None:
    session = _make_session()
    task = asyncio.create_task(session.pause("TOOL_CALL", ModelRequest(content="x")))
    await asyncio.sleep(0.01)

    bp_id = session.pending_bp_id
    assert bp_id is not None
    await session.resume(bp_id)
    await asyncio.wait_for(task, timeout=1.0)

    assert session.breakpoints[0].resumed is True


@pytest.mark.asyncio()
async def test_inject_calls_stream_send() -> None:
    session = _make_session()
    event = ModelRequest(content="injected")
    await session.inject(event)
    session.stream.send.assert_awaited_once_with(event, session.context)
