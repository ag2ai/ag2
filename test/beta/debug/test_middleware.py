# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from autogen.beta.debug.client import serialize_event
from autogen.beta.debug.middleware import DebugMiddleware
from autogen.beta.debug.session import DebugSession
from autogen.beta.events import ToolCallEvent
from autogen.beta.events.types import ModelRequest, ModelResponse
from autogen.beta.middleware.base import AgentTurn, ToolExecution


def _make_session() -> DebugSession:
    stream = MagicMock()
    stream.send = AsyncMock()
    context = MagicMock()
    context.prompt = []
    context.variables = {}
    return DebugSession("test", stream=stream, context=context)


def _make_middleware(session: DebugSession) -> DebugMiddleware:
    return DebugMiddleware(ModelRequest(content="hi"), MagicMock(), session=session)


@pytest.mark.asyncio()
async def test_on_turn_pauses_then_calls_next() -> None:
    session = _make_session()
    session.pause = AsyncMock(wraps=session.pause)  # spy
    mw = _make_middleware(session)

    event = ModelRequest(content="hello")
    call_next: AgentTurn = AsyncMock(return_value=MagicMock(spec=ModelResponse))

    import asyncio

    # Resume immediately from a concurrent task
    async def _resume() -> None:
        await asyncio.sleep(0.02)
        bp_id = session.pending_bp_id
        assert bp_id is not None
        await session.resume(bp_id)

    asyncio.create_task(_resume())
    await mw.on_turn(call_next, event, MagicMock())

    session.pause.assert_awaited_once_with("TURN_START", event)
    call_next.assert_awaited_once()


@pytest.mark.asyncio()
async def test_on_tool_execution_pauses_then_calls_next() -> None:
    session = _make_session()
    mw = _make_middleware(session)

    tool_event = ToolCallEvent(name="my_tool", arguments='{"x": 1}')
    call_next: ToolExecution = AsyncMock(return_value=MagicMock())

    import asyncio

    async def _resume() -> None:
        await asyncio.sleep(0.02)
        bp_id = session.pending_bp_id
        assert bp_id is not None
        await session.resume(bp_id)

    asyncio.create_task(_resume())
    await mw.on_tool_execution(call_next, tool_event, MagicMock())
    call_next.assert_awaited_once()


@pytest.mark.asyncio()
async def test_on_turn_passes_modified_event_to_call_next() -> None:
    """Modifications made at resume time must reach call_next."""
    session = _make_session()
    mw = _make_middleware(session)

    event = ModelRequest(content="original")
    received: list[object] = []

    async def _call_next(ev, ctx):  # type: ignore[no-untyped-def]
        received.append(ev)
        return MagicMock(spec=ModelResponse)

    import asyncio

    async def _resume() -> None:
        await asyncio.sleep(0.02)
        bp_id = session.pending_bp_id
        assert bp_id is not None
        await session.resume(bp_id, event_modifications={"content": "mutated"})

    asyncio.create_task(_resume())
    await mw.on_turn(_call_next, event, MagicMock())

    assert received[0].content == "mutated"  # type: ignore[union-attr]


# ── Serialisation tests ────────────────────────────────────────────────────

def test_serialize_event_basic() -> None:
    event = ModelRequest(content="test")
    result = serialize_event(event)
    assert result["type"] == "ModelRequest"
    assert result["data"]["content"] == "test"


def test_serialize_event_no_private_fields() -> None:
    event = ToolCallEvent(name="fn", arguments='{"x": 1}')
    result = serialize_event(event)
    assert "_serialized_arguments" not in result["data"]


def test_serialize_event_tool_call() -> None:
    event = ToolCallEvent(name="my_tool", arguments='{"x": 1}')
    result = serialize_event(event)
    assert result["type"] == "ToolCallEvent"
    assert result["data"]["name"] == "my_tool"
