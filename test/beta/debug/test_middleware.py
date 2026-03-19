# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from autogen.beta.debug.client import DebugClient, serialize_event
from autogen.beta.debug.middleware import DebugMiddleware
from autogen.beta.debug.session import DebugSession
from autogen.beta.events import ToolCallEvent
from autogen.beta.events.types import ModelRequest, ModelResponse
from autogen.beta.middleware.base import AgentTurn, ToolExecution


def _make_session(hit_breakpoint_return: dict | None = None) -> DebugSession:  # type: ignore[type-arg]
    """Return a DebugSession whose HTTP client is fully mocked."""
    stream = MagicMock()
    stream.send = AsyncMock()
    context = MagicMock()
    context.prompt = []
    context.variables = {}
    client = MagicMock(spec=DebugClient)
    client.send_event = AsyncMock()
    client.hit_breakpoint = AsyncMock(return_value=hit_breakpoint_return or {})
    return DebugSession("test", stream=stream, context=context, client=client)  # type: ignore[arg-type]


def _make_middleware(session: DebugSession) -> DebugMiddleware:
    return DebugMiddleware(ModelRequest(content="hi"), MagicMock(), session=session)


@pytest.mark.asyncio()
async def test_on_turn_pauses_then_calls_next() -> None:
    session = _make_session()
    session.pause = AsyncMock(wraps=session.pause)  # spy
    mw = _make_middleware(session)

    event = ModelRequest(content="hello")
    call_next: AgentTurn = AsyncMock(return_value=MagicMock(spec=ModelResponse))

    # hit_breakpoint returns immediately (mock), so on_turn should run to completion
    await mw.on_turn(call_next, event, MagicMock())

    session.pause.assert_awaited_once_with("TURN_START", event)
    call_next.assert_awaited_once()


@pytest.mark.asyncio()
async def test_on_tool_execution_pauses_then_calls_next() -> None:
    session = _make_session()
    mw = _make_middleware(session)

    tool_event = ToolCallEvent(name="my_tool", arguments='{"x": 1}')
    call_next: ToolExecution = AsyncMock(return_value=MagicMock())

    await mw.on_tool_execution(call_next, tool_event, MagicMock())
    call_next.assert_awaited_once()


@pytest.mark.asyncio()
async def test_on_turn_passes_modified_event_to_call_next() -> None:
    """Modifications returned by hit_breakpoint must reach call_next."""
    session = _make_session(hit_breakpoint_return={"event_modifications": {"content": "mutated"}})
    mw = _make_middleware(session)

    event = ModelRequest(content="original")
    received: list[object] = []

    async def _call_next(ev, ctx):  # type: ignore[no-untyped-def]
        received.append(ev)
        return MagicMock(spec=ModelResponse)

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
