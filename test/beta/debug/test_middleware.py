# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from autogen.beta.debug.client import DebugClient, serialize_event
from autogen.beta.debug.middleware import DebugMiddleware
from autogen.beta.events import BaseEvent, ToolCallEvent
from autogen.beta.events.types import ModelRequest, ModelResponse
from autogen.beta.middleware.base import AgentTurn, ToolExecution


def _make_middleware(mock_client: DebugClient, session_id: str = "test-session") -> DebugMiddleware:
    event = ModelRequest(content="hi")
    context = MagicMock()
    return DebugMiddleware(event, context, client=mock_client, session_id=session_id)


@pytest.mark.asyncio()
async def test_on_turn_calls_hit_breakpoint() -> None:
    mock_client = MagicMock(spec=DebugClient)
    mock_client.hit_breakpoint = AsyncMock()

    mw = _make_middleware(mock_client, session_id="s1")

    call_next: AgentTurn = AsyncMock(return_value=MagicMock(spec=ModelResponse))
    event = ModelRequest(content="hello")
    context = MagicMock()

    await mw.on_turn(call_next, event, context)

    mock_client.hit_breakpoint.assert_awaited_once_with("s1", "TURN_START", event)
    call_next.assert_awaited_once_with(event, context)


@pytest.mark.asyncio()
async def test_on_tool_execution_calls_hit_breakpoint() -> None:
    mock_client = MagicMock(spec=DebugClient)
    mock_client.hit_breakpoint = AsyncMock()

    mw = _make_middleware(mock_client, session_id="s2")

    call_next: ToolExecution = AsyncMock(return_value=MagicMock())
    tool_event = ToolCallEvent(name="my_tool", arguments='{"x": 1}')
    context = MagicMock()

    await mw.on_tool_execution(call_next, tool_event, context)

    mock_client.hit_breakpoint.assert_awaited_once_with("s2", "TOOL_CALL", tool_event)
    call_next.assert_awaited_once_with(tool_event, context)


def test_serialize_event_basic() -> None:
    event = ModelRequest(content="test")
    result = serialize_event(event)
    assert result["type"] == "ModelRequest"
    assert result["data"]["content"] == "test"


def test_serialize_event_tool_call() -> None:
    event = ToolCallEvent(name="my_tool", arguments='{"x": 1}')
    result = serialize_event(event)
    assert result["type"] == "ToolCallEvent"
    assert result["data"]["name"] == "my_tool"
    assert result["data"]["arguments"] == '{"x": 1}'


def test_serialize_event_skips_private_fields() -> None:
    event = ToolCallEvent(name="my_tool", arguments='{"x": 1}')
    result = serialize_event(event)
    # _serialized_arguments is private and should not appear
    assert "_serialized_arguments" not in result["data"]
