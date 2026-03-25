# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from autogen.beta.debug.client import DebugClient, serialize_event
from autogen.beta.debug.middleware import DebugMiddleware
from autogen.beta.debug.session import DebugSession
from autogen.beta.events import ToolCallEvent
from autogen.beta.events.types import ModelRequest
from autogen.beta.middleware.base import ToolExecution


def _make_session(hit_breakpoint_return: dict | None = None) -> DebugSession:  # type: ignore[type-arg]
    """Return a DebugSession whose HTTP client is fully mocked."""
    client = MagicMock(spec=DebugClient)
    client.send_event = AsyncMock()
    client.register_stream = AsyncMock()
    client.register_session = AsyncMock()
    client.end_session = AsyncMock()
    session = DebugSession(name="test", url="http://test:8765")
    session._client = client
    session.stream = MagicMock()
    session.stream.send = AsyncMock()
    session.context = MagicMock()
    session.context.prompt = []
    session.context.variables = {}
    return session


def _make_middleware(session: DebugSession) -> DebugMiddleware:
    return DebugMiddleware(ModelRequest(content="hi"), MagicMock(), session=session)


@pytest.mark.asyncio()
async def test_on_tool_execution_pauses_then_calls_next() -> None:
    session = _make_session()
    mw = _make_middleware(session)

    tool_event = ToolCallEvent(name="my_tool", arguments='{"x": 1}')
    call_next: ToolExecution = AsyncMock(return_value=MagicMock())

    await mw.on_tool_execution(call_next, tool_event, MagicMock())
    call_next.assert_awaited_once()


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
