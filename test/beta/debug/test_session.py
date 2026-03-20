# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from autogen.beta.debug.client import DebugClient
from autogen.beta.debug.session import DebugSession
from autogen.beta.events import ToolCallEvent
from autogen.beta.events.tool_events import ToolCallsEvent
from autogen.beta.events.types import ModelRequest, ModelResponse


def _make_client(**overrides: AsyncMock) -> DebugClient:
    """Return a DebugClient whose async methods are replaced with AsyncMocks."""
    client = MagicMock(spec=DebugClient)
    client.send_event = overrides.get("send_event", AsyncMock())
    client.hit_breakpoint = overrides.get("hit_breakpoint", AsyncMock(return_value={}))
    return client  # type: ignore[return-value]


def _make_session(client: DebugClient | None = None) -> DebugSession:
    stream = MagicMock()
    stream.send = AsyncMock()
    context = MagicMock()
    context.prompt = ["You are helpful."]
    context.variables = {}
    return DebugSession("test-session", stream=stream, context=context, client=client or _make_client())


@pytest.mark.asyncio()
async def test_record_event_forwards_to_server() -> None:
    send_event = AsyncMock()
    session = _make_session(_make_client(send_event=send_event))
    event = ModelRequest(content="hello")
    await session.record_event(event)
    send_event.assert_awaited_once()
    args = send_event.call_args
    assert args[0][0] == "test-session"
    assert args[0][1] == "ModelRequest"
    assert args[0][2]["content"] == "hello"


@pytest.mark.asyncio()
async def test_inject_calls_stream_send() -> None:
    session = _make_session()
    event = ModelRequest(content="injected")
    await session.inject(event)
    session.stream.send.assert_awaited_once_with(event, session.context)


@pytest.mark.asyncio()
async def test_replay_events_emits_tool_calls() -> None:
    """replay_events should emit a synthetic ToolCallsEvent for ModelResponse with tool_calls."""
    send_event = AsyncMock()
    session = _make_session(_make_client(send_event=send_event))

    tool_calls = ToolCallsEvent(calls=[ToolCallEvent(name="fn", arguments="{}")])
    response = ModelResponse(tool_calls=tool_calls)
    request = ModelRequest(content="hi")

    await session.replay_events([request, response])

    # Should have 3 calls: request, response, then the synthetic tool_calls event
    assert send_event.await_count == 3
    event_types = [call.args[1] for call in send_event.call_args_list]
    assert event_types == ["ModelRequest", "ModelResponse", "ToolCallsEvent"]


@pytest.mark.asyncio()
async def test_replay_events_skips_empty_tool_calls() -> None:
    """replay_events should NOT emit synthetic ToolCallsEvent when tool_calls is empty."""
    send_event = AsyncMock()
    session = _make_session(_make_client(send_event=send_event))

    response = ModelResponse()  # no tool_calls
    await session.replay_events([response])

    assert send_event.await_count == 1
    assert send_event.call_args.args[1] == "ModelResponse"
