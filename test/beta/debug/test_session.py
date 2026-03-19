# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from autogen.beta.debug.client import DebugClient
from autogen.beta.debug.session import DebugSession
from autogen.beta.events.types import ModelRequest


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
async def test_pause_calls_hit_breakpoint_and_returns_event() -> None:
    hit_bp = AsyncMock(return_value={})
    session = _make_session(_make_client(hit_breakpoint=hit_bp))
    event = ModelRequest(content="hi")
    result = await session.pause("TURN_START", event)
    hit_bp.assert_awaited_once()
    assert result is event


@pytest.mark.asyncio()
async def test_pause_applies_event_modifications() -> None:
    hit_bp = AsyncMock(return_value={"event_modifications": {"content": "mutated"}})
    session = _make_session(_make_client(hit_breakpoint=hit_bp))
    event = ModelRequest(content="original")
    returned = await session.pause("TURN_START", event)
    assert returned.content == "mutated"
    assert returned is event  # same object, mutated in-place


@pytest.mark.asyncio()
async def test_pause_applies_context_modifications() -> None:
    hit_bp = AsyncMock(return_value={"prompt": ["new prompt"], "variables": {"k": 1}})
    session = _make_session(_make_client(hit_breakpoint=hit_bp))
    event = ModelRequest(content="hi")
    await session.pause("TURN_START", event)
    assert session.context.prompt == ["new prompt"]
    assert session.context.variables["k"] == 1


@pytest.mark.asyncio()
async def test_pause_with_no_modifications() -> None:
    """Empty mods dict must not crash or mutate anything."""
    hit_bp = AsyncMock(return_value={})
    session = _make_session(_make_client(hit_breakpoint=hit_bp))
    event = ModelRequest(content="unchanged")
    returned = await session.pause("TOOL_CALL", event)
    assert returned.content == "unchanged"


@pytest.mark.asyncio()
async def test_inject_calls_stream_send() -> None:
    session = _make_session()
    event = ModelRequest(content="injected")
    await session.inject(event)
    session.stream.send.assert_awaited_once_with(event, session.context)
