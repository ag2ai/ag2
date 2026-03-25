# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

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
    client.register_stream = overrides.get("register_stream", AsyncMock())
    client.register_session = overrides.get("register_session", AsyncMock())
    client.add_stream_to_session = overrides.get("add_stream_to_session", AsyncMock())
    client.end_session = overrides.get("end_session", AsyncMock())
    return client  # type: ignore[return-value]


def _make_stream(stream_id: str = "stream-123") -> MagicMock:
    stream = MagicMock()
    stream.id = stream_id
    stream.send = AsyncMock()
    stream.subscribe = MagicMock()
    stream.history = MagicMock()
    stream.history.get_events = AsyncMock(return_value=[])
    return stream


def _make_context() -> MagicMock:
    context = MagicMock()
    context.prompt = ["You are helpful."]
    context.variables = {}
    return context


def _make_session(client: DebugClient | None = None) -> DebugSession:
    session = DebugSession(name="test-session", url="http://test:8765")
    if client:
        session._client = client
    return session


async def _attach_session(
    session: DebugSession,
    stream: MagicMock | None = None,
    context: MagicMock | None = None,
) -> None:
    """Attach session to a stream so it can record events."""
    stream = stream or _make_stream()
    context = context or _make_context()
    await session._attach(stream, context)


@pytest.mark.asyncio()
async def test_record_event_forwards_to_server() -> None:
    send_event = AsyncMock()
    session = _make_session(_make_client(send_event=send_event))
    await _attach_session(session)

    event = ModelRequest(content="hello")
    await session.record_event(event)
    send_event.assert_awaited_once()
    args = send_event.call_args
    assert args[0][0] == "stream-123"
    assert args[0][1] == "ModelRequest"
    assert args[0][2]["content"] == "hello"


@pytest.mark.asyncio()
async def test_inject_calls_stream_send() -> None:
    session = _make_session(_make_client())
    stream = _make_stream()
    context = _make_context()
    await _attach_session(session, stream, context)

    event = ModelRequest(content="injected")
    await session.inject(event)
    stream.send.assert_awaited_once_with(event, context)


@pytest.mark.asyncio()
async def test_replay_events_emits_tool_calls() -> None:
    """replay_events should emit a synthetic ToolCallsEvent for ModelResponse with tool_calls."""
    send_event = AsyncMock()
    session = _make_session(_make_client(send_event=send_event))
    await _attach_session(session)

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
    await _attach_session(session)

    response = ModelResponse()  # no tool_calls
    await session.replay_events([response])

    assert send_event.await_count == 1
    assert send_event.call_args.args[1] == "ModelResponse"


@pytest.mark.asyncio()
async def test_attach_registers_stream_and_session() -> None:
    """First attach should register stream and session with the server."""
    register_stream = AsyncMock()
    register_session = AsyncMock()
    client = _make_client(register_stream=register_stream, register_session=register_session)
    session = _make_session(client)
    stream = _make_stream()
    context = _make_context()

    await session._attach(stream, context)

    register_stream.assert_awaited_once_with("stream-123", ["You are helpful."])
    register_session.assert_awaited_once_with(session.id, "test-session", "stream-123")


@pytest.mark.asyncio()
async def test_attach_subscribes_once() -> None:
    """Multiple attaches with the same stream should only subscribe once."""
    client = _make_client()
    session = _make_session(client)
    stream = _make_stream()
    context = _make_context()

    await session._attach(stream, context)
    await session._attach(stream, context)

    assert stream.subscribe.call_count == 1


@pytest.mark.asyncio()
async def test_attach_multiple_streams() -> None:
    """Attaching a second stream should call add_stream_to_session."""
    add_stream = AsyncMock()
    client = _make_client(add_stream_to_session=add_stream)
    session = _make_session(client)

    stream_a = _make_stream("stream-a")
    stream_b = _make_stream("stream-b")
    context = _make_context()

    await session._attach(stream_a, context)
    await session._attach(stream_b, context)

    # First stream uses register_session, second uses add_stream_to_session
    client.register_session.assert_awaited_once()
    add_stream.assert_awaited_once_with(session.id, "stream-b")
    assert session.stream_ids == ["stream-a", "stream-b"]


@pytest.mark.asyncio()
async def test_multi_stream_events_routed_correctly() -> None:
    """Events from different streams should be sent with the correct stream_id."""
    send_event = AsyncMock()
    client = _make_client(send_event=send_event)
    session = _make_session(client)

    stream_a = _make_stream("stream-a")
    stream_b = _make_stream("stream-b")
    context = _make_context()

    await session._attach(stream_a, context)
    await session._attach(stream_b, context)

    # Get the subscriber callbacks that were registered
    assert stream_a.subscribe.call_count == 1
    assert stream_b.subscribe.call_count == 1

    callback_a = stream_a.subscribe.call_args[0][0]
    callback_b = stream_b.subscribe.call_args[0][0]

    await callback_a(ModelRequest(content="from-a"))
    await callback_b(ModelRequest(content="from-b"))

    assert send_event.await_count == 2
    assert send_event.call_args_list[0][0][0] == "stream-a"
    assert send_event.call_args_list[1][0][0] == "stream-b"


@pytest.mark.asyncio()
async def test_close_calls_end_session() -> None:
    end_session = AsyncMock()
    session = _make_session(_make_client(end_session=end_session))
    await session.close()
    end_session.assert_awaited_once_with(session.id)


@pytest.mark.asyncio()
async def test_context_manager() -> None:
    end_session = AsyncMock()
    client = _make_client(end_session=end_session)
    async with DebugSession(name="ctx-test", url="http://test:8765") as session:
        session._client = client
    end_session.assert_awaited_once()


@pytest.mark.asyncio()
async def test_session_without_url_is_noop() -> None:
    """A session created without a URL and no env var should be a silent no-op."""
    with patch.dict("os.environ", {}, clear=True):
        session = DebugSession(name="no-url")
        assert session._client is None
        # record_event should not raise
        await session.record_event(ModelRequest(content="hi"))
        await session.close()


@pytest.mark.asyncio()
async def test_session_name_defaults_to_random() -> None:
    session = DebugSession(url="http://test:8765")
    assert len(session.name) == 6
    assert session.name.isalpha()
