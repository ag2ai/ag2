# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autogen.beta.debug.client import DebugClient
from autogen.beta.debug.session import DebugSession
from autogen.beta.events.types import ModelRequest


def _make_client(**overrides: AsyncMock) -> DebugClient:
    """Return a DebugClient whose async methods are replaced with AsyncMocks."""
    client = MagicMock(spec=DebugClient)
    client.send_event = overrides.get("send_event", AsyncMock())
    client.register_stream = overrides.get("register_stream", AsyncMock())
    client.register_session = overrides.get("register_session", AsyncMock())
    client.add_stream_to_session = overrides.get("add_stream_to_session", AsyncMock())
    client.end_session = overrides.get("end_session", AsyncMock())
    return client  # type: ignore[return-value]


def _make_context(stream_id: str = "stream-123") -> MagicMock:
    context = MagicMock()
    context.stream.id = stream_id
    context.prompt = ["You are helpful."]
    context.variables = {}
    return context


def _make_session(client: DebugClient | None = None) -> DebugSession:
    session = DebugSession(name="test-session", url="http://test:8765")
    if client:
        session._client = client
    return session


@pytest.mark.asyncio()
async def test_record_event_forwards_to_server() -> None:
    send_event = AsyncMock()
    session = _make_session(_make_client(send_event=send_event))
    context = _make_context()

    event = ModelRequest(content="hello")
    await session.record_event(event, context)
    send_event.assert_awaited_once()
    args = send_event.call_args
    assert args[0][0] == "stream-123"
    assert args[0][1] == "ModelRequest"
    assert args[0][2]["content"] == "hello"


@pytest.mark.asyncio()
async def test_ensure_stream_registers_stream_and_session() -> None:
    """First record_event should register stream and session with the server."""
    register_stream = AsyncMock()
    register_session = AsyncMock()
    client = _make_client(register_stream=register_stream, register_session=register_session)
    session = _make_session(client)
    context = _make_context()

    await session.record_event(ModelRequest(content="hi"), context)

    register_stream.assert_awaited_once_with("stream-123", ["You are helpful."])
    register_session.assert_awaited_once_with(session.id, "test-session", "stream-123")


@pytest.mark.asyncio()
async def test_ensure_stream_registers_once_per_stream() -> None:
    """Multiple record_event calls with the same stream should only register once."""
    register_stream = AsyncMock()
    client = _make_client(register_stream=register_stream)
    session = _make_session(client)
    context = _make_context()

    await session.record_event(ModelRequest(content="hi"), context)
    await session.record_event(ModelRequest(content="hello"), context)

    register_stream.assert_awaited_once()


@pytest.mark.asyncio()
async def test_multiple_streams_adds_to_session() -> None:
    """Recording events from a second stream should call add_stream_to_session."""
    add_stream = AsyncMock()
    client = _make_client(add_stream_to_session=add_stream)
    session = _make_session(client)

    context_a = _make_context("stream-a")
    context_b = _make_context("stream-b")

    await session.record_event(ModelRequest(content="hi"), context_a)
    await session.record_event(ModelRequest(content="hello"), context_b)

    # First stream uses register_session, second uses add_stream_to_session
    client.register_session.assert_awaited_once()
    add_stream.assert_awaited_once_with(session.id, "stream-b")


@pytest.mark.asyncio()
async def test_multi_stream_events_routed_correctly() -> None:
    """Events from different streams should be sent with the correct stream_id."""
    send_event = AsyncMock()
    client = _make_client(send_event=send_event)
    session = _make_session(client)

    context_a = _make_context("stream-a")
    context_b = _make_context("stream-b")

    await session.record_event(ModelRequest(content="from-a"), context_a)
    await session.record_event(ModelRequest(content="from-b"), context_b)

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
        context = _make_context()
        await session.record_event(ModelRequest(content="hi"), context)
        await session.close()


@pytest.mark.asyncio()
async def test_session_name_defaults_to_random() -> None:
    session = DebugSession(url="http://test:8765")
    assert len(session.name) == 6
    assert session.name.isalpha()
