# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autogen.beta.debug.client import DebugClient, serialize_event
from autogen.beta.debug.middleware import DebugMiddleware
from autogen.beta.debug.session import DEBUG_SESSION_VAR, DebugSession
from autogen.beta.events import ToolCallEvent
from autogen.beta.events.types import ModelMessage, ModelRequest, ModelResponse
from autogen.beta.middleware.base import ToolExecution


def _make_session() -> DebugSession:
    """Return a DebugSession whose HTTP client is fully mocked."""
    client = MagicMock(spec=DebugClient)
    client.send_event = AsyncMock()
    client.register_stream = AsyncMock()
    client.register_session = AsyncMock()
    client.add_stream_to_session = AsyncMock()
    client.end_session = AsyncMock()
    session = DebugSession(name="test", url="http://test:8765")
    session._client = client
    return session


def _make_context(session: DebugSession | None = None) -> MagicMock:
    context = MagicMock()
    context.stream.id = "stream-123"
    context.prompt = ["Be helpful."]
    context.variables = {}
    if session:
        context.variables[DEBUG_SESSION_VAR] = session
    return context


def _make_middleware(session: DebugSession | None = None) -> DebugMiddleware:
    session = session or _make_session()
    context = _make_context(session)
    return DebugMiddleware(ModelRequest(content="hi"), context)


@pytest.mark.asyncio()
async def test_on_turn_records_request_and_response() -> None:
    session = _make_session()
    mw = _make_middleware(session)
    response = ModelResponse(message=ModelMessage(content="hello"))
    call_next = AsyncMock(return_value=response)

    context = _make_context(session)
    result = await mw.on_turn(call_next, ModelRequest(content="hi"), context)

    assert result is response
    call_next.assert_awaited_once()
    # Should have recorded the request event
    assert session._client.send_event.await_count >= 1


@pytest.mark.asyncio()
async def test_on_tool_execution_records_event_and_result() -> None:
    session = _make_session()
    mw = _make_middleware(session)

    tool_event = ToolCallEvent(name="my_tool", arguments='{"x": 1}')
    tool_result = MagicMock()
    call_next: ToolExecution = AsyncMock(return_value=tool_result)

    context = _make_context(session)
    result = await mw.on_tool_execution(call_next, tool_event, context)

    assert result is tool_result
    call_next.assert_awaited_once()
    # Should have recorded both the tool call and the result
    assert session._client.send_event.await_count >= 2


@pytest.mark.asyncio()
async def test_auto_session_created_when_env_var_set() -> None:
    context = MagicMock()
    context.stream.id = "stream-123"
    context.prompt = []
    context.variables = {}

    with patch.dict("os.environ", {"AG2_DEBUG_SERVER_URL": "http://localhost:8765"}):
        mw = DebugMiddleware(ModelRequest(content="hi"), context)

    assert mw._enabled is True
    assert mw._auto_session is True
    assert DEBUG_SESSION_VAR in context.variables


@pytest.mark.asyncio()
async def test_disabled_when_no_env_var_and_no_session() -> None:
    context = MagicMock()
    context.stream.id = "stream-123"
    context.prompt = []
    context.variables = {}

    with patch.dict("os.environ", {}, clear=True):
        mw = DebugMiddleware(ModelRequest(content="hi"), context)

    assert mw._enabled is False


@pytest.mark.asyncio()
async def test_auto_session_closed_after_turn() -> None:
    context = MagicMock()
    context.stream.id = "stream-123"
    context.prompt = []
    context.variables = {}

    with patch.dict("os.environ", {"AG2_DEBUG_SERVER_URL": "http://test:8765"}):
        mw = DebugMiddleware(ModelRequest(content="hi"), context)

    # Mock the auto-created session's client
    mw._session._client = MagicMock(spec=DebugClient)
    mw._session._client.send_event = AsyncMock()
    mw._session._client.register_stream = AsyncMock()
    mw._session._client.register_session = AsyncMock()
    mw._session._client.end_session = AsyncMock()

    response = ModelResponse(message=ModelMessage(content="bye"))
    call_next = AsyncMock(return_value=response)

    await mw.on_turn(call_next, ModelRequest(content="hi"), context)

    mw._session._client.end_session.assert_awaited_once()


@pytest.mark.asyncio()
async def test_explicit_session_not_closed_after_turn() -> None:
    session = _make_session()
    mw = _make_middleware(session)

    response = ModelResponse(message=ModelMessage(content="bye"))
    call_next = AsyncMock(return_value=response)
    context = _make_context(session)

    await mw.on_turn(call_next, ModelRequest(content="hi"), context)

    session._client.end_session.assert_not_awaited()


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


@pytest.mark.asyncio()
async def test_on_llm_call_enabled() -> None:
    """on_llm_call should forward the result and record it."""
    session = _make_session()
    mw = _make_middleware(session)
    response = ModelResponse(message=ModelMessage(content="reply"))
    call_next = AsyncMock(return_value=response)
    context = _make_context(session)

    result = await mw.on_llm_call(call_next, [ModelRequest(content="hi")], context)

    assert result is response
    call_next.assert_awaited_once()
    # Should have recorded the response event
    assert session._client.send_event.await_count >= 1


@pytest.mark.asyncio()
async def test_on_llm_call_disabled() -> None:
    """on_llm_call should pass through when disabled."""
    context = MagicMock()
    context.stream.id = "stream-123"
    context.prompt = []
    context.variables = {}

    with patch.dict("os.environ", {}, clear=True):
        mw = DebugMiddleware(ModelRequest(content="hi"), context)

    assert mw._enabled is False
    response = ModelResponse(message=ModelMessage(content="reply"))
    call_next = AsyncMock(return_value=response)

    result = await mw.on_llm_call(call_next, [ModelRequest(content="hi")], context)
    assert result is response
    call_next.assert_awaited_once()


@pytest.mark.asyncio()
async def test_on_tool_execution_disabled() -> None:
    """on_tool_execution should pass through when disabled."""
    context = MagicMock()
    context.stream.id = "stream-123"
    context.prompt = []
    context.variables = {}

    with patch.dict("os.environ", {}, clear=True):
        mw = DebugMiddleware(ModelRequest(content="hi"), context)

    assert mw._enabled is False
    tool_event = ToolCallEvent(name="my_tool", arguments='{"x": 1}')
    tool_result = MagicMock()
    call_next: ToolExecution = AsyncMock(return_value=tool_result)

    result = await mw.on_tool_execution(call_next, tool_event, context)
    assert result is tool_result
    call_next.assert_awaited_once()


@pytest.mark.asyncio()
async def test_on_turn_disabled() -> None:
    """on_turn should pass through when disabled."""
    context = MagicMock()
    context.stream.id = "stream-123"
    context.prompt = []
    context.variables = {}

    with patch.dict("os.environ", {}, clear=True):
        mw = DebugMiddleware(ModelRequest(content="hi"), context)

    assert mw._enabled is False
    response = ModelResponse(message=ModelMessage(content="bye"))
    call_next = AsyncMock(return_value=response)

    result = await mw.on_turn(call_next, ModelRequest(content="hi"), context)
    assert result is response
    call_next.assert_awaited_once()


@pytest.mark.asyncio()
async def test_on_turn_closes_auto_session_on_exception() -> None:
    """Auto-session should be closed even when call_next raises."""
    context = MagicMock()
    context.stream.id = "stream-123"
    context.prompt = []
    context.variables = {}

    with patch.dict("os.environ", {"AG2_DEBUG_SERVER_URL": "http://test:8765"}):
        mw = DebugMiddleware(ModelRequest(content="hi"), context)

    # Mock the auto-created session's client
    mw._session._client = MagicMock(spec=DebugClient)
    mw._session._client.send_event = AsyncMock()
    mw._session._client.register_stream = AsyncMock()
    mw._session._client.register_session = AsyncMock()
    mw._session._client.end_session = AsyncMock()

    call_next = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await mw.on_turn(call_next, ModelRequest(content="hi"), context)

    # Auto-session should still be closed via the finally block
    mw._session._client.end_session.assert_awaited_once()
