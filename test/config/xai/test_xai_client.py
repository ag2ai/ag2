# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fast_depends.use import SerializerCls
from xai_sdk.chat import chat_pb2
from xai_sdk.proto import usage_pb2

from ag2 import Context
from ag2.config.xai import XAIConfig
from ag2.config.xai.events import XAIAssistantEvent
from ag2.events import (
    ModelMessageChunk,
    ModelReasoning,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    Usage,
)
from ag2.response import ResponseSchema
from ag2.stream import MemoryStream


def _fake_response(
    *,
    content: str = "",
    reasoning_content: str = "",
    tool_calls: list[object] | None = None,
    model: str = "grok-4-fast",
    finish_reason: str = "FINISH_REASON_STOP",
    usage: object | None = None,
) -> SimpleNamespace:
    proto = chat_pb2.GetChatCompletionResponse(model=model)
    return SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        usage=usage or usage_pb2.SamplingUsage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
        proto=proto,
    )


def _fake_tool_call(
    call_id: str,
    name: str,
    arguments: str,
    *,
    type: int = chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type=type,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _make_client_with_stub_chat(stub_chat: MagicMock) -> tuple[object, Context]:
    with patch("ag2.config.xai.xai_client.AsyncClient") as mock_async_client:
        instance = MagicMock()
        instance.chat.create.return_value = stub_chat
        mock_async_client.return_value = instance

        client = XAIConfig(model="grok-4-fast", api_key="t").create()

    stream = MemoryStream(persist_all=True)
    return client, Context(stream=stream)


@pytest.mark.asyncio
async def test_non_streaming_happy_path() -> None:
    stub_chat = MagicMock()
    stub_chat.sample = AsyncMock(return_value=_fake_response(content="Hi there"))

    client, context = _make_client_with_stub_chat(stub_chat)

    result = await client(
        messages=[ModelRequest([TextInput("hello")])],
        context=context,
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )

    assert isinstance(result, ModelResponse)
    assert result.provider == "xai"
    assert result.model == "grok-4-fast"
    assert result.message is not None
    assert result.message.content == "Hi there"
    assert result.finish_reason == "stop"
    assert result.usage == Usage(prompt_tokens=3.0, completion_tokens=5.0, total_tokens=8.0)

    events = await context.stream.history.get_events()
    kinds = [type(e).__name__ for e in events]
    assert "ModelMessage" in kinds
    assert "XAIAssistantEvent" in kinds


@pytest.mark.asyncio
async def test_reasoning_event_emitted() -> None:
    stub_chat = MagicMock()
    stub_chat.sample = AsyncMock(return_value=_fake_response(content="ans", reasoning_content="think..."))

    client, context = _make_client_with_stub_chat(stub_chat)

    await client(
        messages=[ModelRequest([TextInput("hi")])],
        context=context,
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )

    events = await context.stream.history.get_events()
    reasoning = [e for e in events if isinstance(e, ModelReasoning)]
    assert len(reasoning) == 1
    assert reasoning[0].content == "think..."


@pytest.mark.asyncio
async def test_tool_calls_surface_in_response() -> None:
    stub_chat = MagicMock()
    stub_chat.sample = AsyncMock(
        return_value=_fake_response(
            content="",
            tool_calls=[_fake_tool_call("tc_1", "get_weather", '{"city": "Berlin"}')],
        )
    )

    client, context = _make_client_with_stub_chat(stub_chat)

    result = await client(
        messages=[ModelRequest([TextInput("weather?")])],
        context=context,
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )

    assert result.tool_calls == ToolCallsEvent([
        ToolCallEvent(id="tc_1", name="get_weather", arguments='{"city": "Berlin"}')
    ])


@pytest.mark.asyncio
async def test_streaming_accumulates_chunks() -> None:
    chunks = [
        _fake_response(content="Hello, ", usage=None, finish_reason=""),
        _fake_response(content="world!", usage=None, finish_reason=""),
        _fake_response(
            content="",
            usage=usage_pb2.SamplingUsage(prompt_tokens=2, completion_tokens=4, total_tokens=6),
        ),
    ]
    final_response = _fake_response(
        content="Hello, world!",
        usage=usage_pb2.SamplingUsage(prompt_tokens=2, completion_tokens=4, total_tokens=6),
    )

    async def fake_stream() -> object:
        for chunk in chunks:
            yield final_response, chunk

    stub_chat = MagicMock()
    stub_chat.stream = MagicMock(return_value=fake_stream())

    with patch("ag2.config.xai.xai_client.AsyncClient") as mock_async_client:
        instance = MagicMock()
        instance.chat.create.return_value = stub_chat
        mock_async_client.return_value = instance

        client = XAIConfig(model="grok-4-fast", api_key="t", streaming=True).create()

    context = Context(stream=MemoryStream(persist_all=True))

    result = await client(
        messages=[ModelRequest([TextInput("hi")])],
        context=context,
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )

    assert result.message is not None
    assert result.message.content == "Hello, world!"
    assert result.finish_reason == "stop"
    assert result.usage == Usage(prompt_tokens=2.0, completion_tokens=4.0, total_tokens=6.0)

    events = await context.stream.history.get_events()
    chunk_events = [e for e in events if isinstance(e, ModelMessageChunk)]
    assert [e.content for e in chunk_events] == ["Hello, ", "world!"]

    assistant_events = [e for e in events if isinstance(e, XAIAssistantEvent)]
    assert len(assistant_events) == 1


@pytest.mark.asyncio
async def test_xai_assistant_event_replays_through_chat_append() -> None:
    """Pre-existing XAIAssistantEvent in history should round-trip via chat.append()."""
    proto = chat_pb2.GetChatCompletionResponse()
    historical = XAIAssistantEvent(proto_bytes=proto.SerializeToString())

    stub_chat = MagicMock()
    stub_chat.sample = AsyncMock(return_value=_fake_response(content="follow up"))

    client, context = _make_client_with_stub_chat(stub_chat)

    await client(
        messages=[ModelRequest([TextInput("first")]), historical],
        context=context,
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )

    # chat.append should be called exactly once with the replay Response
    assert stub_chat.append.call_count == 1


@pytest.mark.asyncio
async def test_response_format_passed_to_chat_create() -> None:
    stub_chat = MagicMock()
    stub_chat.sample = AsyncMock(return_value=_fake_response(content="42"))

    with patch("ag2.config.xai.xai_client.AsyncClient") as mock_async_client:
        instance = MagicMock()
        instance.chat.create.return_value = stub_chat
        mock_async_client.return_value = instance

        client = XAIConfig(model="grok-4-fast", api_key="t").create()

    context = Context(stream=MemoryStream(persist_all=True))

    await client(
        messages=[ModelRequest([TextInput("number?")])],
        context=context,
        tools=[],
        response_schema=ResponseSchema(int, name="N"),
        serializer=SerializerCls,
    )

    create_kwargs = instance.chat.create.call_args.kwargs
    assert "response_format" in create_kwargs
    assert isinstance(create_kwargs["response_format"], chat_pb2.ResponseFormat)
    assert create_kwargs["response_format"].format_type == chat_pb2.FORMAT_TYPE_JSON_SCHEMA
