# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fast_depends.use import SerializerCls
from xai_sdk.chat import assistant, user
from xai_sdk.proto import chat_pb2, sample_pb2

from ag2 import Context
from ag2.config.xai import XAIConfig
from ag2.config.xai.events import XAIAssistantEvent
from ag2.events import (
    BaseEvent,
    ModelMessage,
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

from ._helpers import FakeChatServer, fake_xai_server


def _response(
    *,
    content: str = "",
    reasoning_content: str = "",
    tool_calls: list[chat_pb2.ToolCall] | None = None,
) -> chat_pb2.GetChatCompletionResponse:
    return chat_pb2.GetChatCompletionResponse(
        model="grok-4-fast",
        outputs=[
            chat_pb2.CompletionOutput(
                finish_reason=sample_pb2.REASON_STOP,
                message=chat_pb2.CompletionMessage(
                    role=chat_pb2.ROLE_ASSISTANT,
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                ),
            )
        ],
        usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    )


def _chunk(
    *,
    content: str = "",
    finish_reason: sample_pb2.FinishReason = sample_pb2.REASON_INVALID,
    usage: dict[str, int] | None = None,
) -> chat_pb2.GetChatCompletionChunk:
    return chat_pb2.GetChatCompletionChunk(
        model="grok-4-fast",
        outputs=[
            chat_pb2.CompletionOutputChunk(
                finish_reason=finish_reason,
                delta=chat_pb2.Delta(role=chat_pb2.ROLE_ASSISTANT, content=content),
            )
        ],
        usage=usage,
    )


async def _call(
    server: FakeChatServer,
    messages: list[BaseEvent],
    *,
    streaming: bool = False,
    response_schema: ResponseSchema | None = None,
) -> tuple[ModelResponse, Context]:
    client = XAIConfig(model="grok-4-fast", api_key="test", api_host=server.address, streaming=streaming).create()
    context = Context(stream=MemoryStream(persist_all=True))

    response = await client(
        messages=messages,
        context=context,
        tools=[],
        response_schema=response_schema,
        serializer=SerializerCls,
    )
    return response, context


@pytest.mark.asyncio
async def test_non_streaming_happy_path() -> None:
    async with fake_xai_server() as xai_server:
        xai_server.reply(_response(content="Hi there"))

        result, context = await _call(xai_server, [ModelRequest([TextInput("hello")])])

        assert result.provider == "xai"
        assert result.model == "grok-4-fast"
        assert result.message == ModelMessage("Hi there")
        assert result.finish_reason == "stop"
        assert result.usage == Usage(prompt_tokens=3.0, completion_tokens=5.0, total_tokens=8.0)

        events = await context.stream.history.get_events()
        assert [type(e) for e in events] == [ModelMessage, XAIAssistantEvent]


@pytest.mark.asyncio
async def test_reasoning_event_emitted() -> None:
    async with fake_xai_server() as xai_server:
        xai_server.reply(_response(content="ans", reasoning_content="think..."))

        _, context = await _call(xai_server, [ModelRequest([TextInput("hi")])])

        events = await context.stream.history.get_events()
        assert [e.content for e in events if isinstance(e, ModelReasoning)] == ["think..."]


@pytest.mark.asyncio
async def test_tool_calls_surface_in_response() -> None:
    async with fake_xai_server() as xai_server:
        xai_server.reply(
            _response(
                tool_calls=[
                    chat_pb2.ToolCall(
                        id="tc_1",
                        type=chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
                        function=chat_pb2.FunctionCall(name="get_weather", arguments='{"city": "Berlin"}'),
                    )
                ]
            )
        )

        result, _ = await _call(xai_server, [ModelRequest([TextInput("weather?")])])

        assert result.tool_calls == ToolCallsEvent([
            ToolCallEvent(id="tc_1", name="get_weather", arguments='{"city": "Berlin"}')
        ])


@pytest.mark.asyncio
async def test_streaming_accumulates_chunks() -> None:
    async with fake_xai_server() as xai_server:
        xai_server.stream(
            _chunk(content="Hello, "),
            _chunk(content="world!"),
            _chunk(
                finish_reason=sample_pb2.REASON_STOP,
                usage={"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6},
            ),
        )

        result, context = await _call(xai_server, [ModelRequest([TextInput("hi")])], streaming=True)

        assert result.message == ModelMessage("Hello, world!")
        assert result.finish_reason == "stop"
        assert result.usage == Usage(prompt_tokens=2.0, completion_tokens=4.0, total_tokens=6.0)

        events = await context.stream.history.get_events()
        assert [e.content for e in events if isinstance(e, ModelMessageChunk)] == ["Hello, ", "world!"]
        assert len([e for e in events if isinstance(e, XAIAssistantEvent)]) == 1


@pytest.mark.asyncio
async def test_xai_assistant_event_replays_in_order() -> None:
    """A stored XAIAssistantEvent is replayed between the user turns around it."""
    async with fake_xai_server() as xai_server:
        stored = _response(content="stored answer")
        xai_server.reply(_response(content="follow up"))

        await _call(
            xai_server,
            [
                ModelRequest([TextInput("first")]),
                XAIAssistantEvent(proto_bytes=stored.SerializeToString()),
                ModelRequest([TextInput("second")]),
            ],
        )

        [request] = xai_server.requests
        replayed = assistant("stored answer")
        replayed.reasoning_content = ""
        assert list(request.messages) == [user("first"), replayed, user("second")]


@pytest.mark.asyncio
async def test_response_format_sent_to_server() -> None:
    async with fake_xai_server() as xai_server:
        xai_server.reply(_response(content="42"))

        await _call(xai_server, [ModelRequest([TextInput("number?")])], response_schema=ResponseSchema(int, name="N"))

        [request] = xai_server.requests
        assert request.response_format.format_type == chat_pb2.FORMAT_TYPE_JSON_SCHEMA
