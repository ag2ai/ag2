# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from dashscope.api_entities.dashscope_response import (
    Choice,
    Message,
    MultiModalConversationOutput,
    MultiModalConversationResponse,
    MultiModalConversationUsage,
)
from fast_depends.pydantic import PydanticSerializer

from ag2 import Context
from ag2.config import DashScopeConfig
from ag2.events import ModelMessageChunk, ModelRequest, TextInput, Usage
from ag2.stream import MemoryStream


def _response(content: Any, *, finish_reason: str | None = "stop") -> MultiModalConversationResponse:
    """Build the response the SDK's `from_api_response` does, so `Choice` is the SDK's dict subclass."""
    return MultiModalConversationResponse(
        status_code=200,
        output=MultiModalConversationOutput(
            choices=[
                Choice(
                    finish_reason=finish_reason,  # type: ignore[arg-type]  # the SDK's stub says `str` over a `None` default; a chunk mid-stream carries none
                    message=Message(role="assistant", content=content),
                )
            ],
        ),
        usage=MultiModalConversationUsage(input_tokens=3, output_tokens=5, total_tokens=8),
    )


async def _stream(*chunks: MultiModalConversationResponse) -> AsyncGenerator[MultiModalConversationResponse]:
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_non_streaming_reads_the_sdk_response() -> None:
    client = DashScopeConfig(model="qwen-plus", api_key="t").create()
    context = Context(stream=MemoryStream())

    with patch(
        "ag2.config.dashscope.dashscope_client.AioMultiModalConversation.call",
        AsyncMock(return_value=_response([{"text": "Hi there"}])),
    ):
        result = await client(
            messages=[ModelRequest([TextInput("hello")])],
            context=context,
            tools=[],
            response_schema=None,
            serializer=PydanticSerializer(),
        )

    assert result.message is not None
    assert result.message.content == "Hi there"
    assert result.finish_reason == "stop"
    assert result.usage == Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8)


@pytest.mark.asyncio
async def test_streaming_reads_the_sdk_async_stream() -> None:
    client = DashScopeConfig(model="qwen-plus", api_key="t", streaming=True).create()
    stream = MemoryStream(persist_all=True)
    context = Context(stream=stream)

    with patch(
        "ag2.config.dashscope.dashscope_client.AioMultiModalConversation.call",
        AsyncMock(return_value=_stream(_response("Hello, ", finish_reason=None), _response("world!"))),
    ):
        result = await client(
            messages=[ModelRequest([TextInput("hello")])],
            context=context,
            tools=[],
            response_schema=None,
            serializer=PydanticSerializer(),
        )

    assert result.message is not None
    assert result.message.content == "Hello, world!"
    assert result.finish_reason == "stop"
    chunks = [e.content for e in await stream.history.get_events() if isinstance(e, ModelMessageChunk)]
    assert chunks == ["Hello, ", "world!"]


def test_config_has_no_files_client() -> None:
    with pytest.raises(NotImplementedError, match="does not support Files API"):
        DashScopeConfig(model="qwen-plus").create_files_client()
