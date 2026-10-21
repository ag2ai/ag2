# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fast_depends.use import SerializerCls
from xai_sdk.proto import chat_pb2, sample_pb2

from ag2 import Context, MemoryStream
from ag2.config.xai import XAIConfig
from ag2.events import ModelRequest, TextInput
from test.config._helpers import make_tool

from ._helpers import fake_xai_server

_RESPONSE = chat_pb2.GetChatCompletionResponse(
    model="grok-4-fast",
    outputs=[
        chat_pb2.CompletionOutput(
            finish_reason=sample_pb2.REASON_STOP,
            message=chat_pb2.CompletionMessage(role=chat_pb2.ROLE_ASSISTANT, content="ok"),
        )
    ],
    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
)


async def _request(choice: str, *, with_tools: bool = True) -> chat_pb2.GetCompletionsRequest:
    async with fake_xai_server() as server:
        server.reply(_RESPONSE)
        client = XAIConfig(model="grok-4-fast", api_key="test", api_host=server.address).create()
        await client(
            messages=[ModelRequest([TextInput("hello")])],
            context=Context(stream=MemoryStream()).with_tool_choice(choice),
            tools=[make_tool().schema] if with_tools else [],
            response_schema=None,
            serializer=SerializerCls,
        )
        [request] = server.requests
        return request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("auto", chat_pb2.TOOL_MODE_AUTO),
        ("none", chat_pb2.TOOL_MODE_NONE),
        ("required", chat_pb2.TOOL_MODE_REQUIRED),
    ],
)
async def test_mode_is_sent_with_the_tools(mode: str, expected: int) -> None:
    request = await _request(mode)

    assert request.tool_choice == chat_pb2.ToolChoice(mode=expected)
    assert [t.function.name for t in request.tools] == ["search_docs"]


@pytest.mark.asyncio
async def test_no_tools_sends_no_tool_choice() -> None:
    request = await _request("required", with_tools=False)

    assert not request.HasField("tool_choice")
    assert list(request.tools) == []
