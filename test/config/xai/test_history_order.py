# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from fast_depends.use import SerializerCls
from xai_sdk import AsyncClient
from xai_sdk.aio.chat import Chat
from xai_sdk.chat import Response, assistant, system, tool_result, user
from xai_sdk.proto import chat_pb2

from ag2 import Context, MemoryStream, ToolResult
from ag2.config.xai import XAIConfig
from ag2.config.xai.events import XAIAssistantEvent
from ag2.events import (
    BaseEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_tool", [False, True])
async def test_history_reaches_sdk_in_conversation_order(with_tool: bool) -> None:
    call = ToolCallEvent(id="call-1", name="weather", arguments='{"city":"Paris"}')
    first = assistant("Checking the weather" if with_tool else "Hello")
    if with_tool:
        first.tool_calls.add(
            id=call.id,
            type=chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
            function={"name": call.name, "arguments": call.arguments},
        )
    first.encrypted_content = "opaque-provider-state"
    first.reasoning_content = "Checking the available information"
    first_proto = chat_pb2.GetChatCompletionResponse(
        outputs=[
            {
                "message": {
                    "role": first.role,
                    "content": first.content[0].text,
                    "tool_calls": first.tool_calls,
                    "encrypted_content": first.encrypted_content,
                    "reasoning_content": first.reasoning_content,
                }
            }
        ],
    )
    second_proto = chat_pb2.GetChatCompletionResponse(
        outputs=[{"message": {"role": chat_pb2.ROLE_ASSISTANT, "content": "It is sunny"}}],
    )
    history: list[BaseEvent] = [
        ModelRequest([TextInput("Hello")]),
        XAIAssistantEvent(proto_bytes=first_proto.SerializeToString()),
        ModelResponse(
            message=ModelMessage(first.content[0].text),
            tool_calls=ToolCallsEvent([call] if with_tool else []),
        ),
    ]
    expected = [system("Be helpful"), user("Hello"), first]
    if with_tool:
        history.append(ToolResultsEvent([ToolResultEvent.from_call(call, ToolResult("sunny"))]))
        expected.append(tool_result("sunny", tool_call_id=call.id))
    else:
        history.append(ModelRequest([TextInput("How is the weather?")]))
        expected.append(user("How is the weather?"))
    history.extend([
        XAIAssistantEvent(proto_bytes=second_proto.SerializeToString()),
        ModelResponse(message=ModelMessage("It is sunny")),
        ModelRequest([TextInput("What about tomorrow?")]),
    ])
    second = assistant("It is sunny")
    second.reasoning_content = ""
    second.encrypted_content = ""
    expected.extend([second, user("What about tomorrow?")])

    # Keep real SDK chat creation and append; replace only the network operation.
    with patch.object(Chat, "sample", autospec=True) as sample:
        sample.return_value = Response(second_proto, 0)
        async with AsyncClient(api_key="test") as sdk:
            with patch("ag2.config.xai.xai_client.AsyncClient", return_value=sdk):
                client = XAIConfig(model="grok-4-fast", api_key="test").create()
            await client(
                messages=history,
                context=Context(stream=MemoryStream(), prompt=["Be helpful"]),
                tools=[],
                response_schema=None,
                serializer=SerializerCls,
            )

    [chat] = sample.call_args.args
    assert list(chat.messages) == expected
