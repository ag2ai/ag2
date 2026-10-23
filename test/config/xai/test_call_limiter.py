# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fast_depends.use import SerializerCls
from xai_sdk.chat import Response as XAIResponse
from xai_sdk.proto import chat_pb2

from ag2 import Agent
from ag2.config.xai import XAIAssistantEvent
from ag2.config.xai.mappers import convert_messages
from ag2.events import ModelMessage, ModelResponse, ToolCallEvent, ToolCallsEvent
from ag2.middleware import CallLimiter
from ag2.testing import TestConfig, TrackingConfig


def search() -> str:
    """Search."""
    return "found"


def record() -> str:
    """Record the result."""
    return "recorded"


def native_turn(text: str, names: list[str]) -> tuple[XAIAssistantEvent, ModelResponse]:
    calls = [ToolCallEvent(id=name, name=name) for name in names]
    proto = chat_pb2.GetChatCompletionResponse(
        outputs=[
            {
                "message": {
                    "role": chat_pb2.ROLE_ASSISTANT,
                    "content": text,
                    "encrypted_content": "opaque-state",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
                            "function": {"name": call.name, "arguments": "{}"},
                        }
                        for call in calls
                    ],
                }
            }
        ]
    )
    return (
        XAIAssistantEvent(proto_bytes=proto.SerializeToString()),
        ModelResponse(ModelMessage(text) if text else None, tool_calls=ToolCallsEvent(calls)),
    )


@pytest.mark.asyncio()
@pytest.mark.parametrize("mode", ["discard", "exempt", "text"])
@pytest.mark.parametrize("serialized", [False, True])
async def test_limited_responses_replay_only_accepted_calls(mode: str, serialized: bool) -> None:
    first_event, first_response = native_turn("answer" if mode == "text" else "", ["search", "record"] if mode == "exempt" else ["search"])
    final = native_turn("answer", [])
    original = first_event.proto_bytes
    config = TrackingConfig(TestConfig(first_event, first_response, *final))
    agent = Agent(
        "limited",
        config=config,
        tools=[search, record],
        middleware=[CallLimiter(max_llm_calls=0, exempt={"record"})],
    )

    reply = await agent.ask("Investigate.")
    history = await reply.context.stream.history.get_events()
    if serialized:
        history = [type(event).from_dict(event.to_dict()) for event in history]
    mapped = convert_messages([], history, SerializerCls)
    calls = [
        call.id
        for item in mapped
        if isinstance(item, XAIResponse)
        for output in item.proto.outputs
        for call in output.message.tool_calls
    ]
    results = [
        item.tool_call_id for item in mapped if isinstance(item, chat_pb2.Message) and item.role == chat_pb2.ROLE_TOOL
    ]

    assert reply.body == "answer"
    assert config.mock.call_count == (1 if mode == "text" else 2)
    assert calls == results == (["record"] if mode == "exempt" else [])
    assert first_event.proto_bytes == original
    assert all(
        output.message.encrypted_content == "opaque-state"
        for item in mapped
        if isinstance(item, XAIResponse)
        for output in item.proto.outputs
    )
