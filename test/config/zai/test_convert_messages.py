# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fast_depends.use import SerializerCls

from ag2 import ToolResult
from ag2.compact import CompactionSummary
from ag2.config.zai.mappers import convert_messages
from ag2.events import (
    DataInput,
    ImageInput,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolErrorEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from ag2.exceptions import UnsupportedInputError


def test_user_text_input() -> None:
    result = convert_messages([], [ModelRequest([TextInput("hello")])], SerializerCls)

    assert result == [{"role": "user", "content": "hello"}]


def test_data_input_serialization() -> None:
    data = {"category": "books", "limit": 3}
    result = convert_messages([], [ModelRequest([DataInput(data)])], SerializerCls)

    assert result == [{"role": "user", "content": SerializerCls.encode(data).decode()}]


def test_system_prompt() -> None:
    result = convert_messages(["You are helpful.", "Be brief."], [], SerializerCls)

    assert result == [{"role": "system", "content": "You are helpful.\nBe brief."}]


def test_assistant_text_and_tool_call() -> None:
    response = ModelResponse(
        message=ModelMessage("Let me check."),
        tool_calls=ToolCallsEvent([ToolCallEvent(id="tc_1", name="list_items", arguments='{"category": "books"}')]),
    )
    result = convert_messages([], [response], SerializerCls)

    assert result == [
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "list_items", "arguments": '{"category": "books"}'},
                }
            ],
        }
    ]


def test_assistant_without_message_has_explicit_null_content() -> None:
    response = ModelResponse(message=None, tool_calls=ToolCallsEvent([]))

    result = convert_messages([], [response], SerializerCls)

    assert result == [{"role": "assistant", "content": None}]


def test_tool_result_and_error_result() -> None:
    events = [
        ToolResultsEvent(
            results=[
                ToolResultEvent(parent_id="tc_1", name="list_items", result=ToolResult("apple")),
                ToolErrorEvent(parent_id="tc_2", name="fail", error=ValueError("boom"), result=ToolResult("boom")),
            ]
        )
    ]
    result = convert_messages([], events, SerializerCls)

    assert result == [
        {"role": "tool", "tool_call_id": "tc_1", "content": "apple"},
        {"role": "tool", "tool_call_id": "tc_2", "content": "boom"},
    ]


def test_loose_tool_result() -> None:
    result = convert_messages(
        [],
        [ToolResultEvent(parent_id="tc_1", name="list_items", result=ToolResult("ok"))],
        SerializerCls,
    )

    assert result == [{"role": "tool", "tool_call_id": "tc_1", "content": "ok"}]


def test_compaction_summary() -> None:
    result = convert_messages([], [CompactionSummary(summary="Looked up Paris.", event_count=3)], SerializerCls)

    assert result == [{"role": "user", "content": "[Summary of earlier conversation]\nLooked up Paris."}]


def test_unsupported_input_raises() -> None:
    with pytest.raises(UnsupportedInputError):
        convert_messages([], [ModelRequest([ImageInput("https://example.com/image.png")])], SerializerCls)
