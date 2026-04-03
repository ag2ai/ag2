# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import ToolResult
from autogen.beta.config.anthropic.mappers import convert_messages
from autogen.beta.events import (
    ModelRequest,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
)


def _model_response_with_tool_call(arguments: str | None) -> ModelResponse:
    """Helper to build a ModelResponse containing a single tool call."""
    return ModelResponse(
        message=None,
        tool_calls=ToolCallsEvent(
            calls=[ToolCallEvent(id="tc_1", name="list_items", arguments=arguments)],
        ),
    )


class TestConvertMessagesEmptyArguments:
    """json.loads must not crash on empty or None tool call arguments."""

    @pytest.mark.parametrize("arguments", ["", None])
    def test_empty_arguments_produce_empty_dict(self, arguments: str | None) -> None:
        response = _model_response_with_tool_call(arguments)
        result = convert_messages([response])

        assert len(result) == 1
        tool_use_block = result[0]["content"][0]
        assert tool_use_block["type"] == "tool_use"
        assert tool_use_block["input"] == {}

    def test_valid_arguments_are_preserved(self) -> None:
        response = _model_response_with_tool_call('{"category": "books"}')
        result = convert_messages([response])

        tool_use_block = result[0]["content"][0]
        assert tool_use_block["input"] == {"category": "books"}

    def test_empty_object_arguments(self) -> None:
        response = _model_response_with_tool_call("{}")
        result = convert_messages([response])

        tool_use_block = result[0]["content"][0]
        assert tool_use_block["input"] == {}


class TestConvertMessagesRoundTrip:
    """A request → response-with-tool-call → tool-result sequence should convert cleanly."""

    def test_full_sequence_with_empty_args(self) -> None:
        events = [
            ModelRequest(content="What items do we have?"),
            _model_response_with_tool_call(""),
            ToolResultsEvent(
                results=[
                    ToolResultEvent(
                        parent_id="tc_1",
                        name="list_items",
                        result=ToolResult(content="apple, banana"),
                    )
                ],
            ),
        ]
        result = convert_messages(events)

        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"][0]["input"] == {}
        assert result[2]["role"] == "user"
        assert result[2]["content"][0]["type"] == "tool_result"
