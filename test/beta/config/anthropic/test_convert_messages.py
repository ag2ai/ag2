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


class TestOrphanedToolResults:
    """Tool results whose matching tool_use was trimmed should be filtered out."""

    def test_orphaned_tool_result_is_dropped(self) -> None:
        """A ToolResultsEvent with no matching ModelResponse tool_use should be omitted."""
        events = [
            ToolResultsEvent(
                results=[
                    ToolResultEvent(
                        parent_id="orphan_id",
                        name="missing_tool",
                        result=ToolResult(content="stale"),
                    )
                ],
            ),
            ModelRequest(content="hello"),
        ]
        result = convert_messages(events)

        # Only the ModelRequest should produce output
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"

    def test_mixed_orphaned_and_valid_results(self) -> None:
        """Only valid tool results are kept; orphaned ones are filtered from the same event."""
        events = [
            _model_response_with_tool_call('{"q": "test"}'),
            ToolResultsEvent(
                results=[
                    ToolResultEvent(
                        parent_id="tc_1",
                        name="list_items",
                        result=ToolResult(content="valid"),
                    ),
                    ToolResultEvent(
                        parent_id="orphan_id",
                        name="gone_tool",
                        result=ToolResult(content="stale"),
                    ),
                ],
            ),
        ]
        result = convert_messages(events)

        assert len(result) == 2
        tool_results = result[1]["content"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_use_id"] == "tc_1"

    def test_all_results_orphaned_skips_entire_block(self) -> None:
        """If every result in a ToolResultsEvent is orphaned, no user message is emitted."""
        events = [
            ModelRequest(content="hi"),
            ToolResultsEvent(
                results=[
                    ToolResultEvent(
                        parent_id="gone_1",
                        name="a",
                        result=ToolResult(content="x"),
                    ),
                    ToolResultEvent(
                        parent_id="gone_2",
                        name="b",
                        result=ToolResult(content="y"),
                    ),
                ],
            ),
        ]
        result = convert_messages(events)

        assert len(result) == 1
        assert result[0]["content"] == "hi"

    def test_valid_tool_results_are_preserved(self) -> None:
        """Normal flow: tool results with matching tool_use IDs pass through."""
        events = [
            ModelRequest(content="go"),
            _model_response_with_tool_call("{}"),
            ToolResultsEvent(
                results=[
                    ToolResultEvent(
                        parent_id="tc_1",
                        name="list_items",
                        result=ToolResult(content="items"),
                    )
                ],
            ),
        ]
        result = convert_messages(events)

        assert len(result) == 3
        assert result[2]["role"] == "user"
        assert result[2]["content"][0]["type"] == "tool_result"
        assert result[2]["content"][0]["tool_use_id"] == "tc_1"
