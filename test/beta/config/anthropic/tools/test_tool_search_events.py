# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from anthropic.types import (
    ServerToolUseBlock,
    ToolReferenceBlock,
    ToolSearchToolResultBlock,
    ToolSearchToolResultError,
    ToolSearchToolSearchResultBlock,
)

from ag2.config.anthropic.events import AnthropicServerToolCallEvent, AnthropicServerToolResultEvent
from ag2.tools.builtin.tool_search import TOOL_SEARCH_TOOL_NAME


def test_call_event_from_tool_search_server_tool_use():
    block = ServerToolUseBlock(
        id="srvtoolu_1",
        name="tool_search_tool_regex",
        input={"query": "weather"},
        type="server_tool_use",
    )
    event = AnthropicServerToolCallEvent.from_block(block)
    assert event is not None
    assert event.name == TOOL_SEARCH_TOOL_NAME


def test_result_event_extracts_tool_references():
    block = ToolSearchToolResultBlock(
        tool_use_id="srvtoolu_1",
        type="tool_search_tool_result",
        content=ToolSearchToolSearchResultBlock(
            type="tool_search_tool_search_result",
            tool_references=[ToolReferenceBlock(tool_name="get_weather", type="tool_reference")],
        ),
    )
    event = AnthropicServerToolResultEvent.from_block(block)
    assert event is not None
    assert event.name == TOOL_SEARCH_TOOL_NAME
    assert event.result.metadata["tool_references"] == ["get_weather"]


def test_result_event_handles_error():
    block = ToolSearchToolResultBlock(
        tool_use_id="srvtoolu_1",
        type="tool_search_tool_result",
        content=ToolSearchToolResultError(
            type="tool_search_tool_result_error",
            error_code="unavailable",
        ),
    )
    event = AnthropicServerToolResultEvent.from_block(block)
    assert event is not None
    assert event.result.metadata["error"] is True
    assert event.result.metadata["error_code"] == "unavailable"
