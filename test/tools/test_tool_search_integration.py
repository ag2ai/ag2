# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.config.anthropic.mappers import tool_to_api as anthropic_tool_to_api
from ag2.config.openai.mappers import tool_to_responses_api
from ag2.tools import tool
from ag2.tools.builtin import ToolSearchTool


@pytest.mark.asyncio
async def test_full_tool_list_maps_for_both_providers():
    @tool
    def get_weather(location: str) -> str:
        """Get the weather at a location."""
        return location

    @tool
    def echo(text: str) -> str:
        """Echo text back."""
        return text

    # Build the schema list the way an agent would: get_weather is deferred by
    # wrapping it in the search tool, echo is loaded eagerly.
    schemas = [*(await ToolSearchTool(get_weather).schemas(None)), echo.schema]

    anthropic = [anthropic_tool_to_api(s) for s in schemas]
    assert anthropic == [
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
        {
            "name": "get_weather",
            "description": "Get the weather at a location.",
            "input_schema": {
                "properties": {"location": {"title": "Location", "type": "string"}},
                "required": ["location"],
                "type": "object",
            },
            "defer_loading": True,  # deferred by wrapping in ToolSearchTool
        },
        {
            "name": "echo",
            "description": "Echo text back.",
            "input_schema": {
                "properties": {"text": {"title": "Text", "type": "string"}},
                "required": ["text"],
                "type": "object",
            },
            # no "defer_loading" key: echo is loaded eagerly
        },
    ]

    openai = [tool_to_responses_api(s) for s in schemas]
    assert openai == [
        {"type": "tool_search"},
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather at a location.",
            "parameters": {
                "properties": {"location": {"title": "Location", "type": "string"}},
                "required": ["location"],
                "type": "object",
                "additionalProperties": False,
            },
            "defer_loading": True,
        },
        {
            "type": "function",
            "name": "echo",
            "description": "Echo text back.",
            "parameters": {
                "properties": {"text": {"title": "Text", "type": "string"}},
                "required": ["text"],
                "type": "object",
                "additionalProperties": False,
            },
        },
    ]
