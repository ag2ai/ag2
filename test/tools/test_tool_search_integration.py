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
    @tool(defer_loading=True)
    def get_weather(location: str) -> str:
        """Get the weather at a location."""
        return location

    @tool
    def echo(text: str) -> str:
        """Echo text back."""
        return text

    search = ToolSearchTool()

    # Build the schema list the way an agent would.
    schemas = [(await search.schemas(None))[0], get_weather.schema, echo.schema]

    anthropic = [anthropic_tool_to_api(s) for s in schemas]
    assert anthropic[0] == {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
    assert anthropic[1]["defer_loading"] is True
    assert "defer_loading" not in anthropic[2]
    assert anthropic[1].get("name") == "get_weather"
    assert anthropic[2].get("name") == "echo"

    openai = [tool_to_responses_api(s) for s in schemas]
    assert openai[0] == {"type": "tool_search"}
    assert openai[1]["defer_loading"] is True
    assert "defer_loading" not in openai[2]
    assert openai[1].get("name") == "get_weather"
    assert openai[2].get("name") == "echo"
