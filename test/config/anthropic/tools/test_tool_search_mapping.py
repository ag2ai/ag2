# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.config.anthropic.mappers import tool_to_api
from ag2.tools.builtin.tool_search import ToolSearchToolSchema
from ag2.tools.final.function_tool import FunctionDefinition, FunctionToolSchema


def test_deferred_function_emits_defer_loading():
    schema = FunctionToolSchema(
        function=FunctionDefinition(name="get_weather", description="d", parameters={"type": "object"}),
        defer_loading=True,
    )
    result = tool_to_api(schema)
    assert result["name"] == "get_weather"
    assert result["defer_loading"] is True


def test_non_deferred_function_omits_defer_loading():
    schema = FunctionToolSchema(
        function=FunctionDefinition(name="get_weather", description="d", parameters={"type": "object"}),
    )
    result = tool_to_api(schema)
    assert "defer_loading" not in result


def test_tool_search_regex_type():
    result = tool_to_api(ToolSearchToolSchema(mode="regex"))
    assert result == {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}


def test_tool_search_bm25_type():
    result = tool_to_api(ToolSearchToolSchema(mode="bm25"))
    assert result == {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}
