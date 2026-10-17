# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.config.openai.mappers import tool_to_api, tool_to_responses_api
from ag2.exceptions import UnsupportedToolError
from ag2.tools.builtin.tool_search import ToolSearchToolSchema
from ag2.tools.final.function_tool import FunctionDefinition, FunctionToolSchema


def test_responses_deferred_function_emits_defer_loading():
    schema = FunctionToolSchema(
        function=FunctionDefinition(name="get_weather", description="d", parameters={"type": "object"}),
        defer_loading=True,
    )
    result = tool_to_responses_api(schema)
    assert result["name"] == "get_weather"
    assert result["defer_loading"] is True


def test_responses_non_deferred_function_omits_defer_loading():
    schema = FunctionToolSchema(
        function=FunctionDefinition(name="get_weather", description="d", parameters={"type": "object"}),
    )
    result = tool_to_responses_api(schema)
    assert "defer_loading" not in result


def test_responses_tool_search_type():
    result = tool_to_responses_api(ToolSearchToolSchema())
    assert result == {"type": "tool_search"}


def test_completions_rejects_tool_search():
    with pytest.raises(UnsupportedToolError):
        tool_to_api(ToolSearchToolSchema())


def test_completions_rejects_deferred_function():
    # The Chat Completions API has no tool-search support, so a deferred
    # function must fail fast rather than be silently sent eagerly.
    schema = FunctionToolSchema(
        function=FunctionDefinition(name="get_weather", description="d", parameters={"type": "object"}),
        defer_loading=True,
    )
    with pytest.raises(UnsupportedToolError):
        tool_to_api(schema)


def test_completions_allows_non_deferred_function():
    schema = FunctionToolSchema(
        function=FunctionDefinition(name="get_weather", description="d", parameters={"type": "object"}),
    )
    result = tool_to_api(schema)
    assert result["function"]["name"] == "get_weather"
