# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.exceptions import ToolConflictError
from ag2.tools import tool
from ag2.tools.builtin import ToolSearchTool
from ag2.tools.builtin.tool_search import TOOL_SEARCH_TOOL_NAME, ToolSearchToolSchema
from ag2.tools.final import FunctionToolSchema


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{city}: sunny"


@tool
def get_stock_price(ticker: str) -> str:
    """Get the latest stock price for a ticker symbol."""
    return f"{ticker}: $100"


@pytest.mark.asyncio
async def test_tool_search_default_mode_is_regex():
    [search, *_] = await ToolSearchTool(get_weather).schemas(context=None)
    assert isinstance(search, ToolSearchToolSchema)
    assert search.type == TOOL_SEARCH_TOOL_NAME
    assert search.mode == "regex"


@pytest.mark.asyncio
async def test_tool_search_bm25_mode():
    [search, *_] = await ToolSearchTool(get_weather, mode="bm25").schemas(context=None)
    assert search.mode == "bm25"


def test_tool_search_name():
    assert ToolSearchTool(get_weather).name == TOOL_SEARCH_TOOL_NAME


def test_tool_search_requires_at_least_one_tool():
    with pytest.raises(ValueError, match="at least one tool"):
        ToolSearchTool()


@pytest.mark.asyncio
async def test_wrapped_tools_are_marked_deferred():
    schemas = await ToolSearchTool(get_weather, get_stock_price).schemas(context=None)

    # first schema is the search tool itself, the rest are the deferred wrapped tools
    assert isinstance(schemas[0], ToolSearchToolSchema)
    wrapped = schemas[1:]
    assert {s.function.name for s in wrapped} == {"get_weather", "get_stock_price"}
    assert all(isinstance(s, FunctionToolSchema) and s.defer_loading for s in wrapped)


def test_duplicate_wrapped_tool_names_raise():
    with pytest.raises(ToolConflictError):
        ToolSearchTool(get_weather, get_weather)
