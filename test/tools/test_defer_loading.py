# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.tools import tool
from ag2.tools.builtin import ToolSearchTool


@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return location


def test_plain_tool_has_defer_loading_false():
    assert get_weather.schema.defer_loading is False


@pytest.mark.asyncio
async def test_wrapping_in_tool_search_defers_the_tool():
    [_search, weather] = await ToolSearchTool(get_weather).schemas(context=None)
    assert weather.function.name == "get_weather"
    assert weather.defer_loading is True


@pytest.mark.asyncio
async def test_wrapping_does_not_mutate_the_original_schema():
    await ToolSearchTool(get_weather).schemas(context=None)
    # the source tool's own schema stays eager — only the emitted copy is deferred
    assert get_weather.schema.defer_loading is False
