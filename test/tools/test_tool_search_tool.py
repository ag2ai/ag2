# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.tools.builtin import ToolSearchTool
from ag2.tools.builtin.tool_search import TOOL_SEARCH_TOOL_NAME, ToolSearchToolSchema


@pytest.mark.asyncio
async def test_tool_search_default_mode_is_regex():
    schemas = await ToolSearchTool().schemas(context=None)
    assert len(schemas) == 1
    schema = schemas[0]
    assert isinstance(schema, ToolSearchToolSchema)
    assert schema.type == TOOL_SEARCH_TOOL_NAME
    assert schema.mode == "regex"


@pytest.mark.asyncio
async def test_tool_search_bm25_mode():
    schemas = await ToolSearchTool(mode="bm25").schemas(context=None)
    assert schemas[0].mode == "bm25"


def test_tool_search_name():
    assert ToolSearchTool().name == TOOL_SEARCH_TOOL_NAME
