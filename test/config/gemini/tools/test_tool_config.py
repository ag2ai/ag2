# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from google.genai import types

from ag2 import Context
from ag2.config.gemini.mappers import build_tool_config
from ag2.tools.builtin.code_execution import CodeExecutionTool
from ag2.tools.builtin.file_search import FileSearchTool
from ag2.tools.builtin.google_maps import GoogleMapsTool
from ag2.tools.builtin.web_fetch import WebFetchTool
from ag2.tools.builtin.web_search import WebSearchTool
from ag2.tools.schemas import ToolSchema
from ag2.tools.tool import Tool
from test.config._helpers import make_tool

SERVER_SIDE_TOOLS: list[Tool] = [
    WebSearchTool(),
    WebFetchTool(),
    CodeExecutionTool(),
    FileSearchTool(store_names=["projects/p/locations/l/fileSearchStores/s"]),
    GoogleMapsTool(),
]


@pytest.mark.asyncio
class TestServerSideInvocations:
    """Gemini rejects builtins alongside function calling unless the flag is set."""

    @pytest.mark.parametrize("tool", SERVER_SIDE_TOOLS, ids=lambda t: type(t).__name__)
    async def test_mixed_with_function_tool_enables_flag(self, tool: Tool, context: Context) -> None:
        [server_side] = await tool.schemas(context)

        assert build_tool_config([make_tool().schema, server_side]) == types.ToolConfig(
            include_server_side_tool_invocations=True
        )

    @pytest.mark.parametrize("tool", SERVER_SIDE_TOOLS, ids=lambda t: type(t).__name__)
    async def test_server_side_alone_needs_no_config(self, tool: Tool, context: Context) -> None:
        [server_side] = await tool.schemas(context)

        assert build_tool_config([server_side]) is None

    async def test_function_tools_alone_need_no_config(self) -> None:
        assert build_tool_config([make_tool().schema]) is None

    async def test_all_builtins_mixed_with_function_tool(self, context: Context) -> None:
        schemas: list[ToolSchema] = [make_tool().schema]
        for tool in SERVER_SIDE_TOOLS:
            schemas.extend(await tool.schemas(context))

        assert build_tool_config(schemas) == types.ToolConfig(include_server_side_tool_invocations=True)

    async def test_vertexai_leaves_flag_unset(self, context: Context) -> None:
        """Vertex AI does not support ``include_server_side_tool_invocations``."""
        [server_side] = await WebSearchTool().schemas(context)

        assert build_tool_config([make_tool().schema, server_side], vertexai=True) is None


@pytest.mark.asyncio
class TestGeoBiasCombined:
    async def test_geo_bias_kept_alongside_another_builtin(self, context: Context) -> None:
        [maps] = await GoogleMapsTool(latitude=37.42, longitude=-122.08, language_code="en").schemas(context)
        [search] = await WebSearchTool().schemas(context)

        assert build_tool_config([maps, search]) == types.ToolConfig(
            retrieval_config=types.RetrievalConfig(
                lat_lng=types.LatLng(latitude=37.42, longitude=-122.08),
                language_code="en",
            )
        )

    async def test_geo_bias_mixed_with_function_tool(self, context: Context) -> None:
        [maps] = await GoogleMapsTool(latitude=37.42, longitude=-122.08, language_code="en").schemas(context)

        assert build_tool_config([make_tool().schema, maps]) == types.ToolConfig(
            retrieval_config=types.RetrievalConfig(
                lat_lng=types.LatLng(latitude=37.42, longitude=-122.08),
                language_code="en",
            ),
            include_server_side_tool_invocations=True,
        )

    async def test_geo_bias_on_vertexai_keeps_retrieval_config(self, context: Context) -> None:
        [maps] = await GoogleMapsTool(latitude=37.42, longitude=-122.08, language_code="en").schemas(context)

        assert build_tool_config([make_tool().schema, maps], vertexai=True) == types.ToolConfig(
            retrieval_config=types.RetrievalConfig(
                lat_lng=types.LatLng(latitude=37.42, longitude=-122.08),
                language_code="en",
            )
        )
