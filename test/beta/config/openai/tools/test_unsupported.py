# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Context
from autogen.beta.config.openai.mappers import tool_to_api, tool_to_responses_api
from autogen.beta.exceptions import UnsupportedToolError
from autogen.beta.tools.builtin.code_execution import CodeExecutionTool
from autogen.beta.tools.builtin.image_generation import ImageGenerationTool
from autogen.beta.tools.builtin.mcp_server import MCPServerTool
from autogen.beta.tools.builtin.memory import MemoryTool
from autogen.beta.tools.builtin.shell import ShellTool
from autogen.beta.tools.builtin.web_fetch import WebFetchTool
from autogen.beta.tools.builtin.web_search import WebSearchTool


class TestCompletionsApi:
    @pytest.mark.asyncio
    async def test_web_search(self, context: Context) -> None:
        tool = WebSearchTool()

        [schema] = await tool.schemas(context)

        with pytest.raises(UnsupportedToolError):
            tool_to_api(schema)

    @pytest.mark.asyncio
    async def test_web_fetch(self, context: Context) -> None:
        tool = WebFetchTool()

        [schema] = await tool.schemas(context)

        with pytest.raises(UnsupportedToolError):
            tool_to_api(schema)

    @pytest.mark.asyncio
    async def test_code_execution(self, context: Context) -> None:
        tool = CodeExecutionTool()

        [schema] = await tool.schemas(context)

        with pytest.raises(UnsupportedToolError):
            tool_to_api(schema)

    @pytest.mark.asyncio
    async def test_shell(self, context: Context) -> None:
        tool = ShellTool()

        [schema] = await tool.schemas(context)

        with pytest.raises(UnsupportedToolError):
            tool_to_api(schema)

    @pytest.mark.asyncio
    async def test_memory(self, context: Context) -> None:
        tool = MemoryTool()

        [schema] = await tool.schemas(context)

        with pytest.raises(UnsupportedToolError):
            tool_to_api(schema)

    @pytest.mark.asyncio
    async def test_image_generation(self, context: Context) -> None:
        tool = ImageGenerationTool()

        [schema] = await tool.schemas(context)

        with pytest.raises(UnsupportedToolError):
            tool_to_api(schema)

    @pytest.mark.asyncio
    async def test_mcp_server(self, context: Context) -> None:
        tool = MCPServerTool(server_url="https://mcp.example.com/sse", server_label="example-mcp")

        [schema] = await tool.schemas(context)

        with pytest.raises(UnsupportedToolError):
            tool_to_api(schema)


class TestResponsesApi:
    @pytest.mark.asyncio
    async def test_web_fetch(self, context: Context) -> None:
        tool = WebFetchTool()

        [schema] = await tool.schemas(context)

        with pytest.raises(UnsupportedToolError):
            tool_to_responses_api(schema)

    @pytest.mark.asyncio
    async def test_memory(self, context: Context) -> None:
        tool = MemoryTool()

        [schema] = await tool.schemas(context)

        with pytest.raises(UnsupportedToolError):
            tool_to_responses_api(schema)
