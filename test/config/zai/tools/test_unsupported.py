# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Context
from ag2.config.zai.mappers import tool_to_api
from ag2.exceptions import UnsupportedToolError
from ag2.tools.builtin.code_execution import CodeExecutionTool
from ag2.tools.builtin.file_search import FileSearchTool
from ag2.tools.builtin.image_generation import ImageGenerationTool
from ag2.tools.builtin.mcp_server import MCPServerTool
from ag2.tools.builtin.memory import MemoryTool
from ag2.tools.builtin.shell import ShellTool
from ag2.tools.builtin.skills import SkillsTool
from ag2.tools.builtin.web_fetch import WebFetchTool
from ag2.tools.builtin.x_search import XSearchTool


@pytest.mark.asyncio
async def test_web_fetch(context: Context) -> None:
    tool = WebFetchTool()

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError, match="zai"):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_code_execution(context: Context) -> None:
    tool = CodeExecutionTool()

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError, match="zai"):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_shell(context: Context) -> None:
    tool = ShellTool()

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError, match="zai"):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_memory(context: Context) -> None:
    tool = MemoryTool()

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError, match="zai"):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_image_generation(context: Context) -> None:
    tool = ImageGenerationTool()

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError, match="zai"):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_mcp_server(context: Context) -> None:
    tool = MCPServerTool(server_url="https://mcp.example.com/sse", server_label="example-mcp")

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError, match="zai"):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_skills(context: Context) -> None:
    tool = SkillsTool("pptx")

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError, match="zai"):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_x_search(context: Context) -> None:
    tool = XSearchTool()

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError, match="zai"):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_file_search(context: Context) -> None:
    tool = FileSearchTool(vector_store_ids=["vs_1"])

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError, match="zai"):
        tool_to_api(schema)
