# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Context
from autogen.beta.config.gemini.mappers import build_tools
from autogen.beta.exceptions import UnsupportedToolError
from autogen.beta.tools.builtin.image_generation import ImageGenerationTool
from autogen.beta.tools.builtin.mcp_server import MCPServerTool
from autogen.beta.tools.builtin.memory import MemoryTool
from autogen.beta.tools.builtin.shell import ShellTool
from autogen.beta.tools.builtin.skills import SkillsTool


@pytest.mark.asyncio
async def test_shell(context: Context) -> None:
    tool = ShellTool()

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError):
        build_tools([schema])


@pytest.mark.asyncio
async def test_memory(context: Context) -> None:
    tool = MemoryTool()

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError):
        build_tools([schema])


@pytest.mark.asyncio
async def test_image_generation(context: Context) -> None:
    tool = ImageGenerationTool()

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError):
        build_tools([schema])


@pytest.mark.asyncio
async def test_mcp_server(context: Context) -> None:
    tool = MCPServerTool(server_url="https://mcp.example.com/sse", server_label="example-mcp")

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError):
        build_tools([schema])


@pytest.mark.asyncio
async def test_skills(context: Context) -> None:
    tool = SkillsTool("pptx")

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError):
        build_tools([schema])
