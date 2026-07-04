# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Context
from ag2.config.anthropic.mappers import tool_to_api
from ag2.exceptions import UnsupportedToolError
from ag2.tools.builtin.file_search import FileSearchTool
from ag2.tools.builtin.image_generation import ImageGenerationTool
from ag2.tools.builtin.retrieval import RetrievalTool
from ag2.tools.builtin.shell import ShellTool
from ag2.tools.builtin.x_search import XSearchTool


@pytest.mark.asyncio
async def test_image_generation(context: Context) -> None:
    tool = ImageGenerationTool()

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_shell(context: Context) -> None:
    """ShellTool is unsupported on Anthropic (client-side bash; use SandboxShellTool)."""
    tool = ShellTool()

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_x_search(context: Context) -> None:
    tool = XSearchTool()

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_retrieval(context: Context) -> None:
    tool = RetrievalTool("kb_123")

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_file_search(context: Context) -> None:
    tool = FileSearchTool(vector_store_ids=["vs_1"])

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError):
        tool_to_api(schema)
