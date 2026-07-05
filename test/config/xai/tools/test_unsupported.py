# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Context
from ag2.config.xai.mappers import tool_to_api
from ag2.exceptions import UnsupportedToolError
from ag2.tools.builtin.file_search import FileSearchTool
from ag2.tools.builtin.image_generation import ImageGenerationTool
from ag2.tools.builtin.memory import MemoryTool
from ag2.tools.builtin.retrieval import RetrievalTool
from ag2.tools.builtin.shell import ShellTool
from ag2.tools.builtin.skills import SkillsTool
from ag2.tools.builtin.web_fetch import WebFetchTool


@pytest.mark.asyncio
async def test_shell(context: Context) -> None:
    tool = ShellTool()

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError, match="xai"):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_memory(context: Context) -> None:
    tool = MemoryTool()

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError, match="xai"):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_image_generation(context: Context) -> None:
    tool = ImageGenerationTool()

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError, match="xai"):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_skills(context: Context) -> None:
    tool = SkillsTool("pptx")

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError, match="xai"):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_web_fetch(context: Context) -> None:
    tool = WebFetchTool()

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError, match="xai"):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_retrieval(context: Context) -> None:
    tool = RetrievalTool("kb_123")

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError, match="xai"):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_file_search(context: Context) -> None:
    tool = FileSearchTool(vector_store_ids=["vs_1"])

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError, match="xai"):
        tool_to_api(schema)
