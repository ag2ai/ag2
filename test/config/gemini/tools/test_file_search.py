# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from google.genai import types

from ag2 import Context
from ag2.config.gemini.mappers import build_tools
from ag2.tools.builtin.file_search import FileSearchTool


@pytest.mark.asyncio
async def test_store_names(context: Context) -> None:
    tool = FileSearchTool(store_names=["projects/p/fileSearchStores/s"])

    [schema] = await tool.schemas(context)

    assert build_tools([schema]) == [
        types.Tool(file_search=types.FileSearch(file_search_store_names=["projects/p/fileSearchStores/s"])),
    ]


@pytest.mark.asyncio
async def test_top_k_and_metadata_filter(context: Context) -> None:
    tool = FileSearchTool(store_names=["s"], max_num_results=5, metadata_filter='author="A"')

    [schema] = await tool.schemas(context)

    assert build_tools([schema]) == [
        types.Tool(
            file_search=types.FileSearch(
                file_search_store_names=["s"],
                top_k=5,
                metadata_filter='author="A"',
            )
        ),
    ]
