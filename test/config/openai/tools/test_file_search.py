# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Context
from ag2.config.openai.mappers import responses_api_includes, tool_to_responses_api
from ag2.tools.builtin.file_search import FileSearchTool


@pytest.mark.asyncio
async def test_responses_api_defaults(context: Context) -> None:
    tool = FileSearchTool(vector_store_ids=["vs_1"])

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {"type": "file_search", "vector_store_ids": ["vs_1"]}


@pytest.mark.asyncio
async def test_responses_api_all_options(context: Context) -> None:
    tool = FileSearchTool(
        vector_store_ids=["vs_1", "vs_2"],
        max_num_results=2,
        filters={"type": "in", "key": "category", "value": ["blog"]},
    )

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {
        "type": "file_search",
        "vector_store_ids": ["vs_1", "vs_2"],
        "max_num_results": 2,
        "filters": {"type": "in", "key": "category", "value": ["blog"]},
    }


@pytest.mark.asyncio
async def test_includes_off_by_default(context: Context) -> None:
    tool = FileSearchTool(vector_store_ids=["vs_1"])

    schemas = await tool.schemas(context)

    assert responses_api_includes(schemas) == []


@pytest.mark.asyncio
async def test_includes_results_opt_in(context: Context) -> None:
    tool = FileSearchTool(vector_store_ids=["vs_1"], include_results=True)

    schemas = await tool.schemas(context)

    assert responses_api_includes(schemas) == ["file_search_call.results"]
