# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Context
from autogen.beta.config.zai.mappers import tool_to_api
from autogen.beta.tools.builtin.web_search import WebSearchTool


@pytest.mark.asyncio
async def test_defaults(context: Context) -> None:
    tool = WebSearchTool()

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {"type": "web_search", "web_search": {"search_engine": "search-prime"}}


@pytest.mark.asyncio
async def test_search_context_size_maps_to_content_size(context: Context) -> None:
    tool = WebSearchTool(search_context_size="high")

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {
        "type": "web_search",
        "web_search": {"search_engine": "search-prime", "content_size": "high"},
    }


@pytest.mark.asyncio
async def test_fields_zai_cannot_express_are_ignored(context: Context) -> None:
    tool = WebSearchTool(max_uses=5, allowed_domains=["example.com"], blocked_domains=["spam.com"])

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {"type": "web_search", "web_search": {"search_engine": "search-prime"}}
