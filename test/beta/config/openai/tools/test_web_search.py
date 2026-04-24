# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Context
from autogen.beta.config.openai.mappers import tool_to_responses_api
from autogen.beta.tools.builtin.web_search import UserLocation, WebSearchTool


@pytest.mark.asyncio
async def test_responses_api_defaults(context: Context) -> None:
    tool = WebSearchTool()

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {"type": "web_search"}


@pytest.mark.asyncio
async def test_responses_api_with_context_size(context: Context) -> None:
    tool = WebSearchTool(search_context_size="high")

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {"type": "web_search", "search_context_size": "high"}


@pytest.mark.asyncio
async def test_responses_api_with_max_uses(context: Context) -> None:
    tool = WebSearchTool(max_uses=5)

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {"type": "web_search", "max_uses": 5}


@pytest.mark.asyncio
async def test_responses_api_all_options(context: Context) -> None:
    tool = WebSearchTool(search_context_size="low", max_uses=3)

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {
        "type": "web_search",
        "search_context_size": "low",
        "max_uses": 3,
    }


@pytest.mark.asyncio
async def test_responses_api_with_user_location(context: Context) -> None:
    tool = WebSearchTool(
        user_location=UserLocation(city="San Francisco", region="California", country="US"),
    )

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {
        "type": "web_search",
        "user_location": {
            "type": "approximate",
            "city": "San Francisco",
            "region": "California",
            "country": "US",
        },
    }


@pytest.mark.asyncio
async def test_responses_api_with_user_location_partial(context: Context) -> None:
    tool = WebSearchTool(
        user_location=UserLocation(country="DE", timezone="Europe/Berlin"),
    )

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {
        "type": "web_search",
        "user_location": {
            "type": "approximate",
            "country": "DE",
            "timezone": "Europe/Berlin",
        },
    }


@pytest.mark.asyncio
async def test_responses_api_with_allowed_domains(context: Context) -> None:
    tool = WebSearchTool(allowed_domains=["example.com", "docs.example.com"])

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {
        "type": "web_search",
        "filters": {"allowed_domains": ["example.com", "docs.example.com"]},
    }
