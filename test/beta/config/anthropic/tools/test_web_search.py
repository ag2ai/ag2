# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Context
from autogen.beta.config.anthropic.mappers import tool_to_api
from autogen.beta.tools.builtin.web_search import UserLocation, WebSearchTool


@pytest.mark.asyncio
async def test_defaults(context: Context) -> None:
    tool = WebSearchTool()

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {
        "type": "web_search_20250305",
        "name": "web_search",
    }


@pytest.mark.asyncio
async def test_with_max_uses(context: Context) -> None:
    tool = WebSearchTool(max_uses=10)

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 10,
    }


@pytest.mark.asyncio
async def test_with_user_location(context: Context) -> None:
    tool = WebSearchTool(
        user_location=UserLocation(city="London", country="GB", timezone="Europe/London"),
    )

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {
        "type": "web_search_20250305",
        "name": "web_search",
        "user_location": {
            "type": "approximate",
            "city": "London",
            "country": "GB",
            "timezone": "Europe/London",
        },
    }


@pytest.mark.asyncio
async def test_with_allowed_domains(context: Context) -> None:
    tool = WebSearchTool(allowed_domains=["example.com", "trusteddomain.org"])

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {
        "type": "web_search_20250305",
        "name": "web_search",
        "allowed_domains": ["example.com", "trusteddomain.org"],
    }


@pytest.mark.asyncio
async def test_with_blocked_domains(context: Context) -> None:
    tool = WebSearchTool(blocked_domains=["untrusted.com"])

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {
        "type": "web_search_20250305",
        "name": "web_search",
        "blocked_domains": ["untrusted.com"],
    }


@pytest.mark.asyncio
async def test_dynamic_version(context: Context) -> None:
    tool = WebSearchTool(version="web_search_20260209")

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {
        "type": "web_search_20260209",
        "name": "web_search",
    }


@pytest.mark.asyncio
async def test_dynamic_version_with_domains(context: Context) -> None:
    tool = WebSearchTool(
        max_uses=5,
        allowed_domains=["docs.example.com"],
        blocked_domains=["spam.example.com"],
        version="web_search_20260209",
    )

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": 5,
        "allowed_domains": ["docs.example.com"],
        "blocked_domains": ["spam.example.com"],
    }
