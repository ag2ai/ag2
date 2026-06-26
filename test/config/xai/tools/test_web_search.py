# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Context
from ag2.config.xai.mappers import tool_to_api
from ag2.tools.builtin.web_search import UserLocation, WebSearchTool


@pytest.mark.asyncio
async def test_defaults(context: Context) -> None:
    tool = WebSearchTool()

    [schema] = await tool.schemas(context)
    api = tool_to_api(schema)

    assert api.HasField("web_search")


@pytest.mark.asyncio
async def test_allowed_and_blocked_domains(context: Context) -> None:
    tool = WebSearchTool(allowed_domains=["example.com"], blocked_domains=["spam.com"])

    [schema] = await tool.schemas(context)
    api = tool_to_api(schema)

    assert list(api.web_search.allowed_domains) == ["example.com"]
    assert list(api.web_search.excluded_domains) == ["spam.com"]


@pytest.mark.asyncio
async def test_user_location(context: Context) -> None:
    tool = WebSearchTool(user_location=UserLocation(country="US", city="San Francisco", timezone="America/Los_Angeles"))

    [schema] = await tool.schemas(context)
    api = tool_to_api(schema)

    assert api.web_search.user_location.country == "US"
    assert api.web_search.user_location.city == "San Francisco"
    assert api.web_search.user_location.timezone == "America/Los_Angeles"
