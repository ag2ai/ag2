# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.config.anthropic.mappers import tool_to_api
from autogen.beta.context import Context
from autogen.beta.tools.builtin.web_fetch import WebFetchTool


@pytest.mark.asyncio
async def test_defaults(context: Context) -> None:
    tool = WebFetchTool()

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
    }


@pytest.mark.asyncio
async def test_full(context: Context) -> None:
    tool = WebFetchTool(
        max_uses=5,
        allowed_domains=["docs.example.com"],
        blocked_domains=["private.example.com"],
        citations=True,
        max_content_tokens=50000,
    )

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
        "max_uses": 5,
        "allowed_domains": ["docs.example.com"],
        "blocked_domains": ["private.example.com"],
        "citations": {"enabled": True},
        "max_content_tokens": 50000,
    }


@pytest.mark.asyncio
async def test_dynamic_version(context: Context) -> None:
    tool = WebFetchTool(version="web_fetch_20260209")

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {
        "type": "web_fetch_20260209",
        "name": "web_fetch",
    }
