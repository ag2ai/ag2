# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Context
from ag2.config.anthropic.mappers import tool_to_api
from ag2.exceptions import WebFetchOptionUnsupportedError
from ag2.tools.builtin.web_fetch import WEB_FETCH_VERSIONS, WebFetchTool, WebFetchVersions


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


USE_CACHE_VERSIONS = ("web_fetch_20260309", "web_fetch_20260318")
RESPONSE_INCLUSION_VERSIONS = ("web_fetch_20260318",)
BEFORE_USE_CACHE = [v for v in WEB_FETCH_VERSIONS if v not in USE_CACHE_VERSIONS]
BEFORE_RESPONSE_INCLUSION = [v for v in WEB_FETCH_VERSIONS if v not in RESPONSE_INCLUSION_VERSIONS]


@pytest.mark.asyncio
@pytest.mark.parametrize("version", WEB_FETCH_VERSIONS)
async def test_strict_is_carried_by_every_version(context: Context, version: WebFetchVersions) -> None:
    tool = WebFetchTool(strict=True, version=version)

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {
        "type": version,
        "name": "web_fetch",
        "strict": True,
    }


@pytest.mark.asyncio
class TestUseCache:
    @pytest.mark.parametrize("version", USE_CACHE_VERSIONS)
    async def test_reaches_the_request(self, context: Context, version: WebFetchVersions) -> None:
        tool = WebFetchTool(use_cache=False, version=version)

        [schema] = await tool.schemas(context)

        assert tool_to_api(schema) == {
            "type": version,
            "name": "web_fetch",
            "use_cache": False,
        }

    @pytest.mark.parametrize("version", BEFORE_USE_CACHE)
    async def test_earlier_version_is_refused(self, context: Context, version: WebFetchVersions) -> None:
        tool = WebFetchTool(use_cache=True, version=version)

        [schema] = await tool.schemas(context)

        with pytest.raises(WebFetchOptionUnsupportedError) as exc_info:
            tool_to_api(schema)

        assert "use_cache" in str(exc_info.value)
        assert version in str(exc_info.value)


@pytest.mark.asyncio
class TestResponseInclusion:
    async def test_reaches_the_request(self, context: Context) -> None:
        tool = WebFetchTool(response_inclusion="excluded", version="web_fetch_20260318")

        [schema] = await tool.schemas(context)

        assert tool_to_api(schema) == {
            "type": "web_fetch_20260318",
            "name": "web_fetch",
            "response_inclusion": "excluded",
        }

    @pytest.mark.parametrize("version", BEFORE_RESPONSE_INCLUSION)
    async def test_earlier_version_is_refused(self, context: Context, version: WebFetchVersions) -> None:
        tool = WebFetchTool(response_inclusion="full", version=version)

        [schema] = await tool.schemas(context)

        with pytest.raises(WebFetchOptionUnsupportedError) as exc_info:
            tool_to_api(schema)

        assert "response_inclusion" in str(exc_info.value)
        assert version in str(exc_info.value)


@pytest.mark.asyncio
async def test_every_option_on_the_newest_version(context: Context) -> None:
    tool = WebFetchTool(
        max_uses=5,
        allowed_domains=["docs.example.com"],
        blocked_domains=["private.example.com"],
        citations=True,
        max_content_tokens=50000,
        strict=True,
        use_cache=False,
        response_inclusion="excluded",
        version="web_fetch_20260318",
    )

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {
        "type": "web_fetch_20260318",
        "name": "web_fetch",
        "max_uses": 5,
        "allowed_domains": ["docs.example.com"],
        "blocked_domains": ["private.example.com"],
        "citations": {"enabled": True},
        "max_content_tokens": 50000,
        "strict": True,
        "use_cache": False,
        "response_inclusion": "excluded",
    }


@pytest.mark.asyncio
async def test_the_refusal_names_the_version_the_option_arrived_in(context: Context) -> None:
    tool = WebFetchTool(use_cache=True, version="web_fetch_20250910")

    [schema] = await tool.schemas(context)

    with pytest.raises(WebFetchOptionUnsupportedError, match="web_fetch_20260309"):
        tool_to_api(schema)
