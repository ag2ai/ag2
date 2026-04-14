# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from autogen.beta import Context
from autogen.beta.annotations import Variable
from autogen.beta.tools import ImageGenerationTool, UserLocation, WebSearchTool
from autogen.beta.tools.builtin._resolve import resolve_variable
from autogen.beta.tools.builtin.image_generation import ImageGenerationToolSchema
from autogen.beta.tools.builtin.mcp_server import MCPServerTool, MCPServerToolSchema
from autogen.beta.tools.builtin.shell import ContainerAutoEnvironment, ShellTool, ShellToolSchema
from autogen.beta.tools.builtin.web_fetch import WebFetchTool, WebFetchToolSchema
from autogen.beta.tools.builtin.web_search import WebSearchToolSchema


def _make_context(**variables: object) -> Context:
    return Context(stream=MagicMock(), variables=variables)


class TestResolveVariable:
    def test_passthrough(self) -> None:
        ctx = _make_context()
        assert resolve_variable("hello", ctx) == "hello"
        assert resolve_variable(42, ctx) == 42
        assert resolve_variable(None, ctx) is None

    def test_from_context(self) -> None:
        loc = UserLocation(country="US")
        ctx = _make_context(user_location=loc)

        result = resolve_variable(Variable("user_location"), ctx)

        assert result is loc

    def test_default(self) -> None:
        ctx = _make_context()
        fallback = UserLocation(country="DE")

        result = resolve_variable(Variable("user_location", default=fallback), ctx)

        assert result is fallback

    def test_default_factory(self) -> None:
        ctx = _make_context()

        result = resolve_variable(Variable("counter", default_factory=dict), ctx)

        assert result == {}

    def test_context_takes_precedence_over_default(self) -> None:
        ctx = _make_context(mode="fast")

        result = resolve_variable(Variable("mode", default="slow"), ctx)

        assert result == "fast"

    def test_missing_raises(self) -> None:
        ctx = _make_context()

        with pytest.raises(KeyError, match="user_location"):
            resolve_variable(Variable("user_location"), ctx)


class TestWebSearchToolVariable:
    @pytest.mark.asyncio
    async def test_resolved(self) -> None:
        loc = UserLocation(city="Berlin", country="DE")
        tool = WebSearchTool(user_location=Variable("loc"))
        ctx = _make_context(loc=loc)

        [schema] = await tool.schemas(ctx)

        assert isinstance(schema, WebSearchToolSchema)
        assert schema.user_location is loc

    @pytest.mark.asyncio
    async def test_missing_raises(self) -> None:
        tool = WebSearchTool(user_location=Variable("loc"))
        ctx = _make_context()

        with pytest.raises(KeyError, match="loc"):
            await tool.schemas(ctx)


class TestWebFetchToolVariable:
    @pytest.mark.asyncio
    async def test_resolved(self) -> None:
        tool = WebFetchTool(max_uses=Variable("limit"))
        ctx = _make_context(limit=10)

        [schema] = await tool.schemas(ctx)

        assert isinstance(schema, WebFetchToolSchema)
        assert schema.max_uses == 10

    @pytest.mark.asyncio
    async def test_missing_raises(self) -> None:
        tool = WebFetchTool(max_uses=Variable("limit"))
        ctx = _make_context()

        with pytest.raises(KeyError, match="limit"):
            await tool.schemas(ctx)


class TestShellToolVariable:
    @pytest.mark.asyncio
    async def test_resolved(self) -> None:
        env = ContainerAutoEnvironment()
        tool = ShellTool(environment=Variable("env"))
        ctx = _make_context(env=env)

        [schema] = await tool.schemas(ctx)

        assert isinstance(schema, ShellToolSchema)
        assert schema.environment is env

    @pytest.mark.asyncio
    async def test_missing_raises(self) -> None:
        tool = ShellTool(environment=Variable("env"))
        ctx = _make_context()

        with pytest.raises(KeyError, match="env"):
            await tool.schemas(ctx)


class TestImageGenerationToolVariable:
    @pytest.mark.asyncio
    async def test_resolved(self) -> None:
        tool = ImageGenerationTool(quality="high", size=Variable("image_size"))
        ctx = _make_context(image_size="1536x1024")

        [schema] = await tool.schemas(ctx)

        assert isinstance(schema, ImageGenerationToolSchema)
        assert schema.quality == "high"
        assert schema.size == "1536x1024"

    @pytest.mark.asyncio
    async def test_missing_raises(self) -> None:
        tool = ImageGenerationTool(partial_images=Variable("partial_images"))
        ctx = _make_context()

        with pytest.raises(KeyError, match="partial_images"):
            await tool.schemas(ctx)


class TestMCPServerToolVariable:
    @pytest.mark.asyncio
    async def test_resolved(self) -> None:
        tool = MCPServerTool(server_url=Variable("url"), server_label="test-mcp")
        ctx = _make_context(url="https://mcp.example.com/sse")

        [schema] = await tool.schemas(ctx)

        assert isinstance(schema, MCPServerToolSchema)
        assert schema.server_url == "https://mcp.example.com/sse"

    @pytest.mark.asyncio
    async def test_missing_raises(self) -> None:
        tool = MCPServerTool(server_url=Variable("url"), server_label="test-mcp")
        ctx = _make_context()

        with pytest.raises(KeyError, match="url"):
            await tool.schemas(ctx)
