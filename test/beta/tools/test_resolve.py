# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pytest

from autogen.beta import Context, Variable
from autogen.beta.tools import ImageGenerationTool, UserLocation, WebSearchTool
from autogen.beta.tools.builtin._resolve import resolve_variable
from autogen.beta.tools.builtin.image_generation import ImageGenerationToolSchema
from autogen.beta.tools.builtin.mcp_server import MCPServerTool, MCPServerToolSchema
from autogen.beta.tools.builtin.shell import ContainerAutoEnvironment, ShellTool, ShellToolSchema
from autogen.beta.tools.builtin.web_fetch import WebFetchTool, WebFetchToolSchema
from autogen.beta.tools.builtin.web_search import WebSearchToolSchema


class TestResolveVariable:
    def test_passthrough(self, context: Context) -> None:
        assert resolve_variable("hello", context) == "hello"
        assert resolve_variable(42, context) == 42
        assert resolve_variable(None, context) is None

    def test_from_context(self, make_context: Callable[..., Context]) -> None:
        loc = UserLocation(country="US")
        ctx = make_context(user_location=loc)

        result = resolve_variable(Variable("user_location"), ctx)

        assert result is loc

    def test_default(self, context: Context) -> None:
        fallback = UserLocation(country="DE")

        result = resolve_variable(Variable("user_location", default=fallback), context)

        assert result is fallback

    def test_default_factory(self, context: Context) -> None:
        result = resolve_variable(Variable("counter", default_factory=dict), context)

        assert result == {}

    def test_context_takes_precedence_over_default(self, make_context: Callable[..., Context]) -> None:
        ctx = make_context(mode="fast")
        result = resolve_variable(Variable("mode", default="slow"), ctx)

        assert result == "fast"

    def test_missing_raises(self, context: Context) -> None:
        with pytest.raises(KeyError, match="user_location"):
            resolve_variable(Variable("user_location"), context)


class TestWebSearchToolVariable:
    @pytest.mark.asyncio
    async def test_resolved(self, make_context: Callable[..., Context]) -> None:
        loc = UserLocation(city="Berlin", country="DE")
        ctx = make_context(loc=loc)
        tool = WebSearchTool(user_location=Variable("loc"))

        [schema] = await tool.schemas(ctx)

        assert isinstance(schema, WebSearchToolSchema)
        assert schema.user_location is loc

    @pytest.mark.asyncio
    async def test_missing_raises(self, context: Context) -> None:
        tool = WebSearchTool(user_location=Variable("loc"))

        with pytest.raises(KeyError, match="loc"):
            await tool.schemas(context)


class TestWebFetchToolVariable:
    @pytest.mark.asyncio
    async def test_resolved(self, make_context: Callable[..., Context]) -> None:
        ctx = make_context(limit=10)
        tool = WebFetchTool(max_uses=Variable("limit"))

        [schema] = await tool.schemas(ctx)

        assert isinstance(schema, WebFetchToolSchema)
        assert schema.max_uses == 10

    @pytest.mark.asyncio
    async def test_missing_raises(self, context: Context) -> None:
        tool = WebFetchTool(max_uses=Variable("limit"))

        with pytest.raises(KeyError, match="limit"):
            await tool.schemas(context)


class TestShellToolVariable:
    @pytest.mark.asyncio
    async def test_resolved(self, make_context: Callable[..., Context]) -> None:
        env = ContainerAutoEnvironment()
        ctx = make_context(env=env)
        tool = ShellTool(environment=Variable("env"))

        [schema] = await tool.schemas(ctx)

        assert isinstance(schema, ShellToolSchema)
        assert schema.environment is env

    @pytest.mark.asyncio
    async def test_missing_raises(self, context: Context) -> None:
        tool = ShellTool(environment=Variable("env"))

        with pytest.raises(KeyError, match="env"):
            await tool.schemas(context)


class TestImageGenerationToolVariable:
    @pytest.mark.asyncio
    async def test_resolved(self, make_context: Callable[..., Context]) -> None:
        ctx = make_context(image_size="1536x1024")
        tool = ImageGenerationTool(quality="high", size=Variable("image_size"))

        [schema] = await tool.schemas(ctx)

        assert isinstance(schema, ImageGenerationToolSchema)
        assert schema.quality == "high"
        assert schema.size == "1536x1024"

    @pytest.mark.asyncio
    async def test_missing_raises(self, context: Context) -> None:
        tool = ImageGenerationTool(partial_images=Variable("partial_images"))

        with pytest.raises(KeyError, match="partial_images"):
            await tool.schemas(context)


class TestMCPServerToolVariable:
    @pytest.mark.asyncio
    async def test_resolved(self, make_context: Callable[..., Context]) -> None:
        ctx = make_context(url="https://mcp.example.com/sse")
        tool = MCPServerTool(server_url=Variable("url"), server_label="test-mcp")

        [schema] = await tool.schemas(ctx)

        assert isinstance(schema, MCPServerToolSchema)
        assert schema.server_url == "https://mcp.example.com/sse"

    @pytest.mark.asyncio
    async def test_missing_raises(self, context: Context) -> None:
        tool = MCPServerTool(server_url=Variable("url"), server_label="test-mcp")

        with pytest.raises(KeyError, match="url"):
            await tool.schemas(context)
