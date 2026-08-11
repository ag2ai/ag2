# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from mcp.server.lowlevel import NotificationOptions
from mcp.types import ListResourceTemplatesRequest

from ag2 import Agent
from ag2.mcp import MCPServer, Resource, ResourceTemplate
from ag2.mcp.errors import MCPResourceNotFoundError
from ag2.mcp.resources import ResourceProvider
from ag2.testing import TestConfig


@pytest.mark.asyncio
class TestResourceRead:
    async def test_reads_static_resource(self) -> None:
        provider = ResourceProvider([Resource(uri="config://app", name="app", read=lambda: "hello")], [])

        [contents] = await provider.read("config://app")

        assert contents.content == "hello"

    async def test_reads_async_resource(self) -> None:
        async def _read() -> str:
            return "async-body"

        provider = ResourceProvider([Resource(uri="config://app", name="app", read=_read)], [])

        [contents] = await provider.read("config://app")

        assert contents.content == "async-body"

    async def test_matches_template_and_extracts_vars(self) -> None:
        provider = ResourceProvider(
            [], [ResourceTemplate("weather://{city}", "weather", lambda v: f"sunny in {v['city']}")]
        )

        [contents] = await provider.read("weather://London")

        assert contents.content == "sunny in London"

    async def test_reserved_template_spans_slashes(self) -> None:
        provider = ResourceProvider([], [ResourceTemplate("file:///{+path}", "file", lambda v: v["path"])])

        [contents] = await provider.read("file:///a/b/c.txt")

        assert contents.content == "a/b/c.txt"

    async def test_plain_var_stops_at_slash(self) -> None:
        provider = ResourceProvider([], [ResourceTemplate("x://{seg}", "x", lambda v: v["seg"])])

        with pytest.raises(MCPResourceNotFoundError):
            await provider.read("x://a/b")  # plain {seg} won't match across '/'

    async def test_static_takes_precedence_over_template(self) -> None:
        provider = ResourceProvider(
            [Resource(uri="weather://London", name="exact", read=lambda: "cached")],
            [ResourceTemplate("weather://{city}", "weather", lambda v: f"live {v['city']}")],
        )

        [contents] = await provider.read("weather://London")

        assert contents.content == "cached"

    async def test_unknown_uri_raises(self) -> None:
        provider = ResourceProvider([], [])

        with pytest.raises(MCPResourceNotFoundError):
            await provider.read("nope://x")


class TestResourceCapability:
    def test_advertised_only_when_resources_present(self) -> None:
        agent = Agent("a", config=TestConfig("hi"))
        opts = NotificationOptions()

        without = MCPServer(agent).server.get_capabilities(opts, {})
        with_res = MCPServer(
            agent, resources=[Resource(uri="config://app", name="app", read=lambda: "hi")]
        ).server.get_capabilities(opts, {})

        assert without.resources is None
        assert with_res.resources is not None

    def test_templates_listed_only_when_present(self) -> None:
        agent = Agent("a", config=TestConfig("hi"))

        static_only = MCPServer(agent, resources=[Resource(uri="config://app", name="app", read=lambda: "hi")]).server
        with_tpl = MCPServer(agent, resource_templates=[ResourceTemplate("x://{v}", "x", lambda v: v["v"])]).server

        assert ListResourceTemplatesRequest not in static_only.request_handlers
        assert ListResourceTemplatesRequest in with_tpl.request_handlers
