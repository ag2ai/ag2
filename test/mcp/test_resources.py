# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.server.lowlevel import NotificationOptions
from mcp.types import TextResourceContents

from ag2.mcp import MCPServer, Resource, ResourceTemplate
from ag2.mcp.errors import MCPResourceNotFoundError
from ag2.mcp.resources import ResourceProvider
from ag2.mcp.testing import connect, connect_modern

from ._helpers import greeter

# ``mcp`` 2.0 keys request handlers by method string rather than by request type.
_TEMPLATES_LIST = "resources/templates/list"


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


@pytest.mark.asyncio
@pytest.mark.parametrize("connect_client", [connect, connect_modern], ids=["handshake", "modern"])
class TestResourceTemplateRead:
    @pytest.mark.parametrize(
        ("uri_template", "uri", "relative_path"),
        [
            ("files:///{path}", "files:///readme.txt", "readme.txt"),
            ("files:///{path}", "files:///hello%20world.txt", "hello world.txt"),
            ("files:///{path}", "files:///%E6%8A%A5%E5%91%8A.txt", "报告.txt"),
            ("files:///{path}", "files:///a+b.txt", "a+b.txt"),
            ("files:///{path}", "files:///a%2Bb.txt", "a+b.txt"),
            ("files:///{path}", "files:///literal%2520name.txt", "literal%20name.txt"),
            ("files:///{path}", "files:///nested%2Freport.txt", "nested/report.txt"),
            ("files:///{+path}", "files:///nested/hello%20world.txt", "nested/hello world.txt"),
            ("files:///{+path}", "files:///nested/%E6%8A%A5%E5%91%8A.txt", "nested/报告.txt"),
            ("files:///{+path}", "files:///literal%2520name.txt", "literal%20name.txt"),
            ("files:///my%20files/{path}", "files:///my%20files/hello%20world.txt", "hello world.txt"),
        ],
    )
    async def test_reads_file_from_template(
        self,
        connect_client: Callable[[MCPServer], AbstractAsyncContextManager[ClientSession]],
        tmp_path: Path,
        uri_template: str,
        uri: str,
        relative_path: str,
    ) -> None:
        content = f"Read {relative_path}"
        resource_path = tmp_path / relative_path
        resource_path.parent.mkdir(parents=True, exist_ok=True)
        resource_path.write_text(content, encoding="utf-8")
        server = MCPServer(
            greeter(),
            resource_templates=[
                ResourceTemplate(
                    uri_template,
                    "file",
                    lambda variables: (tmp_path / variables["path"]).read_text(encoding="utf-8"),
                )
            ],
        )

        async with connect_client(server) as session:
            result = await session.read_resource(uri)

        assert result.contents == [TextResourceContents(uri=uri, mimeType="text/plain", text=content)]

    async def test_async_reader_receives_decoded_variables(
        self, connect_client: Callable[[MCPServer], AbstractAsyncContextManager[ClientSession]]
    ) -> None:
        async def read_weather(variables: dict[str, str]) -> str:
            return f"sunny in {variables['city']}"

        uri = "weather:///New%20York"
        server = MCPServer(
            greeter(), resource_templates=[ResourceTemplate("weather:///{city}", "weather", read_weather)]
        )

        async with connect_client(server) as session:
            result = await session.read_resource(uri)

        assert result.contents == [TextResourceContents(uri=uri, mimeType="text/plain", text="sunny in New York")]

    async def test_static_resource_keeps_encoded_uri_and_takes_precedence(
        self, connect_client: Callable[[MCPServer], AbstractAsyncContextManager[ClientSession]]
    ) -> None:
        uri = "files:///hello%20world.txt"
        server = MCPServer(
            greeter(),
            resources=[Resource(uri=uri, name="exact", read=lambda: "cached")],
            resource_templates=[ResourceTemplate("files:///{path}", "file", lambda variables: "live")],
        )

        async with connect_client(server) as session:
            result = await session.read_resource(uri)

        assert result.contents == [TextResourceContents(uri=uri, mimeType="text/plain", text="cached")]


class TestResourceCapability:
    def test_advertised_only_when_resources_present(self) -> None:
        agent = greeter()
        opts = NotificationOptions()

        without = MCPServer(agent).server.get_capabilities(opts, {})
        with_res = MCPServer(
            agent, resources=[Resource(uri="config://app", name="app", read=lambda: "hi")]
        ).server.get_capabilities(opts, {})

        assert without.resources is None
        assert with_res.resources is not None

    def test_templates_listed_only_when_present(self) -> None:
        agent = greeter()

        static_only = MCPServer(agent, resources=[Resource(uri="config://app", name="app", read=lambda: "hi")]).server
        with_tpl = MCPServer(agent, resource_templates=[ResourceTemplate("x://{v}", "x", lambda v: v["v"])]).server

        assert static_only.get_request_handler(_TEMPLATES_LIST) is None
        assert with_tpl.get_request_handler(_TEMPLATES_LIST) is not None
