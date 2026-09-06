# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""MCP Apps: an agent serving an interactive document.

Everything here drives a real :class:`~ag2.mcp.MCPServer` through the real
protocol and asserts on what crosses the wire — the tool list a client receives,
the document it reads, the result it gets back. The one unavoidable exception is
:meth:`~ag2.mcp.apps.AppResource.runtime_script`, which is a string as far as Python is
concerned; what it does in a browser is not reachable from this suite and no
attempt is made to fake it.
"""

from dataclasses import dataclass
from typing import Any

import pytest
from mcp.client.extension import advertise
from mcp.server.apps import APP_MIME_TYPE, EXTENSION_ID, ResourceCsp, ResourcePermissions
from mcp.types import CallToolResult, DiscoverResult, TextContent, TextResourceContents
from mcp_types.version import LATEST_MODERN_VERSION

from ag2 import Agent
from ag2.mcp import AppResource, MCPServer, mcp_tool
from ag2.mcp.apps import TOOL_META_KEY
from ag2.mcp.errors import MCPAppURIError, MCPDuplicateAppURIError, MCPToolNameConflictError
from ag2.mcp.testing import connect, connect_modern
from ag2.testing import TestConfig

CARD = "<html><head><title>Card</title></head><body><div id='card'></div></body></html>"

# What an MCP Apps client advertises: the extension, listing the one MIME type a
# ``ui://`` document is ever served under.
_UI_AD = {EXTENSION_ID: {"mimeTypes": [APP_MIME_TYPE]}}


def _agent() -> Agent:
    return Agent("shopkeeper", config=TestConfig("hi"))


@dataclass
class Item:
    name: str
    price: int

    def __str__(self) -> str:
        return f"{self.name} costs {self.price}."


def _card_app(html: Any = CARD, **kwargs: Any) -> AppResource:
    """An app with one tool, built fresh so each test owns its own tool list."""
    app = AppResource("ui://shop/card", html, **kwargs)

    @app.tool
    async def show_item(item_id: str) -> Item:
        """Show a product card."""
        return Item(name=item_id, price=10)

    return app


def _tool(result: Any, name: str) -> Any:
    """The advertised tool named ``name`` in a ``tools/list`` result."""
    return next(t for t in result.tools if t.name == name)


def _text(result: CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def _body(contents: Any) -> str:
    assert isinstance(contents, TextResourceContents)
    return contents.text


@pytest.mark.asyncio
class TestTheBinding:
    async def test_a_tool_advertises_the_documents_uri(self) -> None:
        server = MCPServer(_agent(), apps=[_card_app()])

        async with connect(server, extensions=_UI_AD) as session:
            listed = await session.list_tools()

        assert _tool(listed, "show_item").meta == {"ui": {"resourceUri": "ui://shop/card"}}

    async def test_several_tools_on_one_app_all_carry_it(self) -> None:
        app = AppResource("ui://shop/card", CARD)

        @app.tool
        async def render() -> Item:
            """Render the card."""
            return Item("a", 1)

        @app.tool(visibility=["app"])
        async def refresh() -> Item:
            """Refresh the card in place."""
            return Item("b", 2)

        async with connect(MCPServer(_agent(), apps=[app]), extensions=_UI_AD) as session:
            listed = await session.list_tools()

        assert _tool(listed, "render").meta == {"ui": {"resourceUri": "ui://shop/card"}}
        assert _tool(listed, "refresh").meta == {"ui": {"resourceUri": "ui://shop/card", "visibility": ["app"]}}

    async def test_authors_own_meta_travels_alongside_the_binding(self) -> None:
        app = AppResource("ui://shop/card", CARD)

        @app.tool(meta={"com.example/hint": "left"})
        async def render() -> Item:
            """Render the card."""
            return Item("a", 1)

        async with connect(MCPServer(_agent(), apps=[app]), extensions=_UI_AD) as session:
            listed = await session.list_tools()

        assert _tool(listed, "render").meta == {
            "com.example/hint": "left",
            "ui": {"resourceUri": "ui://shop/card"},
        }

    async def test_a_ui_meta_key_is_refused(self) -> None:
        app = AppResource("ui://shop/card", CARD)

        with pytest.raises(ValueError, match="owns _meta"):

            @app.tool(meta={"ui": {"resourceUri": "ui://somewhere/else"}})
            async def render() -> Item:
                """Render the card."""
                return Item("a", 1)


@pytest.mark.asyncio
class TestTheDocument:
    async def test_it_is_read_with_the_required_mime_type(self) -> None:
        server = MCPServer(_agent(), apps=[_card_app(bootstrap=False)])

        async with connect(server) as session:
            read = await session.read_resource("ui://shop/card")

        assert read.contents[0].mime_type == APP_MIME_TYPE
        assert _body(read.contents[0]) == CARD

    async def test_it_is_hidden_from_the_listing_by_default(self) -> None:
        server = MCPServer(_agent(), apps=[_card_app()])

        async with connect(server) as session:
            listed = await session.list_resources()
            read = await session.read_resource("ui://shop/card")

        assert [str(r.uri) for r in listed.resources] == []
        # Hidden from browsing, still readable by URI — which is how a host gets it.
        assert _body(read.contents[0])

    async def test_it_is_listed_when_the_app_asks(self) -> None:
        server = MCPServer(_agent(), apps=[_card_app(listed=True)])

        async with connect(server) as session:
            listed = await session.list_resources()

        assert [str(r.uri) for r in listed.resources] == ["ui://shop/card"]

    async def test_a_callable_body_is_invoked_per_read(self) -> None:
        reads = []

        def body() -> str:
            reads.append(1)
            return f"<p>read {len(reads)}</p>"

        server = MCPServer(_agent(), apps=[_card_app(body)])

        async with connect(server) as session:
            first = await session.read_resource("ui://shop/card")
            second = await session.read_resource("ui://shop/card")

        assert "read 1" in _body(first.contents[0])
        assert "read 2" in _body(second.contents[0])

    async def test_an_async_callable_body_is_awaited(self) -> None:
        async def body() -> str:
            return "<p>async</p>"

        server = MCPServer(_agent(), apps=[_card_app(body)])

        async with connect(server) as session:
            read = await session.read_resource("ui://shop/card")

        assert "async" in _body(read.contents[0])

    async def test_sandbox_policy_reaches_the_documents_meta(self) -> None:
        app = _card_app(
            csp=ResourceCsp(connect_domains=["https://api.example.com"]),
            permissions=ResourcePermissions(camera={}),
            domain="https://shop.example.com",
            prefers_border=True,
        )

        async with connect(MCPServer(_agent(), apps=[app])) as session:
            read = await session.read_resource("ui://shop/card")

        assert read.contents[0].meta == {
            "ui": {
                "csp": {"connectDomains": ["https://api.example.com"]},
                "permissions": {"camera": {}},
                "domain": "https://shop.example.com",
                "prefersBorder": True,
            }
        }

    async def test_raw_meta_passes_through_alongside_the_policy(self) -> None:
        app = _card_app(prefers_border=False, meta={"com.example/rev": 3})

        async with connect(MCPServer(_agent(), apps=[app])) as session:
            read = await session.read_resource("ui://shop/card")

        assert read.contents[0].meta == {"com.example/rev": 3, "ui": {"prefersBorder": False}}

    async def test_title_and_description_reach_the_listing(self) -> None:
        app = _card_app(listed=True, title="Product card", description="A product card.")

        async with connect(MCPServer(_agent(), apps=[app])) as session:
            listed = await session.list_resources()

        assert (listed.resources[0].title, listed.resources[0].description) == (
            "Product card",
            "A product card.",
        )

    async def test_a_non_ui_uri_raises_at_construction(self) -> None:
        with pytest.raises(MCPAppURIError, match="ui://"):
            AppResource("https://shop/card", CARD)

    async def test_two_apps_sharing_a_uri_raise_at_construction(self) -> None:
        with pytest.raises(MCPDuplicateAppURIError, match="ui://shop/card"):
            MCPServer(_agent(), apps=[_card_app(), _card_app()])


@pytest.mark.asyncio
class TestTheRuntime:
    async def test_a_served_document_carries_it_by_default(self) -> None:
        server = MCPServer(_agent(), apps=[_card_app()])

        async with connect(server) as session:
            read = await session.read_resource("ui://shop/card")

        assert "window.ag2ui" in _body(read.contents[0])

    async def test_the_text_exposed_is_the_text_injected(self) -> None:
        app = _card_app()
        server = MCPServer(_agent(), apps=[app])

        async with connect(server) as session:
            read = await session.read_resource("ui://shop/card")

        assert app.runtime_script() in _body(read.contents[0])

    async def test_it_runs_before_the_authors_own_script(self) -> None:
        app = _card_app()
        server = MCPServer(_agent(), apps=[app])

        async with connect(server) as session:
            served = _body((await session.read_resource("ui://shop/card")).contents[0])

        # Injected inside <head>, so a script later in the document may call it.
        assert served.index("window.ag2ui") < served.index("<body>")

    async def test_it_lands_inside_a_head_tag_carrying_attributes(self) -> None:
        app = AppResource(
            "ui://shop/card",
            '<html><head\n  lang="en"><title>Card</title></head><body></body></html>',
        )
        server = MCPServer(_agent(), apps=[app])

        async with connect(server) as session:
            served = _body((await session.read_resource("ui://shop/card")).contents[0])

        # Inside <head>, not wedged between <html> and it.
        assert served.index("window.ag2ui") > served.index('lang="en"')
        assert served.index("window.ag2ui") < served.index("<title>")

    async def test_a_fragment_gets_it_at_the_front(self) -> None:
        app = AppResource("ui://shop/card", "<div id='card'></div>")
        server = MCPServer(_agent(), apps=[app])

        async with connect(server) as session:
            served = _body((await session.read_resource("ui://shop/card")).contents[0])

        assert served == app.runtime_script() + "<div id='card'></div>"

    async def test_turning_injection_off_leaves_the_document_byte_for_byte(self) -> None:
        server = MCPServer(_agent(), apps=[_card_app(bootstrap=False)])

        async with connect(server) as session:
            read = await session.read_resource("ui://shop/card")

        assert _body(read.contents[0]) == CARD

    async def test_it_makes_no_network_call_and_constructs_no_function(self) -> None:
        # The specification's CSP gives a document no network access unless the app
        # declared connect domains, and forbids eval / dynamic function construction.
        script = _card_app().runtime_script()

        for forbidden in ("eval(", "new Function", "fetch(", "XMLHttpRequest", "WebSocket", "import("):
            assert forbidden not in script

    async def test_it_announces_the_dialect_revision(self) -> None:
        # ``ui/initialize`` params require protocolVersion; without it the handshake
        # fails validation at the host and every later message is discarded.
        script = _card_app().runtime_script()

        assert '"ui/initialize"' in script
        assert 'protocolVersion: "2026-01-26"' in script

    async def test_it_cannot_be_closed_by_the_apps_own_name(self) -> None:
        app = AppResource("ui://x/y", "<p>hi</p>", name="</script><script>alert(1)</script>")

        assert "</script><script>" not in app.runtime_script()


@pytest.mark.asyncio
class TestDegradation:
    async def test_a_client_without_the_extension_does_not_see_the_binding(self) -> None:
        server = MCPServer(_agent(), apps=[_card_app()])

        async with connect(server) as session:
            listed = await session.list_tools()
            called = await session.call_tool("show_item", {"item_id": "mug"})

        assert _tool(listed, "show_item").meta is None
        # Still listed, still callable, and the text is the whole answer for it.
        assert _text(called) == "mug costs 10."

    async def test_the_extension_without_the_mime_type_is_not_support(self) -> None:
        # Advertising the identifier alone says nothing about being able to render
        # ``text/html;profile=mcp-app``, which is the only thing the binding promises.
        server = MCPServer(_agent(), apps=[_card_app()])

        async with connect(server, extensions={EXTENSION_ID: {}}) as session:
            listed = await session.list_tools()

        assert _tool(listed, "show_item").meta is None

    async def test_other_meta_survives_the_withholding(self) -> None:
        app = AppResource("ui://shop/card", CARD)

        @app.tool(meta={"com.example/hint": "left"})
        async def render() -> Item:
            """Render the card."""
            return Item("a", 1)

        async with connect(MCPServer(_agent(), apps=[app])) as session:
            listed = await session.list_tools()

        assert _tool(listed, "render").meta == {"com.example/hint": "left"}

    async def test_a_modern_client_that_advertised_sees_the_binding(self) -> None:
        server = MCPServer(_agent(), apps=[_card_app()])

        async with connect_modern(server, extensions=[advertise(EXTENSION_ID, {"mimeTypes": [APP_MIME_TYPE]})]) as s:
            listed = await s.list_tools()

        assert _tool(listed, "show_item").meta == {"ui": {"resourceUri": "ui://shop/card"}}

    async def test_a_modern_client_that_advertised_nothing_does_not_see_it(self) -> None:
        server = MCPServer(_agent(), apps=[_card_app()])

        async with connect_modern(server) as session:
            listed = await session.list_tools()
            called = await session.call_tool("show_item", {"item_id": "mug"})

        assert _tool(listed, "show_item").meta is None
        assert _text(called) == "mug costs 10."

    async def test_a_server_holding_an_app_advertises_the_extension(self) -> None:
        server = MCPServer(_agent(), apps=[_card_app()])

        async with connect_modern(server) as session:
            discovered = DiscoverResult.model_validate(await session.send_discover(LATEST_MODERN_VERSION))

        assert discovered.capabilities.extensions == {EXTENSION_ID: {}}

    async def test_an_explicit_setting_wins_over_the_automatic_one(self) -> None:
        server = MCPServer(_agent(), apps=[_card_app()], extensions={EXTENSION_ID: {"mimeTypes": ["x"]}})

        async with connect_modern(server) as session:
            discovered = DiscoverResult.model_validate(await session.send_discover(LATEST_MODERN_VERSION))

        assert discovered.capabilities.extensions == {EXTENSION_ID: {"mimeTypes": ["x"]}}

    async def test_a_server_without_apps_advertises_nothing(self) -> None:
        @mcp_tool
        async def plain() -> str:
            """A tool with no document."""
            return "ok"

        async with connect_modern(MCPServer(_agent(), tools=[plain])) as session:
            discovered = DiscoverResult.model_validate(await session.send_discover(LATEST_MODERN_VERSION))

        assert discovered.capabilities.extensions is None


@pytest.mark.asyncio
class TestWhichToolAnswered:
    async def test_a_result_names_the_tool_that_produced_it(self) -> None:
        server = MCPServer(_agent(), apps=[_card_app()])

        async with connect(server, extensions=_UI_AD) as session:
            result = await session.call_tool("show_item", {"item_id": "mug"})

        assert result.meta == {TOOL_META_KEY: "show_item"}
        assert result.structured_content == {"name": "mug", "price": 10}

    async def test_a_renamed_tool_is_stamped_with_its_advertised_name(self) -> None:
        app = AppResource("ui://shop/card", CARD)

        @app.tool(name="show")
        async def show_item() -> Item:
            """Show a product card."""
            return Item("mug", 10)

        async with connect(MCPServer(_agent(), apps=[app]), extensions=_UI_AD) as session:
            result = await session.call_tool("show", {})

        assert result.meta == {TOOL_META_KEY: "show"}

    async def test_a_fully_formed_result_is_stamped_too(self) -> None:
        app = AppResource("ui://shop/card", CARD)

        @app.tool
        async def render() -> CallToolResult:
            """Render, stating text and data separately."""
            return CallToolResult(
                content=[TextContent(type="text", text="a mug")],
                structuredContent={"name": "mug"},
            )

        async with connect(MCPServer(_agent(), apps=[app]), extensions=_UI_AD) as session:
            result = await session.call_tool("render", {})

        assert result.meta == {TOOL_META_KEY: "render"}
        assert _text(result) == "a mug"
        assert result.structured_content == {"name": "mug"}

    async def test_an_author_who_sets_the_key_keeps_their_value(self) -> None:
        app = AppResource("ui://shop/card", CARD)

        @app.tool
        async def render() -> CallToolResult:
            """Route by my own scheme rather than by tool name."""
            return CallToolResult(content=[], _meta={TOOL_META_KEY: "card:refresh"})

        async with connect(MCPServer(_agent(), apps=[app]), extensions=_UI_AD) as session:
            result = await session.call_tool("render", {})

        assert result.meta == {TOOL_META_KEY: "card:refresh"}

    async def test_other_meta_the_author_set_is_preserved_alongside_the_stamp(self) -> None:
        app = AppResource("ui://shop/card", CARD)

        @app.tool
        async def render() -> CallToolResult:
            """Render with metadata of my own."""
            return CallToolResult(content=[], _meta={"com.example/rev": 3})

        async with connect(MCPServer(_agent(), apps=[app]), extensions=_UI_AD) as session:
            result = await session.call_tool("render", {})

        assert result.meta == {"com.example/rev": 3, TOOL_META_KEY: "render"}

    async def test_a_tool_outside_an_app_carries_no_stamp(self) -> None:
        @mcp_tool
        async def plain() -> Item:
            """A tool with no document."""
            return Item("mug", 10)

        async with connect(MCPServer(_agent(), tools=[plain])) as session:
            result = await session.call_tool("plain", {})

        assert result.meta is None

    async def test_the_stamp_survives_the_binding_being_withheld(self) -> None:
        # The stamp is on the *result* and the binding is on the *tool*; a client
        # without UI support loses one and not the other.
        server = MCPServer(_agent(), apps=[_card_app()])

        async with connect(server) as session:
            result = await session.call_tool("show_item", {"item_id": "mug"})

        assert result.meta == {TOOL_META_KEY: "show_item"}


@pytest.mark.asyncio
class TestDecomposition:
    async def test_registering_the_pieces_by_hand_is_the_same_server(self) -> None:
        """``apps=[app]`` is shorthand, and this holds it to that.

        Compares what a client actually receives — the advertised tool, the
        resource listing, the document read — between the two registrations.
        """
        shorthand = MCPServer(_agent(), apps=[_card_app(listed=True)])
        by_hand_app = _card_app(listed=True)
        by_hand = MCPServer(_agent(), tools=by_hand_app.tools, resources=[by_hand_app.resource])

        async with connect(shorthand, extensions=_UI_AD) as session:
            a_tools = await session.list_tools()
            a_resources = await session.list_resources()
            a_read = await session.read_resource("ui://shop/card")
            a_call = await session.call_tool("show_item", {"item_id": "mug"})
        async with connect(by_hand, extensions=_UI_AD) as session:
            b_tools = await session.list_tools()
            b_resources = await session.list_resources()
            b_read = await session.read_resource("ui://shop/card")
            b_call = await session.call_tool("show_item", {"item_id": "mug"})

        assert a_tools.model_dump() == b_tools.model_dump()
        assert a_resources.model_dump() == b_resources.model_dump()
        assert a_read.model_dump() == b_read.model_dump()
        assert a_call.model_dump() == b_call.model_dump()

    async def test_the_extension_advertisement_follows_the_tools_too(self) -> None:
        app = _card_app()
        by_hand = MCPServer(_agent(), tools=app.tools, resources=[app.resource])

        async with connect_modern(by_hand) as session:
            discovered = DiscoverResult.model_validate(await session.send_discover(LATEST_MODERN_VERSION))

        assert discovered.capabilities.extensions == {EXTENSION_ID: {}}

    async def test_an_apps_tool_names_still_conflict_with_the_agents(self) -> None:
        app = AppResource("ui://shop/card", CARD)

        @app.tool(name="ask")
        async def render() -> Item:
            """Collide with the conversational tool."""
            return Item("a", 1)

        with pytest.raises(MCPToolNameConflictError, match="conversational tool"):
            MCPServer(_agent(), apps=[app])


@pytest.mark.asyncio
class TestOrdinaryToolsAreUnaffected:
    async def test_a_plain_tool_alongside_an_app_keeps_its_shape(self) -> None:
        @mcp_tool
        async def plain(word: str) -> str:
            """Echo a word."""
            return word

        server = MCPServer(_agent(), tools=[plain], apps=[_card_app()])

        async with connect(server, extensions=_UI_AD) as session:
            listed = await session.list_tools()
            result = await session.call_tool("plain", {"word": "hi"})

        assert _tool(listed, "plain").meta is None
        assert _tool(listed, "plain").output_schema is None
        assert _text(result) == "hi"
        assert result.meta is None
