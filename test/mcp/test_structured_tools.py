# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

import pytest
from dirty_equals import IsPartialDict
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel

from ag2.mcp import MCPFunctionTool, MCPServer, Resource, mcp_tool
from ag2.mcp.testing import connect

from ._helpers import make_agent, tool_named


class Item(BaseModel):
    id: str
    price: float

    def __str__(self) -> str:
        return f"Item {self.id} costs {self.price}"


@dataclass
class Crate:
    label: str


@mcp_tool
async def get_item(item_id: str) -> Item:
    """Return an item."""
    return Item(id=item_id, price=9.5)


@mcp_tool
async def get_crate() -> Crate:
    """Return a crate."""
    return Crate(label="wood")


@mcp_tool
async def get_mapping() -> dict[str, Any]:
    """Return a bare mapping."""
    return {"total": 3}


@mcp_tool
async def get_both() -> Item:
    """State the text and the data separately."""
    return CallToolResult(
        content=[TextContent(type="text", text="one item, cheap")],
        structuredContent={"id": "a", "price": 1.0},
    )


@mcp_tool
async def get_blocks() -> list[TextContent]:
    """Return content blocks under a parameterised annotation."""
    return [TextContent(type="text", text="block")]


@mcp_tool
async def get_count() -> int:
    """Return a bare int, which the ladder cannot map onto a result."""
    return 7


@mcp_tool
async def get_raw() -> CallToolResult:
    """Return a fully-formed result."""
    return CallToolResult(
        content=[TextContent(type="text", text="verbatim")],
        structuredContent={"untouched": True},
        isError=True,
    )


@pytest.mark.asyncio
class TestTypedReturn:
    async def test_model_annotation_is_advertised_as_output_schema(self) -> None:
        async with connect(MCPServer(make_agent(), tools=[get_item])) as session:
            tools = await session.list_tools()

        assert tool_named(tools, "get_item").output_schema == IsPartialDict({
            "type": "object",
            "required": ["id", "price"],
        })

    async def test_model_return_produces_structured_content_and_its_string_form(self) -> None:
        async with connect(MCPServer(make_agent(), tools=[get_item])) as session:
            result = await session.call_tool("get_item", {"item_id": "a"})

        assert result.structured_content == {"id": "a", "price": 9.5}
        assert result.content == [TextContent(type="text", text="Item a costs 9.5")]

    async def test_dataclass_return_is_structured_too(self) -> None:
        async with connect(MCPServer(make_agent(), tools=[get_crate])) as session:
            tools = await session.list_tools()
            result = await session.call_tool("get_crate", {})

        assert tool_named(tools, "get_crate").output_schema == IsPartialDict({"type": "object", "required": ["label"]})
        assert result.structured_content == {"label": "wood"}


@pytest.mark.asyncio
async def test_a_mapping_is_structured_content_with_no_schema() -> None:
    async with connect(MCPServer(make_agent(), tools=[get_mapping])) as session:
        tools = await session.list_tools()
        result = await session.call_tool("get_mapping", {})

    assert tool_named(tools, "get_mapping").output_schema is None
    assert result.structured_content == {"total": 3}
    assert result.content == [TextContent(type="text", text='{"total": 3}')]


@pytest.mark.asyncio
class TestTextAndDataStatedSeparately:
    async def test_text_and_data_are_taken_verbatim(self) -> None:
        async with connect(MCPServer(make_agent(), tools=[get_both])) as session:
            result = await session.call_tool("get_both", {})

        assert result.content == [TextContent(type="text", text="one item, cheap")]
        assert result.structured_content == {"id": "a", "price": 1.0}

    async def test_schema_still_comes_from_the_annotation(self) -> None:
        # The schema follows the *annotation*, decided once at decoration time —
        # so assembling the result by hand does not withdraw the promise the
        # listing already made. Annotating ``-> CallToolResult`` is what opts out.
        async with connect(MCPServer(make_agent(), tools=[get_both])) as session:
            tools = await session.list_tools()

        assert tool_named(tools, "get_both").output_schema == IsPartialDict({"type": "object"})


@pytest.mark.asyncio
class TestCallToolResultReturn:
    async def test_it_reaches_the_wire_unchanged(self) -> None:
        async with connect(MCPServer(make_agent(), tools=[get_raw])) as session:
            result = await session.call_tool("get_raw", {})

        assert result.content == [TextContent(type="text", text="verbatim")]
        assert result.structured_content == {"untouched": True}
        assert result.is_error is True

    async def test_nothing_is_derived_from_the_annotation(self) -> None:
        async with connect(MCPServer(make_agent(), tools=[get_raw])) as session:
            tools = await session.list_tools()

        assert tool_named(tools, "get_raw").output_schema is None


@pytest.mark.asyncio
async def test_a_parameterised_annotation_advertises_no_schema() -> None:
    # ``list[TextContent]`` is not a class the schema can be read off. Until
    # 3.11 it nonetheless answered ``isinstance(x, type)`` with ``True``,
    # so classifying it must not reach ``issubclass``.
    async with connect(MCPServer(make_agent(), tools=[get_blocks])) as session:
        tools = await session.list_tools()
        result = await session.call_tool("get_blocks", {})

    assert tool_named(tools, "get_blocks").output_schema is None
    assert result.content == [TextContent(type="text", text="block")]
    assert result.structured_content is None


@pytest.mark.asyncio
async def test_a_return_the_ladder_cannot_map_reaches_the_caller_as_an_error() -> None:
    # The one arm of the ladder that refuses: an author who returns something
    # that is not a ToolResult should be told which type, not handed a silent
    # empty result.
    async with connect(MCPServer(make_agent(), tools=[get_count])) as session:
        result = await session.call_tool("get_count", {})

    assert result.is_error is True
    assert result.content == [
        TextContent(type="text", text="A tool handler cannot return int; see ag2.mcp.tools.ToolResult.")
    ]


@pytest.mark.asyncio
class TestToolMetadata:
    async def test_metadata_is_advertised(self) -> None:
        tool = MCPFunctionTool("m", "with meta", lambda a, c: "ok", meta={"vendor/key": {"n": 1}})

        async with connect(MCPServer(make_agent(), tools=[tool])) as session:
            tools = await session.list_tools()

        assert tool_named(tools, "m").meta == {"vendor/key": {"n": 1}}

    async def test_no_metadata_puts_no_meta_on_the_wire(self) -> None:
        tool = MCPFunctionTool("m", "no meta", lambda a, c: "ok")

        async with connect(MCPServer(make_agent(), tools=[tool])) as session:
            tools = await session.list_tools()

        assert tool_named(tools, "m").meta is None

    async def test_decorator_passes_metadata_through(self) -> None:
        @mcp_tool(meta={"vendor/key": "v"})
        async def tagged() -> str:
            """Tagged."""
            return "ok"

        async with connect(MCPServer(make_agent(), tools=[tagged])) as session:
            tools = await session.list_tools()

        assert tool_named(tools, "tagged").meta == {"vendor/key": "v"}


@pytest.mark.asyncio
class TestResourceMetadata:
    async def test_metadata_is_carried_in_a_read_result(self) -> None:
        resource = Resource("res://a", "a", lambda: "body", meta={"vendor/key": {"n": 1}})

        async with connect(MCPServer(make_agent(), resources=[resource])) as session:
            read = await session.read_resource("res://a")

        assert read.contents[0].meta == {"vendor/key": {"n": 1}}

    async def test_no_metadata_puts_no_meta_on_the_wire(self) -> None:
        resource = Resource("res://a", "a", lambda: "body")

        async with connect(MCPServer(make_agent(), resources=[resource])) as session:
            read = await session.read_resource("res://a")
            listed = await session.list_resources()

        assert read.contents[0].meta is None
        assert listed.resources[0].meta is None
