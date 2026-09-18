# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import pytest
from dirty_equals import IsPartialDict
from pydantic import BaseModel

from ag2 import Agent, Inject, Variable
from ag2.events import ModelResponse
from ag2.mcp import MCPServer, Resource, mcp_tool
from ag2.mcp.executor import AgentExecutor, AskContext
from ag2.mcp.testing import connect
from ag2.mcp.tools import MCPRequestContext
from ag2.testing import TestConfig

from ._helpers import text_of


class _Weather(BaseModel):
    city: str


@pytest.mark.asyncio
class TestContextProvider:
    async def test_provider_invoked_and_forwarded(self) -> None:
        seen: dict[str, object] = {}

        async def provider(access: object) -> AskContext:
            seen["called"] = True
            seen["access"] = access
            return AskContext(variables={"x": 1}, prompt="custom system prompt")

        agent = Agent("greeter", config=TestConfig("hi"))
        executor = AgentExecutor(agent, stream_progress=False, context_provider=provider)

        result = await executor.call("ask", message="hello", context=None, request_context=None)

        assert seen["called"] is True
        # No auth context bound in this unit test, so the provider gets None.
        assert seen["access"] is None
        # The reply came back (the injected variables/prompt were accepted by ask()).
        assert text_of(result) == "hi"

    async def test_no_provider_is_stateless(self) -> None:
        agent = Agent("greeter", config=TestConfig("hi"))
        executor = AgentExecutor(agent, stream_progress=False)

        result = await executor.call("ask", message="hello", context=None, request_context=None)

        assert text_of(result) == "hi"

    async def test_empty_message_is_error(self) -> None:
        executor = AgentExecutor(Agent("greeter", config=TestConfig("hi")), stream_progress=False)

        result = await executor.call("ask", message="", context=None, request_context=None)

        assert result.is_error is True  # type: ignore[union-attr]

    async def test_provider_tools_field_forwarded(self) -> None:
        # AskContext.tools set (variables/prompt left None) exercises the tools
        # branch and the skipped variables/prompt branches.
        async def provider(access: object) -> AskContext:
            return AskContext(tools=[])

        executor = AgentExecutor(
            Agent("greeter", config=TestConfig("hi")), stream_progress=False, context_provider=provider
        )

        result = await executor.call("ask", message="hello", context=None, request_context=None)

        assert text_of(result) == "hi"

    async def test_structured_output_none_is_error(self) -> None:
        # Object response_schema + an empty model reply -> content() is None ->
        # to_structured_dict is None -> the executor returns an isError result.
        agent = Agent("weather", config=TestConfig(ModelResponse(message=None)), response_schema=_Weather)
        executor = AgentExecutor(agent, stream_progress=False)

        result = await executor.call("ask", message="weather?", context=None, request_context=None)

        assert result.is_error is True  # type: ignore[union-attr]


@pytest.mark.asyncio
class TestRequestScopedContext:
    async def test_resource_reads_resolve_variables_dependencies_and_mcp_context(self) -> None:
        async def provider(access: object) -> AskContext:
            return AskContext(variables={"tenant": "north"}, dependencies={"catalog": {"sku": "mug"}})

        async def read(
            tenant: Annotated[str, Variable("tenant")],
            catalog: Annotated[dict[str, str], Inject("catalog")],
            ctx: MCPRequestContext,
        ) -> str:
            return f"{tenant}:{catalog['sku']}:{ctx.session is not None}"

        server = MCPServer(
            Agent("g", config=TestConfig("hi")),
            resources=[Resource("catalog://card", "card", read)],
            context_provider=provider,
        )

        async with connect(server) as session:
            result = await session.read_resource("catalog://card")

        assert [c.model_dump() for c in result.contents] == [IsPartialDict({"text": "north:mug:True"})]

    async def test_resource_reads_do_not_receive_an_ag2_context(self) -> None:
        async def read(ctx: MCPRequestContext) -> str:
            return f"mcp={ctx.session is not None}"

        server = MCPServer(
            Agent("g", config=TestConfig("hi")),
            resources=[Resource("catalog://card", "card", read)],
        )

        async with connect(server) as session:
            result = await session.read_resource("catalog://card")

        assert [c.model_dump() for c in result.contents] == [IsPartialDict({"text": "mcp=True"})]

    async def test_resource_listing_resolves_request_scoped_metadata(self) -> None:
        async def provider(access: object) -> AskContext:
            return AskContext(variables={"tenant": "North catalog", "kind": "text/markdown"})

        server = MCPServer(
            Agent("g", config=TestConfig("hi")),
            resources=[
                Resource(
                    "catalog://card",
                    "card",
                    lambda: "body",
                    title=Variable("tenant"),
                    mime_type=Variable("kind"),
                )
            ],
            context_provider=provider,
        )

        async with connect(server) as session:
            listed = await session.list_resources()
            read = await session.read_resource("catalog://card")

        assert [r.model_dump() for r in listed.resources] == [
            IsPartialDict({"title": "North catalog", "mime_type": "text/markdown"})
        ]
        # The read result carries the resolved value too, not the Variable.
        assert [c.model_dump() for c in read.contents] == [IsPartialDict({"mime_type": "text/markdown"})]

    async def test_tool_listing_resolves_request_scoped_metadata(self) -> None:
        @mcp_tool(title=Variable("tool_title"), meta={"tenant": Variable("tenant")})
        async def read_scope() -> str:
            """Read request-scoped data."""
            return "ok"

        async def provider(access: object) -> AskContext:
            return AskContext(variables={"tenant": "north", "tool_title": "North catalog"})

        server = MCPServer(Agent("g", config=TestConfig("hi")), tools=[read_scope], context_provider=provider)

        async with connect(server) as session:
            result = await session.list_tools()

        listed = next(t for t in result.tools if t.name == "read_scope")
        assert listed.model_dump() == IsPartialDict({"title": "North catalog", "meta": {"tenant": "north"}})

    async def test_custom_tools_resolve_request_scoped_values(self) -> None:
        @mcp_tool
        async def read_scope(
            tenant: Annotated[str, Variable("tenant")],
            catalog: Annotated[dict[str, str], Inject("catalog")],
            ctx: MCPRequestContext,
        ) -> str:
            """Read request-scoped data."""
            return f"{tenant}:{catalog['sku']}:{ctx.session is not None}"

        async def provider(access: object) -> AskContext:
            return AskContext(variables={"tenant": "north"}, dependencies={"catalog": {"sku": "mug"}})

        server = MCPServer(Agent("g", config=TestConfig("hi")), tools=[read_scope], context_provider=provider)

        async with connect(server) as session:
            result = await session.call_tool("read_scope", {})

        assert text_of(result) == "north:mug:True"
