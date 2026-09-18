# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A deterministic served tool asking the calling client for something.

A plain ``@mcp_tool`` function served next to the conversational tool can ask
too, through the SDK's resolver mechanism. The contract is the opposite of the
conversational tool's, which is why the two live apart: a resolver runs again on
every round, while a held turn resumes where it stopped and re-runs nothing.
"""

from typing import Annotated, Any

import pytest
from mcp.client.session import ClientRequestContext
from mcp.server.apps import APP_MIME_TYPE, EXTENSION_ID
from mcp.server.mcpserver import Elicit, ListRoots, Resolve
from mcp.types import (
    ElicitRequest,
    ElicitRequestParams,
    ElicitResult,
    InputRequiredResult,
    ListRootsRequest,
    ListRootsResult,
    Root,
    TextContent,
)
from pydantic import BaseModel

from ag2.mcp import MCPApp, MCPServer, mcp_tool
from ag2.mcp.testing import connect, connect_modern

from ._helpers import greeter, outstanding


class Colour(BaseModel):
    answer: str


# Module level, so the resolver's wire key is stable across rounds the way the
# SDK derives it (``module:qualname``); a closure per test would not be.
RESOLVER_RUNS: list[str] = []
BODY_RUNS: list[str] = []


def pick_colour() -> Elicit[Colour]:
    """Ask the client's human, once per round this resolver is still unanswered."""
    RESOLVER_RUNS.append("resolver")
    return Elicit("What colour?", Colour)


@mcp_tool
def paint(room: str, colour: Annotated[Colour, Resolve(pick_colour)]) -> str:
    """Paint a room the colour the client's human picked."""
    BODY_RUNS.append("body")
    return f"painted {room} {colour.answer}"


def project_roots() -> ListRoots:
    """Read the client's roots — available here, and *not* to a conversational turn.

    The served agent has no filesystem of its own to scope, so the spec excluded
    roots there; a deterministic tool may well be the code that wants them.
    """
    return ListRoots()


@mcp_tool
def where(roots: Annotated[ListRootsResult, Resolve(project_roots)]) -> str:
    """Report the directories the calling client says its work lives in."""
    return ", ".join(str(root.uri) for root in roots.roots)


@pytest.fixture(autouse=True)
def _reset() -> None:
    RESOLVER_RUNS.clear()
    BODY_RUNS.clear()


def served() -> MCPServer:
    return MCPServer(greeter("unused", name="host"), tools=[paint])


def rooted() -> MCPServer:
    return MCPServer(greeter("unused", name="host"), tools=[where])


async def lists_one_root(context: ClientRequestContext) -> ListRootsResult:
    return ListRootsResult(roots=[Root(uri="file:///work")])


def accepts(colour: str) -> ElicitResult:
    """A ``Colour`` the client's human picked, as the wire carries it."""
    return ElicitResult(action="accept", content={"answer": colour})


async def accepts_blue(context: ClientRequestContext, params: ElicitRequestParams) -> ElicitResult:
    return accepts("blue")


async def _call(session: Any, **kwargs: Any) -> Any:
    return await session.call_tool("paint", {"room": "kitchen"}, allow_input_required=True, **kwargs)


async def _answer(session: Any, paused: InputRequiredResult, response: Any) -> Any:
    """Retry the paused call, answering the question it came back with."""
    key, _request = outstanding(paused)
    return await _call(session, input_responses={key: response}, request_state=paused.request_state)


@pytest.mark.asyncio
class TestTheModernEraRoundTrip:
    async def test_the_question_comes_back_as_the_calls_result(self) -> None:
        async with connect_modern(served(), elicitation_callback=accepts_blue) as session:
            first = await _call(session)

        assert isinstance(first, InputRequiredResult)
        _key, request = outstanding(first)
        assert isinstance(request, ElicitRequest)
        assert request.params.message == "What colour?"
        assert BODY_RUNS == [], "the body ran before its parameter was resolved"

    async def test_the_answered_retry_completes_the_call(self) -> None:
        async with connect_modern(served(), elicitation_callback=accepts_blue) as session:
            first = await _call(session)
            assert isinstance(first, InputRequiredResult)
            final = await _answer(session, first, accepts("blue"))

        assert not isinstance(final, InputRequiredResult)
        assert final.is_error is False
        assert final.content == [TextContent(type="text", text="painted kitchen blue")]

    async def test_the_resolver_re_runs_every_round_and_the_body_runs_once(self) -> None:
        """The documented contract, pinned: this is why a resolver must be idempotent."""
        async with connect_modern(served(), elicitation_callback=accepts_blue) as session:
            first = await _call(session)
            assert isinstance(first, InputRequiredResult)
            await _answer(session, first, accepts("blue"))

        assert RESOLVER_RUNS == ["resolver", "resolver"], "the resolver body did not re-run on the answered round"
        assert BODY_RUNS == ["body"], "the tool body ran on a round that had nothing to run with"

    async def test_a_resolver_can_ask_for_the_clients_roots(self) -> None:
        """Elicitation is not the only marker the pass-through carries."""
        async with connect_modern(rooted(), list_roots_callback=lists_one_root) as session:
            first = await session.call_tool("where", {}, allow_input_required=True)
            assert isinstance(first, InputRequiredResult)
            key, request = outstanding(first)
            assert isinstance(request, ListRootsRequest)
            final = await session.call_tool(
                "where",
                {},
                allow_input_required=True,
                input_responses={key: ListRootsResult(roots=[Root(uri="file:///work")])},
                request_state=first.request_state,
            )

        assert not isinstance(final, InputRequiredResult)
        assert final.is_error is False
        assert final.content == [TextContent(type="text", text="file:///work")]

    async def test_the_resolved_parameter_is_not_advertised(self) -> None:
        """The caller supplies ``room``; ``colour`` is what the tool goes and asks for."""
        async with connect_modern(served(), elicitation_callback=accepts_blue) as session:
            tools = {t.name: t for t in (await session.list_tools()).tools}

        schema = tools["paint"].input_schema
        assert set(schema["properties"]) == {"room"}


@pytest.mark.asyncio
async def test_the_handshake_era_answers_the_question_inline_with_no_second_round() -> None:
    """That era has a back-channel, so the answer arrives inside the one call."""
    async with connect(served(), elicitation_callback=accepts_blue) as session:
        result = await session.call_tool("paint", {"room": "kitchen"})

    assert result.is_error is False
    assert result.content == [TextContent(type="text", text="painted kitchen blue")]
    assert RESOLVER_RUNS == ["resolver"], "the inline path ran the resolver more than once"
    assert BODY_RUNS == ["body"]


@pytest.mark.asyncio
async def test_an_app_bound_tool_retains_its_resolver() -> None:
    """MCP Apps decoration must preserve a deterministic tool's private resolver plan."""
    app = MCPApp("ui://paint/card", "<p>Paint</p>")
    app.tool(paint)
    server = MCPServer(greeter("unused", name="host"), apps=[app])

    async with connect(
        server,
        elicitation_callback=accepts_blue,
        extensions={EXTENSION_ID: {"mimeTypes": [APP_MIME_TYPE]}},
    ) as session:
        listed = {tool.name: tool for tool in (await session.list_tools()).tools}
        result = await session.call_tool("paint", {"room": "kitchen"})

    assert listed["paint"].meta == {"ui": {"resourceUri": "ui://paint/card"}}
    assert result.content == [TextContent(type="text", text="painted kitchen blue")]
