# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A deterministic served tool asking the calling client for something.

The conversational tool is not the only thing a server exposes. A plain
``@mcp_tool`` function served alongside it can ask too, and for these the SDK's
own resolver mechanism fits: a resolver body is cheap to re-run, so nothing has
to be held between rounds.

**The contract is the opposite of the conversational tool's**, which is the whole
reason these live apart. A conversational turn is held open and resumes exactly
where it stopped, and nothing about it re-runs; a resolver runs again on every
round. These tests pin both halves of that sentence, because an author who
assumes the wrong one writes a side effect that fires once per round.
"""

from typing import Annotated, Any

import pytest
from mcp.client.session import ClientRequestContext
from mcp.server.mcpserver import Elicit, Resolve
from mcp.types import ElicitRequest, ElicitRequestParams, ElicitResult, InputRequiredResult, TextContent
from pydantic import BaseModel

from ag2 import Agent
from ag2.mcp import MCPServer, mcp_tool
from ag2.mcp.testing import connect, connect_modern
from ag2.testing import TestConfig


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


@pytest.fixture(autouse=True)
def _reset() -> None:
    RESOLVER_RUNS.clear()
    BODY_RUNS.clear()


def served() -> MCPServer:
    return MCPServer(Agent("host", config=TestConfig("unused")), tools=[paint])


async def accepts_blue(context: ClientRequestContext, params: ElicitRequestParams) -> ElicitResult:
    return ElicitResult(action="accept", content={"answer": "blue"})


async def _call(session: Any, **kwargs: Any) -> Any:
    return await session.call_tool("paint", {"room": "kitchen"}, allow_input_required=True, **kwargs)


@pytest.mark.asyncio
class TestTheModernEraRoundTrip:
    async def test_the_question_comes_back_as_the_calls_result(self) -> None:
        async with connect_modern(served(), elicitation_callback=accepts_blue) as session:
            first = await _call(session)

        assert isinstance(first, InputRequiredResult)
        ((_key, request),) = (first.input_requests or {}).items()
        assert isinstance(request, ElicitRequest)
        assert request.params.message == "What colour?"
        assert BODY_RUNS == [], "the body ran before its parameter was resolved"

    async def test_the_answered_retry_completes_the_call(self) -> None:
        async with connect_modern(served(), elicitation_callback=accepts_blue) as session:
            first = await _call(session)
            assert isinstance(first, InputRequiredResult)
            (key,) = (first.input_requests or {}).keys()
            final = await _call(
                session,
                input_responses={key: ElicitResult(action="accept", content={"answer": "blue"})},
                request_state=first.request_state,
            )

        assert not isinstance(final, InputRequiredResult)
        assert final.is_error is False
        assert final.content == [TextContent(type="text", text="painted kitchen blue")]

    async def test_the_resolver_re_runs_every_round_and_the_body_runs_once(self) -> None:
        """The documented contract, pinned: this is why a resolver must be idempotent."""
        async with connect_modern(served(), elicitation_callback=accepts_blue) as session:
            first = await _call(session)
            assert isinstance(first, InputRequiredResult)
            (key,) = (first.input_requests or {}).keys()
            await _call(
                session,
                input_responses={key: ElicitResult(action="accept", content={"answer": "blue"})},
                request_state=first.request_state,
            )

        assert RESOLVER_RUNS == ["resolver", "resolver"], "the resolver body did not re-run on the answered round"
        assert BODY_RUNS == ["body"], "the tool body ran on a round that had nothing to run with"

    async def test_the_resolved_parameter_is_not_advertised(self) -> None:
        """The caller supplies ``room``; ``colour`` is what the tool goes and asks for."""
        async with connect_modern(served(), elicitation_callback=accepts_blue) as session:
            tools = {t.name: t for t in (await session.list_tools()).tools}

        schema = tools["paint"].input_schema
        assert set(schema["properties"]) == {"room"}


@pytest.mark.asyncio
class TestTheHandshakeEra:
    async def test_the_question_is_answered_inline_with_no_second_round(self) -> None:
        """That era has a back-channel, so the answer arrives inside the one call."""
        async with connect(served(), elicitation_callback=accepts_blue) as session:
            result = await session.call_tool("paint", {"room": "kitchen"})

        assert result.is_error is False
        assert result.content == [TextContent(type="text", text="painted kitchen blue")]
        assert RESOLVER_RUNS == ["resolver"], "the inline path ran the resolver more than once"
        assert BODY_RUNS == ["body"]
