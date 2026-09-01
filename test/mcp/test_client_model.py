# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A served agent whose reasoning runs on the *calling client's* model.

The point of the feature is a deployment holding no model credentials that can
still serve an agent needing one. The point of these tests is that it is a
decision: off unless asked for, never quietly substituted, and never quietly
absent.

Both eras are covered, because the transport differs and nothing else does — a
standalone ``sampling/createMessage`` up to 2025-11-25, and from 2026-07-28 a
request returned as the call's result and answered by the client's retry.
"""

from typing import Any

import pytest
from mcp.client.session import ClientRequestContext
from mcp.types import (
    CreateMessageRequest,
    CreateMessageRequestParams,
    CreateMessageResult,
    InputRequiredResult,
    TextContent,
)

from ag2 import Agent
from ag2.mcp import ClientModel, MCPServer
from ag2.mcp.testing import connect, connect_modern
from ag2.testing import TestConfig

ASKED: list[CreateMessageRequestParams] = []


@pytest.fixture(autouse=True)
def _reset() -> None:
    ASKED.clear()


async def lends_its_model(context: ClientRequestContext, params: CreateMessageRequestParams) -> CreateMessageResult:
    """A client that answers completions — and records what it was asked for."""
    ASKED.append(params)
    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text="the caller's model says hello"),
        model="caller-model-v1",
        stopReason="end_turn",
    )


def borrowing(*, fallback: bool = False, config: TestConfig | None = None) -> MCPServer:
    """A server whose agent has no model of its own unless one is passed."""
    return MCPServer(Agent("borrower", config=config), client_model=ClientModel(fallback=fallback))


async def _call(session: Any, **kwargs: Any) -> Any:
    return await session.call_tool("ask", {"message": "think about it"}, allow_input_required=True, **kwargs)


@pytest.mark.asyncio
class TestTheHandshakeEra:
    async def test_the_agents_turn_runs_on_the_callers_model(self) -> None:
        async with connect(borrowing(), sampling_callback=lends_its_model) as session:
            result = await session.call_tool("ask", {"message": "think about it"})

        assert result.is_error is False
        first = result.content[0]
        assert isinstance(first, TextContent)
        assert first.text == "the caller's model says hello"
        [asked] = ASKED
        assert [block.text for block in _texts(asked)] == ["think about it"]


@pytest.mark.asyncio
class TestTheModernEra:
    async def test_the_completion_request_comes_back_as_the_calls_result(self) -> None:
        """No back-channel on this revision, so the request rides the same pause."""
        async with connect_modern(borrowing(), sampling_callback=lends_its_model) as session:
            first = await _call(session)

        assert isinstance(first, InputRequiredResult)
        ((_key, request),) = (first.input_requests or {}).items()
        assert isinstance(request, CreateMessageRequest)

    async def test_the_answered_retry_completes_the_turn(self) -> None:
        async with connect_modern(borrowing(), sampling_callback=lends_its_model) as session:
            first = await _call(session)
            assert isinstance(first, InputRequiredResult)
            ((key, _request),) = (first.input_requests or {}).items()
            final = await _call(
                session,
                input_responses={
                    key: CreateMessageResult(
                        role="assistant",
                        content=TextContent(type="text", text="the caller's model says hello"),
                        model="caller-model-v1",
                    )
                },
                request_state=first.request_state,
            )

        assert not isinstance(final, InputRequiredResult)
        assert final.is_error is False
        reply = final.content[0]
        assert isinstance(reply, TextContent)
        assert reply.text == "the caller's model says hello"


@pytest.mark.asyncio
class TestItIsADecision:
    async def test_a_server_that_did_not_enable_it_never_asks(self) -> None:
        """The default: the agent's own model answers, and the client is not touched."""
        server = MCPServer(Agent("own-model", config=TestConfig("my own answer")))

        async with connect(server, sampling_callback=lends_its_model) as session:
            result = await session.call_tool("ask", {"message": "hi"})

        assert result.is_error is False
        assert ASKED == [], "a server that never enabled sampling asked for a completion"

    async def test_a_client_that_cannot_sample_is_not_asked_and_the_turn_fails(self) -> None:
        """No capability declared, and no fallback configured: say so, do not improvise."""
        async with connect(borrowing(), raise_exceptions=False) as session:
            result = await session.call_tool("ask", {"message": "hi"})

        assert result.is_error is True
        first = result.content[0]
        assert isinstance(first, TextContent)
        assert "advertised no sampling capability" in first.text
        assert ASKED == []

    async def test_the_configured_fallback_uses_the_agents_own_model(self) -> None:
        server = borrowing(fallback=True, config=TestConfig("my own answer"))

        async with connect(server, raise_exceptions=False) as session:
            result = await session.call_tool("ask", {"message": "hi"})

        assert result.is_error is False
        first = result.content[0]
        assert isinstance(first, TextContent)
        assert first.text == "my own answer"

    async def test_a_turn_needing_tools_refuses_rather_than_losing_them(self) -> None:
        """An agent whose tools silently vanished would answer as though it had none."""

        def look_up(query: str) -> str:
            """Look something up."""
            return "found"

        server = MCPServer(Agent("borrower", tools=[look_up]), client_model=ClientModel())

        async with connect(server, raise_exceptions=False, sampling_callback=lends_its_model) as session:
            result = await session.call_tool("ask", {"message": "hi"})

        assert result.is_error is True
        first = result.content[0]
        assert isinstance(first, TextContent)
        assert "cannot borrow the calling client's model" in first.text
        assert ASKED == []


def _texts(params: CreateMessageRequestParams) -> list[TextContent]:
    blocks: list[TextContent] = []
    for message in params.messages:
        content = message.content if isinstance(message.content, list) else [message.content]
        blocks.extend(block for block in content if isinstance(block, TextContent))
    return blocks
