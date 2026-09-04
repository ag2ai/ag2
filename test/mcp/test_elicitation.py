# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A served agent asking the human behind a *handshake-era* client.

Those revisions still carry a bidirectional channel, so the question goes out as
a standalone ``elicitation/create`` request and the answer comes back inline —
nothing pauses, nothing is stored. The modern era's pause-and-resume path is
covered in ``test_pause_and_resume.py``; behaviour the two eras share is
asserted against both from the same body there.

The scripted config carries ``raise_tool_errors=False`` throughout, which models
a real provider: it is handed a failed tool call as an ordinary result and
carries on. Asserting "this ends the turn" against a re-raising double would
assert the double.
"""

from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from mcp.client.session import ClientRequestContext
from mcp.types import ElicitRequestParams, ElicitResult, ErrorData, TextContent

from ag2 import Agent, Context
from ag2.events import HumanInputRequest, HumanMessage, ToolCallEvent
from ag2.mcp import MCPServer
from ag2.mcp.elicitation import ANSWER_FIELD
from ag2.mcp.testing import connect
from ag2.testing import TestConfig

ElicitationCallback = Callable[[ClientRequestContext, ElicitRequestParams], Awaitable[ElicitResult | ErrorData]]


def asking_agent(
    *,
    hitl_hook: Any = None,
    question: str = "What colour?",
    answers: list[str] | None = None,
) -> Agent:
    """An agent whose one tool asks the human a question.

    ``answers`` collects what ``context.input()`` actually returned, which is the
    assertion that matters: the turn completing proves only that *something*
    answered.
    """

    async def ask_human(ctx: Context) -> str:
        answer = await ctx.input(question)
        if answers is not None:
            answers.append(answer)
        return f"human said: {answer}"

    return Agent(
        "asker",
        config=TestConfig(ToolCallEvent(name="ask_human"), "done", raise_tool_errors=False),
        tools=[ask_human],
        hitl_hook=hitl_hook,
    )


def answering(answer: str, *, seen: list[str] | None = None) -> ElicitationCallback:
    """A client-side elicitation callback that accepts with ``answer``."""

    async def callback(context: ClientRequestContext, params: ElicitRequestParams) -> ElicitResult:
        if seen is not None:
            seen.append(params.message)
        return ElicitResult(action="accept", content={ANSWER_FIELD: answer})

    return callback


def refusing(action: str = "decline", *, seen: list[str] | None = None) -> ElicitationCallback:
    async def callback(context: ClientRequestContext, params: ElicitRequestParams) -> ElicitResult:
        if seen is not None:
            seen.append(params.message)
        return ElicitResult(action=action)  # type: ignore[arg-type]

    return callback


async def _tool_result_text(server: MCPServer, **session_kwargs: Any) -> tuple[bool, str]:
    """Call ``ask`` once and return ``(is_error, the first text block)``."""
    async with connect(server, raise_exceptions=False, **session_kwargs) as session:
        result = await session.call_tool("ask", {"message": "go"})
    first = result.content[0]
    assert isinstance(first, TextContent)
    return bool(result.is_error), first.text


@pytest.mark.asyncio
class TestHandshakeEraElicitation:
    async def test_a_question_reaches_the_client_and_the_answer_continues_the_turn(self) -> None:
        seen: list[str] = []
        server = MCPServer(asking_agent())

        async with connect(server, elicitation_callback=answering("blue", seen=seen)) as session:
            result = await session.call_tool("ask", {"message": "go"})

        assert seen == ["What colour?"], "the client was not asked the tool's question"
        assert result.is_error is False
        reply, _trailer = result.content
        assert reply == TextContent(type="text", text="done")

    async def test_the_answer_is_what_the_tool_receives(self) -> None:
        """``context.input()`` returns the client's answer, not merely a completion."""
        answers: list[str] = []
        server = MCPServer(asking_agent(answers=answers))

        async with connect(server, elicitation_callback=answering("teal")) as session:
            await session.call_tool("ask", {"message": "go"})

        assert answers == ["teal"]

    async def test_a_client_that_cannot_answer_is_never_asked(self) -> None:
        """No elicitation callback means no advertised capability, so nothing is sent."""
        is_error, text = await _tool_result_text(MCPServer(asking_agent()))

        assert is_error is True
        assert "Human input was requested but not provided" in text

    async def test_with_no_hook_the_failure_names_the_hook(self) -> None:
        is_error, text = await _tool_result_text(MCPServer(asking_agent()))

        assert is_error is True
        # The existing instructional message, not a silent decline.
        assert "Agent(..., hitl_hook=func)" in text

    async def test_the_agents_own_hook_answers_when_the_client_cannot(self) -> None:
        def hitl_hook(event: HumanInputRequest) -> HumanMessage:
            return HumanMessage("from the server-side human")

        answers: list[str] = []

        async with connect(MCPServer(asking_agent(hitl_hook=hitl_hook, answers=answers))) as session:
            result = await session.call_tool("ask", {"message": "go"})

        assert result.is_error is False
        assert answers == ["from the server-side human"]

    async def test_the_client_wins_over_the_agents_own_hook(self) -> None:
        """The whole point: HITL over MCP without wiring a second, server-side human."""
        asked: list[str] = []

        def hitl_hook(event: HumanInputRequest) -> HumanMessage:
            asked.append(event.content)
            return HumanMessage("from the server-side human")

        answers: list[str] = []
        server = MCPServer(asking_agent(hitl_hook=hitl_hook, answers=answers))

        async with connect(server, elicitation_callback=answering("from the client")) as session:
            await session.call_tool("ask", {"message": "go"})

        assert asked == [], "the server-side hook was consulted even though the client could answer"
        assert answers == ["from the client"]

    async def test_the_decline_policy_never_asks_a_client_that_could_answer(self) -> None:
        seen: list[str] = []
        server = MCPServer(asking_agent(), elicitation_policy="decline")

        is_error, text = await _tool_result_text(server, elicitation_callback=answering("blue", seen=seen))

        assert seen == [], "a question was sent under the 'decline' policy"
        assert is_error is True
        assert "Human input was requested but not provided" in text

    async def test_the_decline_policy_still_lets_a_server_side_hook_answer(self) -> None:
        def hitl_hook(event: HumanInputRequest) -> HumanMessage:
            return HumanMessage("server-side")

        server = MCPServer(asking_agent(hitl_hook=hitl_hook), elicitation_policy="decline")

        async with connect(server, elicitation_callback=answering("client-side")) as session:
            result = await session.call_tool("ask", {"message": "go"})

        assert result.is_error is False

    @pytest.mark.parametrize("action", ["decline", "cancel"])
    async def test_a_refusal_ends_the_turn_deliberately(self, action: str) -> None:
        """A refusal is not a transport failure: the turn ends, and says why."""
        server = MCPServer(asking_agent())

        is_error, text = await _tool_result_text(server, elicitation_callback=refusing(action))

        assert is_error is True
        assert action in text

    async def test_an_accept_with_no_answer_field_is_not_an_answer(self) -> None:
        async def callback(context: ClientRequestContext, params: ElicitRequestParams) -> ElicitResult:
            return ElicitResult(action="accept", content={})

        is_error, text = await _tool_result_text(MCPServer(asking_agent()), elicitation_callback=callback)

        assert is_error is True
        assert "no answer to continue from" in text

    async def test_the_form_asks_for_one_string_field(self) -> None:
        """The question renders identically on both transports, so pin its shape once."""
        forms: list[ElicitRequestParams] = []

        async def callback(context: ClientRequestContext, params: ElicitRequestParams) -> ElicitResult:
            forms.append(params)
            return ElicitResult(action="accept", content={ANSWER_FIELD: "x"})

        async with connect(MCPServer(asking_agent()), elicitation_callback=callback) as session:
            await session.call_tool("ask", {"message": "go"})

        (form,) = forms
        assert form.mode == "form"
        assert form.requested_schema["required"] == [ANSWER_FIELD]  # type: ignore[union-attr]
        assert form.requested_schema["properties"][ANSWER_FIELD]["type"] == "string"  # type: ignore[union-attr]
