# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A served agent asking a *modern-era* client, which has no back-channel to ask over.

Revision 2026-07-28 defines no server-to-client request, so the question comes
back as the result of the call and the client answers by retrying. These tests
drive that from the client's side of the wire — the questions it receives, the
result it gets on retry — and never assert how the pause is stored, keyed or
sealed. The state token is opaque by design; a test reading it would be asserting
an implementation detail and would break on a key rotation that changes nothing
observable.

The handshake era's inline path is covered in ``test_elicitation.py``.
"""

from typing import Any

import pytest
from mcp.client.session import ClientRequestContext
from mcp.types import ElicitRequest, ElicitRequestParams, ElicitResult, InputRequiredResult, TextContent

from ag2 import Agent, Context
from ag2.events import ToolCallEvent
from ag2.mcp import MCPServer
from ag2.mcp.elicitation import ANSWER_FIELD
from ag2.mcp.testing import connect_modern
from ag2.testing import TestConfig


def asking_agent(
    *,
    questions: tuple[str, ...] = ("What colour?",),
    answers: list[str] | None = None,
    side_effects: list[str] | None = None,
    timeout: float | None = None,
) -> Agent:
    """An agent whose one tool asks ``questions`` in order.

    ``side_effects`` records once per tool invocation, which is how "the run
    resumed rather than restarted" is asserted: a restart would run the tool
    body — and its LLM call — a second time.
    """

    async def ask_human(ctx: Context) -> str:
        if side_effects is not None:
            side_effects.append("ran")
        collected = []
        for question in questions:
            answer = await ctx.input(question, timeout=timeout)
            collected.append(answer)
            if answers is not None:
                answers.append(answer)
        return "human said: " + ", ".join(collected)

    return Agent(
        "asker",
        config=TestConfig(ToolCallEvent(name="ask_human"), "done", raise_tool_errors=False),
        tools=[ask_human],
    )


def accepting(answer: str) -> ElicitResult:
    return ElicitResult(action="accept", content={ANSWER_FIELD: answer})


async def declares_elicitation(context: ClientRequestContext, params: ElicitRequestParams) -> ElicitResult:
    """Supplying a callback is what makes the client declare it can answer.

    These tests drive the retry loop by hand rather than through the SDK's own
    driver, so this is never actually invoked — but a server only asks a client
    that said it could answer, which is the behaviour under test elsewhere.
    """
    raise AssertionError("these tests answer by retrying, not through the callback")


def served(**agent_kwargs: Any) -> MCPServer:
    return MCPServer(asking_agent(**agent_kwargs))


async def _call(session: Any, **kwargs: Any) -> Any:
    return await session.call_tool("ask", {"message": "go"}, allow_input_required=True, **kwargs)


@pytest.mark.asyncio
class TestModernEraPause:
    async def test_the_question_comes_back_as_the_calls_result(self) -> None:
        """Not an error and not a hang: the outstanding question, plus state."""
        async with connect_modern(served(), elicitation_callback=declares_elicitation) as session:
            first = await _call(session)

        assert isinstance(first, InputRequiredResult)
        assert first.request_state is not None
        ((_key, request),) = (first.input_requests or {}).items()
        assert isinstance(request, ElicitRequest)
        assert request.params.message == "What colour?"

    async def test_an_answered_retry_completes_the_turn(self) -> None:
        answers: list[str] = []

        async with connect_modern(served(answers=answers), elicitation_callback=declares_elicitation) as session:
            first = await _call(session)
            assert isinstance(first, InputRequiredResult)
            (key,) = (first.input_requests or {}).keys()
            final = await _call(
                session,
                input_responses={key: accepting("blue")},
                request_state=first.request_state,
            )

        assert answers == ["blue"]
        assert not isinstance(final, InputRequiredResult)
        assert final.is_error is False
        reply, _trailer = final.content
        assert reply == TextContent(type="text", text="done")

    async def test_the_run_resumed_rather_than_restarted(self) -> None:
        """The whole point: the work already done — and paid for — is not thrown away."""
        side_effects: list[str] = []

        async with connect_modern(
            served(side_effects=side_effects), elicitation_callback=declares_elicitation
        ) as session:
            first = await _call(session)
            assert isinstance(first, InputRequiredResult)
            (key,) = (first.input_requests or {}).keys()
            await _call(session, input_responses={key: accepting("blue")}, request_state=first.request_state)

        assert side_effects == ["ran"], "the tool body ran again, so the run restarted rather than resumed"

    async def test_a_second_question_pauses_again(self) -> None:
        answers: list[str] = []
        agent = asking_agent(questions=("First?", "Second?"), answers=answers)

        async with connect_modern(MCPServer(agent), elicitation_callback=declares_elicitation) as session:
            first = await _call(session)
            assert isinstance(first, InputRequiredResult)
            (key1,) = (first.input_requests or {}).keys()
            second = await _call(session, input_responses={key1: accepting("one")}, request_state=first.request_state)
            assert isinstance(second, InputRequiredResult)
            (key2,) = (second.input_requests or {}).keys()
            assert (second.input_requests or {})[key2].params.message == "Second?"  # type: ignore[union-attr]
            final = await _call(session, input_responses={key2: accepting("two")}, request_state=second.request_state)

        assert answers == ["one", "two"]
        assert not isinstance(final, InputRequiredResult)
        assert final.is_error is False

    async def test_a_stale_answer_is_not_consumed_and_the_question_is_re_asked(self) -> None:
        """An answer minted for a question the run has moved past must not be applied."""
        agent = asking_agent(questions=("First?", "Second?"))

        async with connect_modern(MCPServer(agent), elicitation_callback=declares_elicitation) as session:
            first = await _call(session)
            assert isinstance(first, InputRequiredResult)
            (key1,) = (first.input_requests or {}).keys()
            second = await _call(session, input_responses={key1: accepting("one")}, request_state=first.request_state)
            assert isinstance(second, InputRequiredResult)
            # Round 1's answer replayed against round 2's state: the key names a
            # question this run is no longer waiting on.
            third = await _call(session, input_responses={key1: accepting("one")}, request_state=second.request_state)

        assert isinstance(third, InputRequiredResult), "a stale answer completed the round"
        ((_key, request),) = (third.input_requests or {}).items()
        assert request.params.message == "Second?"  # type: ignore[union-attr]

    async def test_a_client_that_cannot_answer_is_never_asked(self) -> None:
        """No elicitation capability declared, so the existing failure surfaces instead."""
        async with connect_modern(MCPServer(asking_agent()), raise_exceptions=False) as session:
            result = await _call(session)

        assert not isinstance(result, InputRequiredResult)
        assert result.is_error is True
        first = result.content[0]
        assert isinstance(first, TextContent)
        assert "Human input was requested but not provided" in first.text
