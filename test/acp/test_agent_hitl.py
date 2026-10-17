# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Answering a served agent's ``context.input()`` from the hosting application."""

from typing import Any

import acp
import pytest
from acp.exceptions import RequestError
from dirty_equals import IsPartialDict

from ag2 import Agent, Context
from ag2.acp import ACPAgent
from ag2.acp.executor import CANCELLED_TOOL_RESULT, HUMAN_INPUT_ERROR_CATEGORY
from ag2.acp.testing import connect
from ag2.events import (
    HumanInputRequest,
    HumanMessage,
    ToolCallEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from ag2.history import HUMAN_INPUT_ABANDONED_TOOL_RESULT
from ag2.hitl import HumanHook
from ag2.middleware import approval_required
from ag2.testing import TestConfig

QUESTION = "which colour?"


def _asking_agent(
    *,
    hitl_hook: HumanHook | None = None,
    variables: dict[Any, Any] | None = None,
) -> tuple[Agent, list[str]]:
    """An agent whose one tool stops to ask the human, and the answers it got."""
    agent = Agent(
        "workie",
        config=TestConfig(ToolCallEvent(name="ask_human", arguments="{}"), "done"),
        hitl_hook=hitl_hook,
        variables=variables,
    )
    answers: list[str] = []

    @agent.tool
    async def ask_human(context: Context) -> str:
        """Put a question to the human and report the answer."""
        answer = await context.input(QUESTION)
        answers.append(answer)
        return answer

    return agent, answers


@pytest.mark.asyncio
class TestWithoutAHook:
    """The default: no human is reachable, so say so instead of hanging."""

    async def test_the_turn_fails_rather_than_waiting_forever(self) -> None:
        agent, answers = _asking_agent()

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            with pytest.raises(RequestError) as caught:
                await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

        assert caught.value.data == IsPartialDict({
            "type": "HumanInputUnsupportedError",
            "category": HUMAN_INPUT_ERROR_CATEGORY,
        })
        assert answers == []

    async def test_the_failure_names_the_way_out(self) -> None:
        """The Client is another process; a bare error leaves nobody a next step."""
        agent, _ = _asking_agent()

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            with pytest.raises(RequestError) as caught:
                await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

        assert "hitl_hook" in caught.value.data["reason"]  # type: ignore[index]

    async def test_the_served_agents_own_hook_is_not_used(self) -> None:
        """An agent's own hook may read a console — which is the ACP transport.

        Serving over stdio makes stdin the protocol's, so a hook the agent
        carries for off-protocol use is exactly the thing that must not run here.
        Reaching a human over ACP is the host's decision, made per connection.
        """
        agent, answers = _asking_agent(hitl_hook=lambda event: HumanMessage("from the agent's own hook"))

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            with pytest.raises(RequestError):
                await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

        assert answers == []


@pytest.mark.asyncio
class TestWithAHook:
    async def test_the_hooks_answer_completes_the_turn(self) -> None:
        agent, answers = _asking_agent()

        def answer(event: HumanInputRequest) -> str:
            return "blue"

        async with connect(ACPAgent(agent, hitl_hook=answer)) as (conn, recorder):
            session = await conn.new_session(cwd="/tmp")
            response = await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

        assert response.stop_reason == "end_turn"
        assert answers == ["blue"]

    async def test_the_hook_is_given_the_question(self) -> None:
        agent, _ = _asking_agent()
        asked: list[str] = []

        def answer(event: HumanInputRequest) -> str:
            asked.append(event.content)
            return "blue"

        async with connect(ACPAgent(agent, hitl_hook=answer)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

        assert asked == [QUESTION]

    async def test_an_async_hook_is_awaited(self) -> None:
        """A host that reaches its human over a network cannot answer synchronously."""
        agent, answers = _asking_agent()

        async def answer(event: HumanInputRequest) -> HumanMessage:
            return HumanMessage("green")

        async with connect(ACPAgent(agent, hitl_hook=answer)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

        assert answers == ["green"]

    async def test_the_hook_resolves_context_like_any_other(self) -> None:
        """It is an ordinary AG2 hook, so the session's variables reach it."""
        agent, answers = _asking_agent(variables={"caller": "ada"})

        async def answer(event: HumanInputRequest, context: Context) -> str:
            return str(context.variables["caller"])

        async with connect(ACPAgent(agent, hitl_hook=answer)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

        assert answers == ["ada"]

    async def test_each_session_asks_again(self) -> None:
        """The hook is per-connection state, not a once-per-agent answer."""
        agent, answers = _asking_agent()
        replies = iter(["first", "second"])

        def answer(event: HumanInputRequest) -> str:
            return next(replies)

        async with connect(ACPAgent(agent, hitl_hook=answer)) as (conn, _):
            one = await conn.new_session(cwd="/tmp")
            two = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=one.session_id, prompt=[acp.text_block("go")])
            await conn.prompt(session_id=two.session_id, prompt=[acp.text_block("go")])

        assert answers == ["first", "second"]


def _results(history: list[Any]) -> list[ToolResultEvent]:
    """Every tool result in ``history``, batched inside ToolResultsEvent or not."""
    results: list[ToolResultEvent] = []
    for event in history:
        if isinstance(event, ToolResultsEvent):
            results.extend(event.results)
        elif isinstance(event, ToolResultEvent):
            results.append(event)
    return results


def _lenient_asking_agent(*, via_middleware: bool = False) -> tuple[Agent, list[str]]:
    """Same agent, but behind a model that does not re-raise tool errors.

    ``TestConfig`` re-raises any ``ToolErrorEvent`` in the history by default,
    so a turn fails under it whenever anything on the tool path raises — which
    hides whether *this* failure ends the turn or is quietly recorded as a tool
    result and answered around.
    """
    agent = Agent(
        "workie",
        config=TestConfig(
            ToolCallEvent(name="ask_human", arguments="{}"),
            "done",
            raise_tool_errors=False,
        ),
    )
    answers: list[str] = []

    if via_middleware:

        @agent.tool(middleware=[approval_required()])
        async def ask_human() -> str:
            """Approved from around the tool rather than inside it."""
            answers.append("ran")
            return "done"

    else:

        @agent.tool
        async def ask_human(context: Context) -> str:
            """Put a question to the human and report the answer."""
            answer = await context.input(QUESTION)
            answers.append(answer)
            return answer

    return agent, answers


@pytest.mark.asyncio
class TestAnUnanswerableQuestionReachesTheClient:
    """The Client is the only one who can act on it, so it has to be told.

    Under a provider that treats a tool error as an ordinary result — every real
    one — a swallowed ``HumanInputUnsupportedError`` becomes the tool's output:
    the model reads a traceback where an answer belongs, answers around it, and
    the Client is handed ``end_turn`` for a turn that never got its answer.
    """

    @pytest.mark.parametrize("via_middleware", [False, True], ids=["tool-body", "approval-middleware"])
    async def test_no_hook_fails_the_turn(self, via_middleware: bool) -> None:
        agent, answers = _lenient_asking_agent(via_middleware=via_middleware)

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            with pytest.raises(RequestError) as caught:
                await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

        assert caught.value.data == IsPartialDict({"type": "HumanInputUnsupportedError"})
        assert answers == []

    async def test_a_failing_hook_fails_the_turn(self) -> None:
        """The hosting application's own channel breaking is still nobody being asked.

        ``category`` is what the Client branches on: the exception type varies
        with what broke — the missing hook here, the queue there — and a Client
        deciding whether to put the question to its own human should not have to
        track a list of AG2 class names to find out.
        """
        agent, answers = _lenient_asking_agent(via_middleware=True)

        async def unreachable(event: HumanInputRequest) -> HumanMessage:
            raise RuntimeError("approval queue unreachable")

        async with connect(ACPAgent(agent, hitl_hook=unreachable)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            with pytest.raises(RequestError) as caught:
                await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

        assert caught.value.data == IsPartialDict({
            "type": "HumanInputFailedError",
            "category": HUMAN_INPUT_ERROR_CATEGORY,
        })
        assert "approval queue unreachable" in caught.value.data["reason"]
        # The approval was never given, so the tool must not have run.
        assert answers == []

    async def test_an_ordinary_turn_failure_carries_no_human_input_category(self) -> None:
        """The discriminator has to discriminate, or a Client cannot use it."""
        agent = Agent(
            "workie",
            config=TestConfig(ToolCallEvent(name="explode", arguments="{}"), "done"),
        )

        @agent.tool
        async def explode() -> str:
            """Fail the ordinary way."""
            raise RuntimeError("the tool itself broke")

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            with pytest.raises(RequestError) as caught:
                await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

        assert "category" not in caught.value.data

    async def test_the_session_survives_the_failed_turn(self) -> None:
        """A turn dies between a tool call and its result — repair it like a cancel.

        Left alone, the transcript keeps an unanswered tool call and the next
        prompt on this session sends a provider something it will reject.
        """
        agent, _ = _lenient_asking_agent()
        acp_agent = ACPAgent(agent)

        async with connect(acp_agent) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            with pytest.raises(RequestError):
                await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

            stored = await acp_agent.sessions.get(session.session_id)
            history = list(await acp_agent.sessions.stream(stored).history.get_events())

        calls = [event for event in history if isinstance(event, ToolCallEvent)]
        results = _results(history)
        answered = {result.parent_id for result in results}
        assert calls, "the turn should have reached a tool call"
        assert all(call.id in answered for call in calls)

        # Repaired at the turn boundary, where the reason is still known, so the
        # transcript says the question went unanswered rather than claiming a
        # cancellation the Client never asked for.
        stand_ins = [result.result.parts[0].content for result in results]
        assert stand_ins == [HUMAN_INPUT_ABANDONED_TOOL_RESULT]
        assert CANCELLED_TOOL_RESULT not in stand_ins
