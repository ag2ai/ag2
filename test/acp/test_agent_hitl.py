# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Answering a served agent's ``context.input()`` from the hosting application."""

from collections.abc import Sequence
from typing import Any

import acp
import pytest
from acp.exceptions import RequestError
from dirty_equals import IsPartialDict

from ag2 import Agent, Context
from ag2.acp import ACPAgent
from ag2.acp.testing import connect
from ag2.config import ModelConfig
from ag2.events import (
    BaseEvent,
    HumanInputRequest,
    HumanMessage,
    ModelMessage,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
)
from ag2.hitl import HumanHook
from ag2.middleware import ToolMiddleware, approval_required
from ag2.testing import TestConfig

QUESTION = "which colour?"


class LenientClient:
    def __init__(self, tool_call: ToolCallEvent) -> None:
        self.tool_call = tool_call

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        if any(type(message).__name__.startswith(("ToolResult", "ToolError")) for message in messages):
            return ModelResponse(ModelMessage("wrapping up"))
        return ModelResponse(tool_calls=ToolCallsEvent([self.tool_call]))


class LenientConfig(ModelConfig):
    def __init__(self, tool_name: str = "ask_human") -> None:
        self.client = LenientClient(ToolCallEvent(name=tool_name, arguments="{}"))

    @property
    def provider(self) -> str:
        return "test"

    @property
    def model(self) -> str:
        return "test"

    def copy(self) -> "LenientConfig":
        return self

    def create(self) -> LenientClient:
        return self.client


def _asking_agent(
    *,
    hitl_hook: HumanHook | None = None,
    config: ModelConfig | None = None,
    middleware: ToolMiddleware | None = None,
    variables: dict[Any, Any] | None = None,
) -> tuple[Agent, list[str]]:
    """An agent whose one tool stops to ask the human, and the answers it got."""
    agent = Agent(
        "workie",
        config=config or TestConfig(ToolCallEvent(name="ask_human", arguments="{}"), "done"),
        hitl_hook=hitl_hook,
        variables=variables,
    )
    answers: list[str] = []

    @agent.tool(middleware=[middleware] if middleware is not None else [])
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

        assert caught.value.data == IsPartialDict({"type": "HumanInputUnsupportedError"})
        assert answers == []

    async def test_the_failure_names_the_way_out(self) -> None:
        """The Client is another process; a bare error leaves nobody a next step."""
        agent, _ = _asking_agent()

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            with pytest.raises(RequestError) as caught:
                await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

        assert "hitl_hook" in caught.value.data["reason"]  # type: ignore[index]

    async def test_the_failure_is_not_returned_to_the_model_as_a_tool_error(self) -> None:
        agent, _ = _asking_agent(config=LenientConfig())

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            with pytest.raises(RequestError) as caught:
                await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

        assert caught.value.data == IsPartialDict({"type": "HumanInputUnsupportedError"})

    async def test_middleware_human_input_failure_reaches_the_client(self) -> None:
        agent, _ = _asking_agent(
            config=LenientConfig(),
            middleware=approval_required(allow_always=False),
        )

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            with pytest.raises(RequestError) as caught:
                await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

        assert caught.value.data == IsPartialDict({"type": "HumanInputUnsupportedError"})

    async def test_regular_tool_errors_remain_tool_results(self) -> None:
        agent = Agent("workie", config=LenientConfig(tool_name="fails"))

        @agent.tool
        async def fails() -> str:
            raise ValueError("expected tool failure")

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            response = await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

        assert response.stop_reason == "end_turn"

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
