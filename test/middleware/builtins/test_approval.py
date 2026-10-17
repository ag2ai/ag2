# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from dirty_equals import IsPartialDict

from ag2 import Agent, tool
from ag2.events import ToolApprovalRequest, ToolCallEvent, ToolResultEvent
from ag2.middleware import approval_required
from ag2.middleware.builtin.tools.approval import BYPASS_KEY
from ag2.testing import TestConfig
from test._helpers import ScriptedHuman


def make_context(
    response: str = "y",
    variables: dict[str, Any] | None = None,
) -> AsyncMock:
    context = AsyncMock()
    context.ask = AsyncMock(return_value=response)
    context.variables = variables if variables is not None else {}
    return context


def asked(context: AsyncMock) -> ToolApprovalRequest:
    """The request the middleware put to the human."""
    [request], _ = context.ask.await_args
    assert isinstance(request, ToolApprovalRequest)
    return request


@pytest.fixture
def tool_call() -> ToolCallEvent:
    return ToolCallEvent(name="calculator", arguments='{"a": 1, "b": 2}')


@pytest.mark.asyncio()
@pytest.mark.parametrize("response", ["y", "Y", "yes", "Yes", "YES", "1"])
async def test_accepts_various_affirmative_inputs(tool_call: ToolCallEvent, response: str) -> None:
    hook = approval_required()
    context = make_context(response)

    expected = ToolResultEvent.from_call(tool_call, result="3")

    async def call_next(event: ToolCallEvent, ctx: object) -> ToolResultEvent:
        return expected

    result = await hook(call_next, tool_call, context)

    assert result == expected
    context.ask.assert_awaited_once()


@pytest.mark.asyncio()
async def test_denies_on_no(tool_call: ToolCallEvent) -> None:
    hook = approval_required()
    context = make_context("n")

    call_next = AsyncMock()

    result = await hook(call_next, tool_call, context)

    call_next.assert_not_awaited()
    assert result == ToolResultEvent.from_call(tool_call, result="User denied the tool call request")


@pytest.mark.asyncio()
async def test_custom_message(tool_call: ToolCallEvent) -> None:
    custom_msg = "Approve {tool_name} with {tool_arguments}?"
    hook = approval_required(message=custom_msg)
    context = make_context("y")

    call_next = AsyncMock(return_value=ToolResultEvent.from_call(tool_call, result="ok"))

    await hook(call_next, tool_call, context)

    assert asked(context) == ToolApprovalRequest(
        'Approve calculator with {"a": 1, "b": 2}?',
        tool_call_id=tool_call.id,
        tool_name="calculator",
    )


@pytest.mark.asyncio()
async def test_custom_timeout(tool_call: ToolCallEvent) -> None:
    hook = approval_required(timeout=60)
    context = make_context("y")

    call_next = AsyncMock(return_value=ToolResultEvent.from_call(tool_call, result="ok"))

    await hook(call_next, tool_call, context)

    assert asked(context).timeout == 60


@pytest.mark.asyncio()
async def test_custom_denied_message(tool_call: ToolCallEvent) -> None:
    hook = approval_required(denied_message="Rejected by user")
    context = make_context("no")

    call_next = AsyncMock(return_value=ToolResultEvent.from_call(tool_call, result="ok"))

    result = await hook(call_next, tool_call, context)

    assert result == ToolResultEvent.from_call(tool_call, result="Rejected by user")


@pytest.mark.asyncio()
async def test_always_sets_bypass_flag(tool_call: ToolCallEvent) -> None:
    hook = approval_required(allow_always=True)
    context = make_context("always")

    expected = ToolResultEvent.from_call(tool_call, result="ok")
    call_next = AsyncMock(return_value=expected)

    # first execution should prompt
    await hook(call_next, tool_call, context)
    context.ask.assert_awaited_once()

    # second execution should not prompt
    await hook(call_next, tool_call, context)
    context.ask.assert_awaited_once()


@pytest.mark.asyncio()
async def test_always_is_per_tool(tool_call: ToolCallEvent) -> None:
    hook = approval_required(allow_always=True)
    context = make_context("y", variables={BYPASS_KEY: {"other_tool": True}})

    expected = ToolResultEvent.from_call(tool_call, result="ok")
    call_next = AsyncMock(return_value=expected)

    result = await hook(call_next, tool_call, context)

    assert result == expected
    # Should still prompt since "calculator" is not in the bypass dict
    context.ask.assert_awaited_once()


@pytest.mark.asyncio()
async def test_always_ignored_when_disabled(tool_call: ToolCallEvent) -> None:
    hook = approval_required(allow_always=False)
    context = make_context("always")

    call_next = AsyncMock()

    result = await hook(call_next, tool_call, context)

    call_next.assert_not_awaited()
    assert result == ToolResultEvent.from_call(tool_call, result="User denied the tool call request")


@pytest.mark.asyncio()
async def test_always_replaces_the_bypass_dict(tool_call: ToolCallEvent) -> None:
    hook = approval_required(allow_always=True)
    shared = {"other_tool": True}
    context = make_context("always", variables={BYPASS_KEY: shared})

    await hook(AsyncMock(), tool_call, context)

    assert context.variables[BYPASS_KEY] is not shared
    assert context.variables[BYPASS_KEY] == IsPartialDict({"other_tool": True})
    assert shared == {"other_tool": True}


@pytest.mark.asyncio()
async def test_always_does_not_approve_another_tool_of_the_same_name() -> None:
    runs: list[str] = []

    @tool(name="deploy", middleware=[approval_required()])
    def deploy_local() -> str:
        runs.append("local")
        return "deployed"

    @tool(name="deploy", middleware=[approval_required()])
    def deploy_other() -> str:
        runs.append("other")
        return "deployed"

    agent = Agent(
        "",
        tools=[deploy_other],
        config=TestConfig(
            ToolCallEvent(name="deploy", arguments="{}"),
            "done",
            ToolCallEvent(name="deploy", arguments="{}"),
            "done",
        ),
    )
    human = ScriptedHuman("always")

    reply = await agent.ask("Deploy", tools=[deploy_local], hitl_hook=human)
    human.answer = "n"
    await reply.ask("Deploy again", hitl_hook=human)

    assert runs == ["local"]
    assert human.questions == 2


@pytest.mark.asyncio()
async def test_always_through_a_shared_hook_does_not_approve_another_tool_of_the_same_name() -> None:
    runs: list[str] = []
    gate = approval_required()

    @tool(name="deploy", middleware=[gate])
    def deploy_local() -> str:
        runs.append("local")
        return "deployed"

    @tool(name="deploy", middleware=[gate])
    def deploy_other() -> str:
        runs.append("other")
        return "deployed"

    agent = Agent(
        "",
        tools=[deploy_other],
        config=TestConfig(
            ToolCallEvent(name="deploy", arguments="{}"),
            "done",
            ToolCallEvent(name="deploy", arguments="{}"),
            "done",
        ),
    )
    human = ScriptedHuman("always")

    reply = await agent.ask("Deploy", tools=[deploy_local], hitl_hook=human)
    human.answer = "n"
    await reply.ask("Deploy again", hitl_hook=human)

    assert runs == ["local"]
    assert human.questions == 2


@pytest.mark.asyncio()
async def test_always_keeps_approving_the_same_tool_on_later_turns() -> None:
    runs: list[str] = []

    @tool(middleware=[approval_required()])
    def deploy() -> str:
        runs.append("deploy")
        return "deployed"

    agent = Agent(
        "",
        tools=[deploy],
        config=TestConfig(
            ToolCallEvent(name="deploy", arguments="{}"),
            "done",
            ToolCallEvent(name="deploy", arguments="{}"),
            "done",
        ),
    )
    human = ScriptedHuman("always")

    reply = await agent.ask("Deploy", hitl_hook=human)
    human.answer = "n"
    await reply.ask("Deploy again", hitl_hook=human)

    assert runs == ["deploy", "deploy"]
    assert human.questions == 1


def deploy() -> str:
    return "deployed"


@pytest.mark.asyncio()
async def test_always_survives_rebuilding_the_tool_from_stored_variables() -> None:
    """A restarted process builds the tool and its hook anew; the grant still applies."""
    human = ScriptedHuman("always")
    first = Agent(
        "",
        tools=[tool(deploy, middleware=[approval_required()])],
        config=TestConfig(ToolCallEvent(name="deploy", arguments="{}"), "done"),
    )
    reply = await first.ask("Deploy", hitl_hook=human)
    stored = json.loads(json.dumps(reply.context.variables))

    human.answer = "n"
    rebuilt = Agent(
        "",
        tools=[tool(deploy, middleware=[approval_required()])],
        config=TestConfig(ToolCallEvent(name="deploy", arguments="{}"), "done"),
    )
    await rebuilt.ask("Deploy again", variables=stored, hitl_hook=human)

    assert human.questions == 1
