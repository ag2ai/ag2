# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A run's shared state: what the client sends in `state`, the variables the server holds, and what comes back.

A `STATE_SNAPSHOT` replaces the client's state wholesale, so every one the server sends is the
complete state: the client's keys and the server's variables together. The browser is a peer like
any other, so it does not author the framework's control-plane keys and is not shown them.
"""

from typing import Annotated, Any

import pytest
from ag_ui.core import CustomEvent as AGUICustomEvent
from ag_ui.core import RunFinishedEvent, RunFinishedSuccessOutcome, StateSnapshotEvent, UserMessage

from ag2 import Agent, Context, Variable
from ag2.ag_ui import AGUIEvent, AGUIStream
from ag2.events import HumanInputRequest, ToolCallEvent
from ag2.middleware import approval_required
from ag2.middleware.builtin.tools.approval import BYPASS_KEY
from ag2.testing import TestConfig
from test.ag_ui.harness import dispatch_events, each, run_input, sole

pytestmark = pytest.mark.asyncio

PREAPPROVAL: dict[str, Any] = {BYPASS_KEY: {"delete_account": True}}


def _snapshots(events: list[Any]) -> list[Any]:
    return [event.snapshot for event in each(events, StateSnapshotEvent)]


class _GatedDeletions:
    """A gated `delete_account` tool, the prompts it raised and the ids it deleted."""

    def __init__(self, answer: str = "n") -> None:
        self.answer = answer
        self.prompts: list[str] = []
        self.deleted: list[str] = []

    def hitl_hook(self, event: HumanInputRequest) -> str:
        self.prompts.append(event.content)
        return self.answer

    def bind(self, agent: Agent) -> Agent:
        @agent.tool(middleware=[approval_required()])
        def delete_account(user_id: str) -> str:
            """Delete a user account."""
            self.deleted.append(user_id)
            return f"deleted {user_id}"

        return agent


class TestVariablesAndSnapshots:
    async def test_the_agent_s_variables_are_announced_once(self) -> None:
        seen: list[str] = []
        agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="my_tool"), "Done"), variables={"var": "123"})

        @agent.tool
        def my_tool(var: Annotated[str, Variable()]) -> str:
            seen.append(var)
            return "result"

        events = await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="m1", content="Hello!")))

        assert _snapshots(events) == [{"var": "123"}]
        assert seen == ["123"]

    async def test_variables_given_for_the_turn_are_announced_once(self) -> None:
        seen: list[str] = []
        agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="my_tool"), "Done"))

        @agent.tool
        def my_tool(var: Annotated[str, Variable()]) -> str:
            seen.append(var)
            return "result"

        events = await dispatch_events(
            AGUIStream(agent), run_input(UserMessage(id="m1", content="Hello!")), variables={"var": "123"}
        )

        assert _snapshots(events) == [{"var": "123"}]
        assert seen == ["123"]

    async def test_state_the_client_sent_reaches_the_tools_and_is_not_announced_back(self) -> None:
        seen: list[str] = []
        agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="my_tool"), "Done"))

        @agent.tool
        def my_tool(var: Annotated[str, Variable()]) -> str:
            seen.append(var)
            return "result"

        incoming = run_input(UserMessage(id="m1", content="Hello!"), state={"var": "123"})

        events = await dispatch_events(AGUIStream(agent), incoming)

        assert _snapshots(events) == []
        assert seen == ["123"]

    async def test_a_run_with_no_variables_sends_no_snapshot(self) -> None:
        agent = Agent("test_agent", config=TestConfig("Done"))

        events = await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="m1", content="Hello!")))

        assert _snapshots(events) == []

    async def test_a_tool_that_sets_variables_sends_the_whole_state_again(self) -> None:
        agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="my_tool"), "Done"), variables={"var": "123"})

        @agent.tool
        def my_tool(var: Annotated[str, Variable()], ctx: Context) -> str:
            ctx.variables["var2"] = "1"
            ctx.variables["var3"] = "1234"
            return "result"

        events = await dispatch_events(
            AGUIStream(agent), run_input(UserMessage(id="m1", content="Hello!")), variables={"var2": "1234"}
        )

        assert _snapshots(events) == [
            {"var": "123", "var2": "1234"},
            {"var": "123", "var2": "1", "var3": "1234"},
        ]


class TestTheClientsState:
    async def test_every_snapshot_holds_the_client_s_keys_and_the_server_s_variables(self) -> None:
        agent = Agent("test_agent", config=TestConfig("done"))
        incoming = run_input(UserMessage(id="m1", content="hi"), state={"draft": "hello"})

        events = await dispatch_events(AGUIStream(agent), incoming, variables={"user_id": "u1"})

        assert _snapshots(events) == [{"draft": "hello", "user_id": "u1"}]

    async def test_a_run_that_changes_a_variable_sends_the_whole_state_back(self) -> None:
        agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="count"), "done"))

        @agent.tool
        def count(context: Context) -> str:
            """Count a visit."""
            context.variables["visits"] = 1
            return "counted"

        incoming = run_input(UserMessage(id="m1", content="hi"), state={"draft": "hello"})

        events = await dispatch_events(AGUIStream(agent), incoming)

        assert _snapshots(events) == [{"draft": "hello", "visits": 1}]

    async def test_a_run_that_changes_nothing_sends_the_client_nothing_to_replace(self) -> None:
        agent = Agent("test_agent", config=TestConfig("done"))
        incoming = run_input(UserMessage(id="m1", content="hi"), state={"draft": "hello"})

        events = await dispatch_events(AGUIStream(agent), incoming)

        assert _snapshots(events) == []

    async def test_a_null_value_in_state_is_kept(self) -> None:
        agent = Agent("test_agent", config=TestConfig("done"))
        incoming = run_input(UserMessage(id="m1", content="hi"), state={"keep": None})

        events = await dispatch_events(AGUIStream(agent), incoming, variables={"user_id": "u1"})

        assert _snapshots(events) == [{"keep": None, "user_id": "u1"}]

    @pytest.mark.parametrize("state", [[1, 2], "draft", 3])
    async def test_a_state_that_is_not_an_object_is_served_and_left_as_it_is(self, state: Any) -> None:
        agent = Agent("test_agent", config=TestConfig("done"))
        incoming = run_input(UserMessage(id="m1", content="hi"), state=state)

        events = await dispatch_events(AGUIStream(agent), incoming, variables={"user_id": "u1"})

        assert sole(events, RunFinishedEvent).outcome == RunFinishedSuccessOutcome()
        assert _snapshots(events) == []


class TestControlPlaneKeys:
    async def test_client_state_cannot_preapprove_a_gated_tool(self) -> None:
        gated = _GatedDeletions(answer="n")
        call = ToolCallEvent(name="delete_account", arguments='{"user_id": "victim-7"}')
        agent = gated.bind(Agent("test_agent", config=TestConfig(call, "Done"), hitl_hook=gated.hitl_hook))
        incoming = run_input(UserMessage(id="m1", content="clean up"), state=PREAPPROVAL)

        await dispatch_events(AGUIStream(agent), incoming)

        assert gated.prompts != []
        assert gated.deleted == []

    async def test_ordinary_client_state_still_reaches_the_turn(self) -> None:
        seen: list[str] = []
        agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="peek"), "Done"))

        @agent.tool
        def peek(tenant_note: Annotated[str, Variable()]) -> str:
            """Report the tenant note."""
            seen.append(tenant_note)
            return "ok"

        incoming = run_input(UserMessage(id="m1", content="hi"), state={"tenant_note": "acme", **PREAPPROVAL})

        events = await dispatch_events(AGUIStream(agent), incoming)

        assert seen == ["acme"]
        assert all(BYPASS_KEY not in snapshot for snapshot in _snapshots(events))

    async def test_a_snapshot_hides_reserved_variables(self) -> None:
        agent = Agent("test_agent", config=TestConfig("Done"), variables={"tenant_note": "acme", **PREAPPROVAL})

        events = await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="m1", content="hi")))

        assert _snapshots(events) == [{"tenant_note": "acme"}]


async def test_a_tool_can_send_a_custom_event_to_the_client() -> None:
    agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="my_tool"), "Done"))

    @agent.tool
    async def my_tool(ctx: Context) -> None:
        await ctx.send(AGUIEvent(AGUICustomEvent(name="test", value=123)))

    events = await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="m1", content="Hello!")))

    [custom] = [event for event in events if isinstance(event, AGUICustomEvent)]
    assert (custom.name, custom.value) == ("test", 123)
