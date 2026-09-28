# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Each delegation reaches the client as one subagent invocation, started and then ended."""

import asyncio
import logging

import pytest
from ag_ui.core import UserMessage
from dirty_equals import IsInt, IsPartialDict, IsStr

from ag2 import Agent, Context
from ag2.ag_ui import AGUIStream
from ag2.events import ToolCallEvent
from ag2.testing import TestConfig
from ag2.tools.subagents.run_task import run_task
from test.ag_ui.harness import dispatch_run, every, outcome_of, run_input, types_of

pytestmark = pytest.mark.asyncio


def _delegating(worker: Agent, *objectives: str) -> Agent:
    """A parent whose model delegates each of `objectives` to `worker`, all in one response."""
    calls = [ToolCallEvent(name="task_worker", arguments=f'{{"objective": "{o}"}}') for o in objectives]
    return Agent(
        "parent",
        config=TestConfig(calls, "summarised"),
        tools=[worker.as_tool(description="Delegate to the worker.")],
    )


async def _run(agent: Agent) -> list[dict[str, object]]:
    return await dispatch_run(AGUIStream(agent), run_input(UserMessage(id="m1", content="go")))


async def test_a_delegation_starts_with_its_agent_and_objective_and_finishes_with_its_result() -> None:
    events = await _run(_delegating(Agent("worker", config=TestConfig("researched")), "look into it"))

    [started] = every(events, "SUBAGENT_STARTED")
    assert started == {
        "type": "SUBAGENT_STARTED",
        "subagentRunId": IsStr(),
        "name": "worker",
        "description": "look into it",
        "timestamp": IsInt(),
    }
    [finished] = every(events, "SUBAGENT_FINISHED")
    assert finished == {
        "type": "SUBAGENT_FINISHED",
        "subagentRunId": started["subagentRunId"],
        "result": "researched",
        "timestamp": IsInt(),
    }
    assert "usage" not in started and "usage" not in finished


async def test_two_parallel_delegations_to_one_agent_are_told_apart() -> None:
    events = await _run(_delegating(Agent("worker", config=TestConfig("researched")), "first", "second"))

    started = {e["subagentRunId"] for e in every(events, "SUBAGENT_STARTED")}
    finished = {e["subagentRunId"] for e in every(events, "SUBAGENT_FINISHED")}
    assert len(started) == 2
    assert finished == started


async def test_a_failed_delegation_is_a_subagent_error_and_the_run_carries_on() -> None:
    worker = Agent("worker", config=TestConfig(RuntimeError("the worker fell over")))

    events = await _run(_delegating(worker, "look into it"))

    [started] = every(events, "SUBAGENT_STARTED")
    [error] = every(events, "SUBAGENT_ERROR")
    assert error == {
        "type": "SUBAGENT_ERROR",
        "subagentRunId": started["subagentRunId"],
        "message": "the worker fell over",
        "timestamp": IsInt(),
    }
    assert "SUBAGENT_FINISHED" not in types_of(events)
    assert outcome_of(events) == {"type": "success"}


async def test_delegations_are_no_longer_steps() -> None:
    events = await _run(_delegating(Agent("worker", config=TestConfig("researched")), "look into it"))

    assert not [t for t in types_of(events) if t.startswith("STEP_")]


async def test_a_task_id_already_announced_in_the_run_is_not_announced_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A caller-supplied id can repeat; the wire must still never see one invocation twice."""
    worker = Agent("worker", config=TestConfig("researched"))
    parent = Agent("parent", config=TestConfig(ToolCallEvent(name="delegate_twice"), "summarised"))

    @parent.tool
    async def delegate_twice(context: Context) -> str:
        """Delegate twice under one id."""
        await run_task(worker, "first", parent_context=context, task_id="same")
        await run_task(worker, "second", parent_context=context, task_id="same")
        return "delegated"

    with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
        events = await _run(parent)

    assert [e["description"] for e in every(events, "SUBAGENT_STARTED")] == ["first"]
    assert len(every(events, "SUBAGENT_FINISHED")) == 1
    assert "same" in caplog.records[0].getMessage()
    assert outcome_of(events) == {"type": "success"}


async def test_a_task_id_still_open_stays_open_until_its_last_delegation_ends() -> None:
    """Ends under one id cannot be told apart, so the first to arrive is not the one sent."""
    running, released = asyncio.Event(), asyncio.Event()
    slow = Agent("slow", config=TestConfig(ToolCallEvent(name="wait"), "slow result"))

    @slow.tool
    async def wait() -> str:
        """Hold the slow delegation open."""
        running.set()
        await released.wait()
        return "waited"

    fast = Agent("fast", config=TestConfig("fast result"))
    parent = Agent("parent", config=TestConfig(ToolCallEvent(name="delegate_twice"), "summarised"))

    @parent.tool
    async def delegate_twice(context: Context) -> str:
        """Delegate twice under one id, the second while the first is still running."""
        first = asyncio.ensure_future(run_task(slow, "first", parent_context=context, task_id="same"))
        await running.wait()
        await run_task(fast, "second", parent_context=context, task_id="same")
        released.set()
        await first
        return "delegated"

    events = await _run(parent)

    [started] = every(events, "SUBAGENT_STARTED")
    assert started == IsPartialDict({"subagentRunId": "same", "description": "first"})
    assert every(events, "SUBAGENT_FINISHED") == [IsPartialDict({"subagentRunId": "same", "result": "slow result"})]
    assert outcome_of(events) == {"type": "success"}
