# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Each delegation reaches the client as one subagent invocation, started and then ended."""

import logging
from uuid import uuid4

import pytest
from ag_ui.core import UserMessage
from dirty_equals import IsInt, IsPartialDict, IsStr

from ag2 import Agent, Context
from ag2.ag_ui import AGUIStream
from ag2.events import TaskCompleted, TaskStarted, ToolCallEvent
from ag2.testing import TestConfig
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
    """An id a tool picks itself can repeat; the wire must still never see one invocation twice."""
    parent = Agent("parent", config=TestConfig(ToolCallEvent(name="delegate_twice"), "summarised"))

    @parent.tool
    async def delegate_twice(context: Context) -> str:
        """Report two delegations under one id."""
        for objective in ("first", "second"):
            await context.send(TaskStarted(task_id="same", agent_name="worker", objective=objective))
            await context.send(_completed("same", objective, "researched"))
        return "delegated"

    with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
        events = await _run(parent)

    assert [e["description"] for e in every(events, "SUBAGENT_STARTED")] == ["first"]
    assert len(every(events, "SUBAGENT_FINISHED")) == 1
    assert "same" in caplog.records[0].getMessage()
    assert outcome_of(events) == {"type": "success"}


async def test_a_task_id_still_open_stays_open_until_its_last_delegation_ends() -> None:
    """Ends under one id cannot be told apart, so the first to arrive is not the one sent."""
    parent = Agent("parent", config=TestConfig(ToolCallEvent(name="delegate_twice"), "summarised"))

    @parent.tool
    async def delegate_twice(context: Context) -> str:
        """Report a second delegation under one id that ends while the first still runs."""
        await context.send(TaskStarted(task_id="same", agent_name="slow", objective="first"))
        await context.send(TaskStarted(task_id="same", agent_name="fast", objective="second"))
        await context.send(_completed("same", "second", "fast result"))
        await context.send(_completed("same", "first", "slow result"))
        return "delegated"

    events = await _run(parent)

    [started] = every(events, "SUBAGENT_STARTED")
    assert started == IsPartialDict({"subagentRunId": "same", "description": "first"})
    assert every(events, "SUBAGENT_FINISHED") == [IsPartialDict({"subagentRunId": "same", "result": "slow result"})]
    assert outcome_of(events) == {"type": "success"}


def _completed(task_id: str, objective: str, result: str) -> TaskCompleted:
    return TaskCompleted(task_id=task_id, agent_name="worker", objective=objective, result=result, task_stream=uuid4())


class TestAnInvocationThatEndsWithoutFinishing:
    """Stopped, expired or never ended: each invocation still closes before its run does."""

    @staticmethod
    def _owning(end: str) -> Agent:
        parent = Agent("parent", config=TestConfig(ToolCallEvent(name="research"), "done"))

        @parent.tool
        async def research(context: Context) -> str:
            """Research, and stop the task the given way."""
            task = parent.task("research", context=context)
            await task.__aenter__()
            if end == "cancel":
                await task.cancel("no longer needed")
            elif end == "expire":
                await task.expire()
            return "stopped"

        return parent

    @pytest.mark.parametrize(("end", "message"), [("cancel", "cancelled: no longer needed"), ("expire", "expired")])
    async def test_a_stopped_task_is_a_subagent_error(self, end: str, message: str) -> None:
        events = await _run(self._owning(end))

        [started] = every(events, "SUBAGENT_STARTED")
        assert every(events, "SUBAGENT_ERROR") == [
            IsPartialDict({"subagentRunId": started["subagentRunId"], "message": message})
        ]
        assert outcome_of(events) == {"type": "success"}

    async def test_an_invocation_still_open_when_the_run_ends_is_closed_first(self) -> None:
        events = await _run(self._owning("never"))

        [started] = every(events, "SUBAGENT_STARTED")
        assert types_of(events)[-2:] == ["SUBAGENT_ERROR", "RUN_FINISHED"]
        assert every(events, "SUBAGENT_ERROR") == [IsPartialDict({"subagentRunId": started["subagentRunId"]})]
        assert outcome_of(events) == {"type": "success"}
