# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from pathlib import Path
from typing import Any

import pytest
from dirty_equals import HasAttributes, IsInstance, IsStr

from ag2 import Agent, Context, TaskInject
from ag2.events import BaseEvent, TaskCompleted, TaskFailed, TaskStarted, ToolCallEvent
from ag2.stream import MemoryStream
from ag2.task import Task, TaskSpec, TaskState
from ag2.testing import TestConfig

DEPRECATION = r"Agent\.task\(\) is deprecated and will be removed in 1\.4\.0"


class Checkpoints:
    def __init__(self) -> None:
        self.states: dict[str, dict[str, Any]] = {"prior": {"next_document": 12}}

    async def read(self, task_id: str) -> dict[str, Any] | None:
        return self.states.get(task_id)

    async def write(self, task_id: str, state: dict[str, Any]) -> None:
        self.states[task_id] = state


async def report_progress(task: TaskInject) -> str:
    """Report progress on the caller's task."""
    if task is not None:
        await task.progress({"from_tool": True})
    return "ok"


def task_events(events: list[BaseEvent]) -> list[BaseEvent]:
    return [event for event in events if isinstance(event, (TaskStarted, TaskCompleted, TaskFailed))]


def legacy_task(agent: Agent, payload: dict[str, Any], context: Context, store: Checkpoints) -> Task:
    with pytest.warns(DeprecationWarning, match=DEPRECATION) as records:
        task = agent.task(
            "index documents",
            description="resume indexing",
            payload=payload,
            capability="indexing",
            context=context,
            ttl_seconds=60,
            checkpoint_store=store,
            resume_from="prior",
        )
    [record] = records
    # The warning points at the user's call site, not at the framework.
    assert Path(record.filename) == Path(__file__)
    return task


def direct_task(agent: Agent, payload: dict[str, Any], context: Context, store: Checkpoints) -> Task:
    return Task(
        owner_id=agent.name,
        spec=TaskSpec(
            title="index documents",
            description="resume indexing",
            payload=dict(payload),
            capability="indexing",
        ),
        context=context,
        ttl_seconds=60,
        checkpoint_store=store,
        resume_from="prior",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("build", [direct_task, legacy_task], ids=["direct-task", "deprecated-factory"])
async def test_migrating_to_direct_task_keeps_the_same_behaviour(build: Any) -> None:
    agent = Agent("indexer", config=TestConfig(ToolCallEvent(name="report_progress"), "done"), tools=[report_progress])
    stream = MemoryStream()
    context = Context(stream=stream)
    store = Checkpoints()
    payload = {"region": "eu"}

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        task = build(agent, payload, context, store)
    payload["region"] = "us"

    async with task:
        resumed = task.resumed_state
        await agent.ask("index", stream=stream, dependencies=context.dependencies)
        await task.checkpoint({"next_document": 24})
        await task.complete({"indexed": 24})

    assert resumed == {"next_document": 12}
    assert task.metadata.progress == {"from_tool": True}
    assert store.states[task.task_id] == {"next_document": 24}
    assert task.state == TaskState.COMPLETED
    spec = TaskSpec(
        title="index documents",
        description="resume indexing",
        payload={"region": "eu"},
        capability="indexing",
    )
    assert task_events(await stream.history.get_events()) == [
        IsInstance(TaskStarted)
        & HasAttributes(task_id=task.task_id, agent_name="indexer", spec=spec, expires_at=IsStr()),
        IsInstance(TaskCompleted) & HasAttributes(task_id=task.task_id, result={"indexed": 24}),
    ]


@pytest.mark.asyncio
async def test_deprecated_factory_still_fails_the_task_on_error() -> None:
    with pytest.warns(DeprecationWarning, match=DEPRECATION):
        task = Agent("worker").task("flaky")

    with pytest.raises(ValueError, match="boom"):
        async with task:
            raise ValueError("boom")

    assert task.state == TaskState.FAILED
    assert [type(event) for event in await task.context.stream.history.get_events()] == [TaskStarted, TaskFailed]
