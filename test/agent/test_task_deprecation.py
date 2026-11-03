# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from pathlib import Path
from typing import Any

import pytest

from ag2 import Agent, Context
from ag2.agent import AGENT_TASK_DEPRECATION
from ag2.events import TaskCompleted, TaskFailed, TaskStarted
from ag2.stream import MemoryStream
from ag2.task import Task, TaskSpec, TaskState


class Checkpoints:
    def __init__(self) -> None:
        self.states: dict[str, dict[str, Any]] = {"prior": {"next_document": 12}}

    async def read(self, task_id: str) -> dict[str, Any] | None:
        return self.states.get(task_id)

    async def write(self, task_id: str, state: dict[str, Any]) -> None:
        self.states[task_id] = state


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True], ids=["direct-task", "deprecated-factory"])
async def test_migration_preserves_lifecycle_metadata_injection_and_recovery(legacy: bool) -> None:
    agent = Agent("indexer")
    stream = MemoryStream()
    context = Context(stream=stream)
    store = Checkpoints()
    payload = {"region": "eu"}

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        if legacy:
            with pytest.warns(DeprecationWarning) as records:
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
            assert str(record.message) == AGENT_TASK_DEPRECATION
            assert Path(record.filename) == Path(__file__)
        else:
            task = Task(
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

        # Both constructors are inert; the deprecated factory still copies the payload.
        payload["region"] = "us"
        assert task.state == TaskState.CREATED
        assert await stream.history.get_events() == []
        assert "ag2.task" not in context.dependencies

        async with task:
            assert task.context is context
            assert context.dependencies["ag2.task"] is task
            assert task.resumed_state == {"next_document": 12}
            assert task.metadata.owner_id == agent.name
            assert task.metadata.spec == TaskSpec(
                title="index documents",
                description="resume indexing",
                payload={"region": "eu"},
                capability="indexing",
            )
            await task.checkpoint({"next_document": 24})
            await task.complete({"indexed": 24})

    assert "ag2.task" not in context.dependencies
    assert task.state == TaskState.COMPLETED
    assert store.states[task.task_id] == {"next_document": 24}
    started, completed = await stream.history.get_events()
    assert isinstance(started, TaskStarted)
    assert started.task_id == task.task_id
    assert started.spec == task.metadata.spec
    assert started.expires_at == task.metadata.expires_at
    assert started.expires_at is not None
    assert isinstance(completed, TaskCompleted)
    assert completed.task_id == task.task_id
    assert completed.result == {"indexed": 24}


@pytest.mark.asyncio
async def test_deprecated_factory_still_auto_fails() -> None:
    with pytest.warns(DeprecationWarning, match=r"Agent\.task\(\).*1\.4\.0"):
        task = Agent("worker").task("flaky")

    with pytest.raises(ValueError, match="boom"):
        async with task:
            raise ValueError("boom")

    assert task.state == TaskState.FAILED
    events = await task.context.stream.history.get_events()
    assert [type(event) for event in events] == [TaskStarted, TaskFailed]


def test_method_is_marked_deprecated_for_type_checkers() -> None:
    assert Agent.task.__deprecated__ == AGENT_TASK_DEPRECATION  # type: ignore[attr-defined]
