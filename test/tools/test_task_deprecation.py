# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from pathlib import Path

import pytest

from ag2 import Agent, MemoryStream, TaskConfig
from ag2.events import TaskCompleted, ToolCallEvent
from ag2.testing import TestConfig


def test_task_config_warns_at_caller() -> None:
    with pytest.warns(DeprecationWarning, match=r"TaskConfig.*removed in 1\.4\.0.*Agent\.as_tool") as records:
        config = TaskConfig(max_concurrency=2)

    assert config.max_concurrency == 2
    assert Path(records[0].filename) == Path(__file__)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("run_subtask", '{"task": "Find the answer"}'),
        ("run_subtasks", '{"tasks": ["Find the answer"], "parallel": true}'),
    ],
)
async def test_deprecated_tasks_still_delegate(name: str, arguments: str) -> None:
    with pytest.warns(DeprecationWarning, match="TaskConfig"):
        task_config = TaskConfig(config=TestConfig("The answer is 42."))

    with pytest.warns(DeprecationWarning, match=r"Agent\(tasks=.*removed in 1\.4\.0.*Agent\.as_tool") as records:
        agent = Agent(
            "coordinator",
            config=TestConfig(ToolCallEvent(name=name, arguments=arguments), "Done."),
            tasks=task_config,
        )

    assert Path(records[0].filename) == Path(__file__)
    assert agent.tasks is task_config

    stream = MemoryStream()
    reply = await agent.ask("Find the answer", stream=stream)

    assert reply.body == "Done."
    [completed] = [event for event in await stream.history.get_events() if isinstance(event, TaskCompleted)]
    assert completed.result == "The answer is 42."


@pytest.mark.asyncio
async def test_explicit_delegation_and_disabled_tasks_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        child = Agent("worker", config=TestConfig("The answer is 42."))
        agent = Agent(
            "coordinator",
            config=TestConfig(
                ToolCallEvent(name="task_worker", arguments='{"objective": "Find the answer"}'),
                "Done.",
            ),
            tools=[child.as_tool(description="Find the answer.")],
        )
        disabled = Agent("disabled", tasks=False)
        assert disabled.tasks is None
        assert agent.tasks is None

        stream = MemoryStream()
        reply = await agent.ask("Find the answer", stream=stream)

    assert reply.body == "Done."
    [completed] = [event for event in await stream.history.get_events() if isinstance(event, TaskCompleted)]
    assert completed.result == "The answer is 42."
