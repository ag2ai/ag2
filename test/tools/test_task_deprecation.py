# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import copy
import dataclasses
import pickle
import warnings
from pathlib import Path

import pytest

from ag2 import Agent, MemoryStream, TaskConfig
from ag2.agent import TASK_CONFIG_DEPRECATION
from ag2.events import TaskCompleted, ToolCallEvent
from ag2.testing import TestConfig, TrackingConfig


def test_task_config_is_flagged_for_type_checkers_only() -> None:
    assert TaskConfig.__deprecated__ == TASK_CONFIG_DEPRECATION  # type: ignore[attr-defined]

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        config = TaskConfig(max_concurrency=2)
        dataclasses.replace(config, max_concurrency=3)
        copy.deepcopy(config)
        pickle.loads(pickle.dumps(config))

    assert config.max_concurrency == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("run_subtask", '{"task": "Find the answer"}'),
        ("run_subtasks", '{"tasks": ["Find the answer"], "parallel": true}'),
    ],
)
async def test_deprecated_tasks_warn_once_and_still_delegate(name: str, arguments: str) -> None:
    task_config = TaskConfig(config=TestConfig("The answer is 42."))

    with pytest.warns(DeprecationWarning) as records:
        agent = Agent(
            "coordinator",
            config=TestConfig(ToolCallEvent(name=name, arguments=arguments), "Done."),
            tasks=task_config,
        )

    [record] = records
    assert str(record.message) == TASK_CONFIG_DEPRECATION
    assert Path(record.filename) == Path(__file__)
    assert agent.tasks is task_config

    stream = MemoryStream()
    reply = await agent.ask("Find the answer", stream=stream)

    assert reply.body == "Done."
    [completed] = [event for event in await stream.history.get_events() if isinstance(event, TaskCompleted)]
    assert completed.result == "The answer is 42."


@pytest.mark.asyncio
async def test_tasks_none_is_disabled_without_warning() -> None:
    tracking = TrackingConfig(TestConfig("Done."))

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        agent = Agent("legacy-default", config=tracking, tasks=None)  # type: ignore[call-overload]

    await agent.ask("Hello")

    assert agent.tasks is None
    [call] = tracking.calls
    assert call.tools == ()


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
