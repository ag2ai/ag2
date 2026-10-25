# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from ag2 import Agent, Context, MemoryStream, tool
from ag2.events import (
    ModelMessage,
    ModelResponse,
    TaskCancelled,
    TaskCompleted,
    TaskFailed,
    TaskStarted,
    ToolCallEvent,
    ToolCallsEvent,
    Usage,
    UsageEvent,
)
from ag2.testing import TestConfig
from ag2.tools.subagents.run_task import run_task
from ag2.usage import UsageReport

pytestmark = pytest.mark.asyncio


@pytest.fixture()
def parent_context() -> Context:
    return Context(MemoryStream())


@pytest.fixture()
def spend() -> Usage:
    return Usage(prompt_tokens=250, completion_tokens=50, total_tokens=300)


@pytest.fixture()
def blocked_worker(spend: Usage) -> tuple[Agent, asyncio.Event]:
    started = asyncio.Event()

    @tool
    async def wait_for_work() -> str:
        """Wait for the delegated external operation."""
        started.set()
        await asyncio.Event().wait()
        return "done"

    worker = Agent(
        "worker",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent([ToolCallEvent(name="wait_for_work", arguments="{}")]),
                usage=spend,
            ),
        ),
        tools=[wait_for_work],
    )
    return worker, started


async def test_cancelled_worker_reports_billed_usage_and_cancellation(
    blocked_worker: tuple[Agent, asyncio.Event], parent_context: Context, spend: Usage
) -> None:
    worker, started = blocked_worker
    child_stream = MemoryStream(storage=parent_context.stream.history.storage)
    task = asyncio.create_task(run_task(worker, "go", parent_context=parent_context, stream=child_stream))
    await started.wait()
    task.cancel("requested by caller")

    with pytest.raises(asyncio.CancelledError, match="requested by caller"):
        await task

    events = await parent_context.stream.history.get_events()
    assert UsageReport.from_events(await child_stream.history.get_events()).total == spend
    assert UsageReport.from_events(events).total == spend
    assert [(event.usage, event.kind, event.label) for event in events if isinstance(event, UsageEvent)] == [
        (spend, "subtask", "worker"),
    ]
    assert [type(event) for event in events] == [TaskStarted, UsageEvent, TaskCancelled]
    [started_event, _, cancelled] = events
    assert cancelled.task_id == started_event.task_id
    assert cancelled.reason == "requested by caller"
    assert task.cancelled()


async def test_cancellation_before_spending_emits_no_empty_rollup(parent_context: Context) -> None:
    cancellation = asyncio.CancelledError("cancel before billing")
    worker = Agent("worker", config=TestConfig(cancellation))

    with pytest.raises(asyncio.CancelledError) as caught:
        await run_task(worker, "go", parent_context=parent_context)

    assert caught.value is cancellation
    events = await parent_context.stream.history.get_events()
    assert [type(event) for event in events] == [TaskStarted, TaskCancelled]
    assert not UsageReport.from_events(events).total


async def test_cancelled_worker_respects_disabled_events(
    blocked_worker: tuple[Agent, asyncio.Event], parent_context: Context
) -> None:
    worker, started = blocked_worker
    task = asyncio.create_task(run_task(worker, "go", parent_context=parent_context, emit_events=False))
    await started.wait()
    task.cancel("caller owns lifecycle")

    with pytest.raises(asyncio.CancelledError, match="caller owns lifecycle"):
        await task

    assert await parent_context.stream.history.get_events() == []


async def test_cancelled_delegation_counts_only_new_spend_on_reused_stream(
    blocked_worker: tuple[Agent, asyncio.Event], parent_context: Context, spend: Usage
) -> None:
    worker, started = blocked_worker
    child_stream = MemoryStream(storage=parent_context.stream.history.storage)
    earlier = Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110)
    await run_task(
        Agent("worker", config=TestConfig(ModelResponse(ModelMessage("done"), usage=earlier))),
        "first",
        parent_context=parent_context,
        stream=child_stream,
    )
    task = asyncio.create_task(run_task(worker, "second", parent_context=parent_context, stream=child_stream))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    events = await parent_context.stream.history.get_events()
    assert UsageReport.from_events(events).total == earlier + spend
    assert [event.usage for event in events if isinstance(event, UsageEvent)] == [earlier, spend]
    assert [type(event) for event in events if isinstance(event, (TaskCompleted, TaskFailed, TaskCancelled))] == [
        TaskCompleted,
        TaskCancelled,
    ]


async def test_nested_cancellation_rolls_up_each_level_once(
    blocked_worker: tuple[Agent, asyncio.Event], parent_context: Context, spend: Usage
) -> None:
    worker, started = blocked_worker
    dispatch = Usage(prompt_tokens=20, completion_tokens=2, total_tokens=22)
    child = Agent(
        "child",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent([
                    ToolCallEvent(name="task_worker", arguments='{"objective": "dig"}'),
                ]),
                usage=dispatch,
            ),
        ),
        tools=[worker.as_tool(description="Delegate to worker")],
    )
    child_stream = MemoryStream(storage=parent_context.stream.history.storage)
    task = asyncio.create_task(run_task(child, "top", parent_context=parent_context, stream=child_stream))
    await started.wait()
    task.cancel("cancel nested work")
    with pytest.raises(asyncio.CancelledError, match="cancel nested work"):
        await task

    events = await parent_context.stream.history.get_events()
    assert UsageReport.from_events(events).total == dispatch + spend
    assert [(event.usage, event.kind, event.label) for event in events if isinstance(event, UsageEvent)] == [
        (dispatch + spend, "subtask", "child"),
    ]
    assert [type(event) for event in events] == [TaskStarted, UsageEvent, TaskCancelled]
    child_events = await child_stream.history.get_events()
    assert [type(event) for event in child_events if isinstance(event, (TaskCompleted, TaskFailed, TaskCancelled))] == [
        TaskCancelled,
    ]


@pytest.mark.parametrize("report_event", [UsageEvent, TaskCompleted], ids=["rollup", "completion"])
async def test_cancellation_during_success_reporting_does_not_repeat_events(
    parent_context: Context, spend: Usage, report_event: type[UsageEvent] | type[TaskCompleted]
) -> None:
    reporting = asyncio.Event()

    async def wait_during_report() -> None:
        reporting.set()
        await asyncio.Event().wait()

    parent_context.stream.where(report_event).subscribe(wait_during_report)
    worker = Agent("worker", config=TestConfig(ModelResponse(ModelMessage("done"), usage=spend)))
    task = asyncio.create_task(run_task(worker, "go", parent_context=parent_context))
    await reporting.wait()
    task.cancel("cancel during reporting")
    with pytest.raises(asyncio.CancelledError, match="cancel during reporting"):
        await task

    events = await parent_context.stream.history.get_events()
    assert UsageReport.from_events(events).total == spend
    expected_types = [TaskStarted, UsageEvent]
    if report_event is TaskCompleted:
        expected_types.append(TaskCompleted)
    assert [type(event) for event in events] == expected_types


@pytest.mark.parametrize("report_event", [UsageEvent, TaskCancelled], ids=["rollup", "cancellation"])
async def test_reporting_failure_preserves_original_cancellation(
    blocked_worker: tuple[Agent, asyncio.Event],
    parent_context: Context,
    spend: Usage,
    report_event: type[UsageEvent] | type[TaskCancelled],
) -> None:
    async def unavailable_reporter() -> None:
        raise ConnectionError("reporter unavailable")

    parent_context.stream.where(report_event).subscribe(unavailable_reporter)
    worker, started = blocked_worker
    task = asyncio.create_task(run_task(worker, "go", parent_context=parent_context))
    await started.wait()
    task.cancel("caller cancellation")
    with pytest.raises(asyncio.CancelledError, match="caller cancellation"):
        await task

    events = await parent_context.stream.history.get_events()
    assert UsageReport.from_events(events).total == spend
    expected_types = [TaskStarted, UsageEvent]
    if report_event is TaskCancelled:
        expected_types.append(TaskCancelled)
    assert [type(event) for event in events] == expected_types
    assert task.cancelled()
