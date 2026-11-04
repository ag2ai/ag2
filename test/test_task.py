# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the framework-core ``Task`` primitive.

Covers standalone usage with no hub or network — pure
``ag2.task`` lifecycle, event emission, dependency stamping
for ``TaskInject``, and TTL field population.
"""

import asyncio
from typing import Any

import pytest
from dirty_equals import IsList

from ag2 import Agent, Context, TaskInject
from ag2.config import AnthropicConfig
from ag2.context import ConversationContext
from ag2.events import (
    BaseEvent,
    ModelResponse,
    TaskCompleted,
    TaskExpired,
    TaskFailed,
    TaskProgress,
    TaskStarted,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from ag2.middleware import ToolExecution, ToolResultType
from ag2.stream import MemoryStream
from ag2.task import TERMINAL_TASK_STATES, Task, TaskSpec, TaskState
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools import tool


def _agent() -> Agent:
    """A bare Agent — no model calls, just a name."""
    return Agent(name="researcher", config=AnthropicConfig(model="claude-sonnet-5"))


async def _persisted(stream: MemoryStream) -> list[BaseEvent]:
    """Read durably-persisted events from a MemoryStream's storage.

    Note: transient events (``TaskProgress``, ``ModelMessageChunk``) are
    not persisted — use ``_subscribe`` to capture them live.
    """
    return list(await stream.history.storage.get_history(stream.id))


def _subscribe(stream: MemoryStream) -> list[BaseEvent]:
    """Subscribe to ``stream`` and return a list mutated as events flow.

    Captures all events (including transient). Subscription must happen
    before any event of interest fires.
    """
    events: list[BaseEvent] = []
    stream.subscribe(lambda ev: events.append(ev), sync_to_thread=False)
    return events


class TestTaskState:
    def test_terminal_set(self) -> None:
        assert TaskState.COMPLETED in TERMINAL_TASK_STATES
        assert TaskState.FAILED in TERMINAL_TASK_STATES
        assert TaskState.EXPIRED in TERMINAL_TASK_STATES
        assert TaskState.CREATED not in TERMINAL_TASK_STATES
        assert TaskState.RUNNING not in TERMINAL_TASK_STATES


class TestTaskSpec:
    def test_defaults(self) -> None:
        spec = TaskSpec(title="hello")
        assert spec.title == "hello"
        assert spec.description == ""
        assert spec.payload == {}

    def test_payload_isolated(self) -> None:
        # Default factory produces a fresh dict each construction.
        a = TaskSpec(title="a")
        b = TaskSpec(title="b")
        a.payload["k"] = 1
        assert "k" not in b.payload


class TestStandaloneLifecycle:
    """Task with no bound context — creates its own MemoryStream."""

    @pytest.mark.asyncio
    async def test_clean_exit_auto_completes(self) -> None:
        agent = _agent()
        async with Task(owner_id=agent.name, spec=TaskSpec(title="research X")) as task:
            assert task.state == TaskState.RUNNING
            assert task.metadata.spec.title == "research X"
            stream = task.context.stream

        assert task.state == TaskState.COMPLETED
        events = await _persisted(stream)
        types = [type(e) for e in events]
        assert TaskStarted in types
        assert TaskCompleted in types

    @pytest.mark.asyncio
    async def test_explicit_complete_records_result(self) -> None:
        agent = _agent()
        async with Task(owner_id=agent.name, spec=TaskSpec(title="compute")) as task:
            await task.complete({"answer": 42})
            assert task.state == TaskState.COMPLETED
            assert task.metadata.result == {"answer": 42}
            stream = task.context.stream

        events = await _persisted(stream)
        completed = [e for e in events if isinstance(e, TaskCompleted)]
        assert len(completed) == 1
        assert completed[0].result == {"answer": 42}
        assert completed[0].agent_name == "researcher"
        assert completed[0].objective == "compute"

    @pytest.mark.asyncio
    async def test_complete_is_idempotent(self) -> None:
        agent = _agent()
        async with Task(owner_id=agent.name, spec=TaskSpec(title="once")) as task:
            await task.complete("first")
            await task.complete("second")  # no-op
            assert task.metadata.result == "first"
            stream = task.context.stream

        events = await _persisted(stream)
        assert sum(1 for e in events if isinstance(e, TaskCompleted)) == 1

    @pytest.mark.asyncio
    async def test_exception_emits_task_failed_and_propagates(self) -> None:
        agent = _agent()
        boom = ValueError("kaboom")
        captured_stream = None
        captured_task = None

        with pytest.raises(ValueError) as exc_info:
            async with Task(owner_id=agent.name, spec=TaskSpec(title="flaky")) as task:
                captured_task = task
                captured_stream = task.context.stream
                raise boom

        assert exc_info.value is boom
        assert captured_task is not None
        assert captured_task.state == TaskState.FAILED
        assert captured_task.metadata.error == "kaboom"

        assert captured_stream is not None
        events = await _persisted(captured_stream)
        failed = [e for e in events if isinstance(e, TaskFailed)]
        assert len(failed) == 1
        assert failed[0].error is boom

    @pytest.mark.asyncio
    async def test_explicit_fail_with_string(self) -> None:
        agent = _agent()
        async with Task(owner_id=agent.name, spec=TaskSpec(title="rejected")) as task:
            await task.fail("not authorized")
            assert task.state == TaskState.FAILED
            assert task.metadata.error == "not authorized"
            stream = task.context.stream

        events = await _persisted(stream)
        failed = [e for e in events if isinstance(e, TaskFailed)]
        assert len(failed) == 1
        assert isinstance(failed[0].error, RuntimeError)
        assert str(failed[0].error) == "not authorized"


class TestProgress:
    @pytest.mark.asyncio
    async def test_progress_accumulates_into_metadata(self) -> None:
        agent = _agent()
        stream = MemoryStream()
        ctx = ConversationContext(stream=stream)
        events = _subscribe(stream)

        async with Task(owner_id=agent.name, spec=TaskSpec(title="pipeline"), context=ctx) as task:
            await task.progress({"step": "search"})
            await task.progress({"hits": 12})
            assert task.metadata.progress == {"step": "search", "hits": 12}
            assert task.metadata.last_progress_at is not None

        progress_events = [e for e in events if isinstance(e, TaskProgress)]
        assert len(progress_events) == 2
        assert progress_events[0].payload == {"step": "search"}
        assert progress_events[1].payload == {"hits": 12}

    @pytest.mark.asyncio
    async def test_progress_after_terminal_is_noop(self) -> None:
        agent = _agent()
        stream = MemoryStream()
        ctx = ConversationContext(stream=stream)
        events = _subscribe(stream)

        async with Task(owner_id=agent.name, spec=TaskSpec(title="done-fast"), context=ctx) as task:
            await task.complete("ok")
            await task.progress({"too": "late"})
            assert "too" not in task.metadata.progress

        assert not [e for e in events if isinstance(e, TaskProgress)]


class TestExpire:
    @pytest.mark.asyncio
    async def test_expire_emits_task_expired(self) -> None:
        agent = _agent()
        async with Task(owner_id=agent.name, spec=TaskSpec(title="slow")) as task:
            await task.expire()
            assert task.state == TaskState.EXPIRED
            stream = task.context.stream

        events = await _persisted(stream)
        expired = [e for e in events if isinstance(e, TaskExpired)]
        assert len(expired) == 1
        assert expired[0].objective == "slow"


class TestBoundContext:
    """Task fed an explicit ConversationContext — events flow on caller's stream."""

    @pytest.mark.asyncio
    async def test_events_fire_on_supplied_stream(self) -> None:
        agent = _agent()
        stream = MemoryStream()
        ctx = ConversationContext(stream=stream)
        events = _subscribe(stream)

        async with Task(owner_id=agent.name, spec=TaskSpec(title="with-ctx"), context=ctx) as task:
            await task.progress({"k": "v"})
            await task.complete("done")

        types = [type(e) for e in events]
        assert types.count(TaskStarted) == 1
        assert types.count(TaskProgress) == 1
        assert types.count(TaskCompleted) == 1

    @pytest.mark.asyncio
    async def test_task_inject_stamped_during_block(self) -> None:
        agent = _agent()
        stream = MemoryStream()
        ctx = ConversationContext(stream=stream)
        assert "ag2.task" not in ctx.dependencies

        async with Task(owner_id=agent.name, spec=TaskSpec(title="inject-test"), context=ctx) as task:
            assert ctx.dependencies["ag2.task"] is task

        assert "ag2.task" not in ctx.dependencies

    @pytest.mark.asyncio
    async def test_nested_tasks_restore_previous(self) -> None:
        agent = _agent()
        stream = MemoryStream()
        ctx = ConversationContext(stream=stream)

        async with Task(owner_id=agent.name, spec=TaskSpec(title="outer"), context=ctx) as outer:
            assert ctx.dependencies["ag2.task"] is outer
            async with Task(owner_id=agent.name, spec=TaskSpec(title="inner"), context=ctx) as inner:
                assert ctx.dependencies["ag2.task"] is inner
            assert ctx.dependencies["ag2.task"] is outer

        assert "ag2.task" not in ctx.dependencies


class TestTtl:
    @pytest.mark.asyncio
    async def test_ttl_seconds_populates_expires_at(self) -> None:
        agent = _agent()
        async with Task(owner_id=agent.name, spec=TaskSpec(title="with-ttl"), ttl_seconds=60) as task:
            assert task.metadata.expires_at is not None
            assert task.metadata.started_at is not None
            # expires_at should be later than started_at
            assert task.metadata.expires_at > task.metadata.started_at

    @pytest.mark.asyncio
    async def test_no_ttl_leaves_expires_at_none(self) -> None:
        agent = _agent()
        async with Task(owner_id=agent.name, spec=TaskSpec(title="no-ttl")) as task:
            assert task.metadata.expires_at is None


class TestPropertiesBeforeEntry:
    def test_state_returns_created(self) -> None:
        agent = _agent()
        task = Task(owner_id=agent.name, spec=TaskSpec(title="not-yet-entered"))
        assert task.state == TaskState.CREATED

    def test_task_id_raises_before_entry(self) -> None:
        agent = _agent()
        task = Task(owner_id=agent.name, spec=TaskSpec(title="not-yet-entered"))
        with pytest.raises(RuntimeError, match="before __aenter__"):
            _ = task.task_id

    def test_metadata_raises_before_entry(self) -> None:
        agent = _agent()
        task = Task(owner_id=agent.name, spec=TaskSpec(title="not-yet-entered"))
        with pytest.raises(RuntimeError, match="before __aenter__"):
            _ = task.metadata

    @pytest.mark.asyncio
    async def test_double_enter_raises(self) -> None:
        agent = _agent()
        task = Task(owner_id=agent.name, spec=TaskSpec(title="once-only"))
        async with task:
            pass
        with pytest.raises(RuntimeError, match="already entered"):
            async with task:
                pass


class TestMetadata:
    @pytest.mark.asyncio
    async def test_owner_id_is_agent_name(self) -> None:
        agent = _agent()
        async with Task(owner_id=agent.name, spec=TaskSpec(title="ownership")) as task:
            assert task.metadata.owner_id == "researcher"
            assert task.metadata.spec.title == "ownership"

    @pytest.mark.asyncio
    async def test_payload_passed_through(self) -> None:
        agent = _agent()
        async with Task(
            owner_id=agent.name, spec=TaskSpec(title="payload-test", payload={"capability": "search"})
        ) as task:
            assert task.metadata.spec.payload == {"capability": "search"}


async def _report(note: str, task: TaskInject) -> str:
    """Report progress on the active task."""
    if task is None:
        return "no active task"
    await task.progress({"note": note})
    return "reported"


def _report_call(note: str) -> ToolCallEvent:
    return ToolCallEvent(id=note, name="report", arguments=f'{{"note": "{note}"}}')


def _reporter(note: str) -> Agent:
    """An agent whose one turn calls ``report(note)``."""
    return Agent(
        "reporter",
        config=TestConfig(ModelResponse(tool_calls=ToolCallsEvent([_report_call(note)])), "Done"),
        tools=[tool(_report, name="report")],
    )


def _progress(events: list[BaseEvent]) -> list[tuple[str, dict[str, Any]]]:
    """``(task title, payload)`` of every ``TaskProgress`` seen, in order."""
    return [(event.objective, event.payload) for event in events if isinstance(event, TaskProgress)]


@pytest.mark.asyncio
class TestTaskInject:
    """A ``@tool`` reports progress through ``TaskInject`` to the task whose
    ``Task(...)`` block encloses the call."""

    async def test_parallel_tool_calls_report_to_their_own_task(self) -> None:
        """Tool middleware opens a task per call; with every block open at once,
        each call's progress lands on its own task, not on the last one opened."""
        notes = ["A", "B", "C"]
        opened: set[str] = set()
        all_open = asyncio.Event()

        async def task_per_call(call_next: ToolExecution, event: ToolCallEvent, ctx: Context) -> ToolResultType:
            async with Task(owner_id=agent.name, spec=TaskSpec(title=event.id), context=ctx):
                opened.add(event.id)
                if opened == set(notes):
                    all_open.set()
                await all_open.wait()
                return await call_next(event, ctx)

        agent = Agent(
            "planner",
            config=TestConfig(ModelResponse(tool_calls=ToolCallsEvent([_report_call(n) for n in notes])), "Done"),
            tools=[tool(_report, name="report", middleware=[task_per_call])],
        )
        stream = MemoryStream()
        events = _subscribe(stream)

        await agent.ask("Run the steps.", stream=stream)

        assert _progress(events) == IsList(*((n, {"note": n}) for n in notes), check_order=False)

    @pytest.mark.parametrize("capability", [None, "reporting"])
    async def test_turn_inside_a_block_reports_to_it(self, capability: str | None) -> None:
        stream = MemoryStream()
        events = _subscribe(stream)

        async with Task(owner_id="researcher", spec=TaskSpec(title="job"), context=ConversationContext(stream=stream)):
            await _reporter("progress").ask("Report.", stream=stream, capability=capability)

        assert _progress(events) == [("job", {"note": "progress"})]

    async def test_nested_blocks_report_to_the_innermost(self) -> None:
        owner = _agent()
        stream = MemoryStream()
        ctx = ConversationContext(stream=stream)
        events = _subscribe(stream)

        async with Task(owner_id=owner.name, spec=TaskSpec(title="outer"), context=ctx):
            async with Task(owner_id=owner.name, spec=TaskSpec(title="inner"), context=ctx):
                await _reporter("inside inner").ask("Report.", stream=stream)
            await _reporter("after inner").ask("Report.", stream=stream)
        await _reporter("after outer").ask("Report.", stream=stream)

        assert _progress(events) == [
            ("inner", {"note": "inside inner"}),
            ("outer", {"note": "after inner"}),
        ]

    async def test_without_a_block_there_is_no_task(self) -> None:
        call = _report_call("progress")
        tracking = TrackingConfig(TestConfig(ModelResponse(tool_calls=ToolCallsEvent([call])), "Done"))
        agent = Agent("reporter", config=tracking, tools=[tool(_report, name="report")])
        stream = MemoryStream()
        events = _subscribe(stream)

        await agent.ask("Report.", stream=stream)

        tracking.mock.assert_called_with(ToolResultsEvent([ToolResultEvent.from_call(call, "no active task")]))
        assert _progress(events) == []

    async def test_a_turn_on_another_stream_does_not_see_the_block(self) -> None:
        stream = MemoryStream()
        events = _subscribe(stream)

        async with Task(owner_id="researcher", spec=TaskSpec(title="job"), context=ConversationContext(stream=stream)):
            await _reporter("elsewhere").ask("Report.", stream=MemoryStream())

        assert _progress(events) == []

    async def test_a_subagent_does_not_see_the_parent_task(self) -> None:
        """The parent opens a task around the delegation; the worker runs its own
        conversation and does not report to it."""

        async def task_per_call(call_next: ToolExecution, event: ToolCallEvent, ctx: Context) -> ToolResultType:
            async with Task(owner_id=parent.name, spec=TaskSpec(title="delegation"), context=ctx):
                return await call_next(event, ctx)

        worker = _reporter("from worker")
        parent = Agent(
            "parent",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent([
                        ToolCallEvent(name="task_reporter", arguments='{"objective": "Report."}')
                    ])
                ),
                "Done",
            ),
            tools=[worker.as_tool(description="Reports on the active task.", middleware=[task_per_call])],
        )
        stream = MemoryStream()
        events = _subscribe(stream)

        await parent.ask("Delegate.", stream=stream)

        assert _progress(events) == []

    async def test_a_task_passed_as_a_dependency_is_injected(self) -> None:
        task_stream = MemoryStream()
        events = _subscribe(task_stream)

        async with Task(
            owner_id="researcher", spec=TaskSpec(title="given"), context=ConversationContext(stream=task_stream)
        ) as given:
            await _reporter("progress").ask("Report.", dependencies={"ag2.task": given})

        assert _progress(events) == [("given", {"note": "progress"})]
