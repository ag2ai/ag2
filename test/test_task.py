# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the framework-core ``Task`` primitive.

Covers standalone usage with no hub or network — pure
``ag2.task`` lifecycle, event emission, dependency stamping
for ``TaskInject``, and TTL field population.
"""

import asyncio
from itertools import permutations

import pytest

from ag2 import Agent, Context, TaskInject, TaskSpec
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
)
from ag2.middleware import ToolExecution, ToolResultType
from ag2.stream import MemoryStream
from ag2.task import TERMINAL_TASK_STATES, Task, TaskState
from ag2.testing import TestConfig
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
        async with agent.task("research X") as task:
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
        async with agent.task("compute") as task:
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
        async with agent.task("once") as task:
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
            async with agent.task("flaky") as task:
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
        async with agent.task("rejected") as task:
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

        async with agent.task("pipeline", context=ctx) as task:
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

        async with agent.task("done-fast", context=ctx) as task:
            await task.complete("ok")
            await task.progress({"too": "late"})
            assert "too" not in task.metadata.progress

        assert not [e for e in events if isinstance(e, TaskProgress)]


class TestExpire:
    @pytest.mark.asyncio
    async def test_expire_emits_task_expired(self) -> None:
        agent = _agent()
        async with agent.task("slow") as task:
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

        async with agent.task("with-ctx", context=ctx) as task:
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

        async with agent.task("inject-test", context=ctx) as task:
            assert ctx.dependencies["ag2.task"] is task

        assert "ag2.task" not in ctx.dependencies

    @pytest.mark.asyncio
    async def test_nested_tasks_restore_previous(self) -> None:
        agent = _agent()
        stream = MemoryStream()
        ctx = ConversationContext(stream=stream)

        async with agent.task("outer", context=ctx) as outer:
            assert ctx.dependencies["ag2.task"] is outer
            async with agent.task("inner", context=ctx) as inner:
                assert ctx.dependencies["ag2.task"] is inner
            assert ctx.dependencies["ag2.task"] is outer

        assert "ag2.task" not in ctx.dependencies


class TestTtl:
    @pytest.mark.asyncio
    async def test_ttl_seconds_populates_expires_at(self) -> None:
        agent = _agent()
        async with agent.task("with-ttl", ttl_seconds=60) as task:
            assert task.metadata.expires_at is not None
            assert task.metadata.started_at is not None
            # expires_at should be later than started_at
            assert task.metadata.expires_at > task.metadata.started_at

    @pytest.mark.asyncio
    async def test_no_ttl_leaves_expires_at_none(self) -> None:
        agent = _agent()
        async with agent.task("no-ttl") as task:
            assert task.metadata.expires_at is None


class TestPropertiesBeforeEntry:
    def test_state_returns_created(self) -> None:
        agent = _agent()
        task = agent.task("not-yet-entered")
        assert task.state == TaskState.CREATED

    def test_task_id_raises_before_entry(self) -> None:
        agent = _agent()
        task = agent.task("not-yet-entered")
        with pytest.raises(RuntimeError, match="before __aenter__"):
            _ = task.task_id

    def test_metadata_raises_before_entry(self) -> None:
        agent = _agent()
        task = agent.task("not-yet-entered")
        with pytest.raises(RuntimeError, match="before __aenter__"):
            _ = task.metadata

    @pytest.mark.asyncio
    async def test_double_enter_raises(self) -> None:
        agent = _agent()
        task = agent.task("once-only")
        async with task:
            pass
        with pytest.raises(RuntimeError, match="already entered"):
            async with task:
                pass


class TestMetadata:
    @pytest.mark.asyncio
    async def test_owner_id_is_agent_name(self) -> None:
        agent = _agent()
        async with agent.task("ownership") as task:
            assert task.metadata.owner_id == "researcher"
            assert task.metadata.spec.title == "ownership"

    @pytest.mark.asyncio
    async def test_payload_passed_through(self) -> None:
        agent = _agent()
        async with agent.task("payload-test", payload={"capability": "search"}) as task:
            assert task.metadata.spec.payload == {"capability": "search"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exit_order",
    [*permutations("AB"), *permutations("ABC")],
    ids="".join,
)
async def test_parallel_tool_tasks_inject_latest_running_task(exit_order: tuple[str, ...]) -> None:
    """Sibling tool calls in one turn each open a task on the shared context.
    A tool resolving ``TaskInject`` after each exit gets the latest-entered
    task still running — never an exited one — and ``None`` after the last."""
    names = sorted(exit_order)
    # One `step_X` call opens task X; one `probe_X` call injects the task once X exited.
    gates = {f"{kind}_{n}": asyncio.Event() for kind in ("step", "probe") for n in names}
    done = {call_id: asyncio.Event() for call_id in gates}
    entered = {n: asyncio.Event() for n in names}
    # Steps enter one after another, so entry order is ``names``.
    entered_before = dict(zip(names[1:], names))
    tasks: dict[str, Task] = {}
    injected: dict[str, Task | None] = {}

    async def hold_until_released(call_next: ToolExecution, event: ToolCallEvent, ctx: Context) -> ToolResultType:
        await gates[event.id].wait()
        return await call_next(event, ctx)

    @tool
    async def step(name: str, ctx: Context) -> str:
        if name in entered_before:
            await entered[entered_before[name]].wait()
        async with agent.task(name, context=ctx) as task:
            tasks[name] = task
            entered[name].set()
            await gates[f"step_{name}"].wait()
        done[f"step_{name}"].set()
        return name

    @tool(middleware=[hold_until_released])
    async def probe(after: str, task: TaskInject) -> str:
        injected[after] = task
        done[f"probe_{after}"].set()
        return after

    agent = Agent(
        "researcher",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent([
                    *(ToolCallEvent(id=f"step_{n}", name="step", arguments=f'{{"name": "{n}"}}') for n in names),
                    *(ToolCallEvent(id=f"probe_{n}", name="probe", arguments=f'{{"after": "{n}"}}') for n in names),
                ]),
            ),
            "Done",
        ),
        tools=[step, probe],
    )

    reply = asyncio.create_task(agent.ask("Run the steps."))
    await asyncio.gather(*(e.wait() for e in entered.values()))

    running = list(names)
    expected: dict[str, Task | None] = {}
    for name in exit_order:
        for call_id in (f"step_{name}", f"probe_{name}"):
            gates[call_id].set()
            await done[call_id].wait()
        running.remove(name)
        expected[name] = tasks[running[-1]] if running else None

    assert (await reply).body == "Done"
    assert injected == expected
