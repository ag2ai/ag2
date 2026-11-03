# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import AsyncExitStack, ExitStack
from datetime import datetime, timezone

import pytest
from dirty_equals import HasAttributes, IsInstance

from ag2 import Agent, Context, TaskInject
from ag2.context import StreamId
from ag2.events import (
    BaseEvent,
    HaltEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TaskCancelled,
    TaskCompleted,
    TaskExpired,
    TaskFailed,
    TaskProgress,
    TaskStarted,
    ToolCallEvent,
    Usage,
    UsageEvent,
)
from ag2.exceptions import ConfigNotProvidedError
from ag2.history import MemoryStorage
from ag2.stream import MemoryStream
from ag2.task import Task, TaskSpec, TaskState
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.subagents import DeliveryError

TASK_EVENTS = (TaskStarted, TaskProgress, TaskCompleted, TaskFailed, TaskCancelled, TaskExpired)


def task_events(events: list[BaseEvent]) -> list[BaseEvent]:
    return [event for event in events if isinstance(event, TASK_EVENTS)]


async def fail() -> str:
    """Fail the model turn."""
    raise ValueError("work failed")


class SlowTool:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cleaned_up = asyncio.Event()

    async def wait(self) -> str:
        """Wait until the invocation is cancelled."""
        self.entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cleaned_up.set()
        return "unreachable"


class FailingCleanup:
    """A user observer whose teardown raises."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def register(self, stack: ExitStack | AsyncExitStack, context: Context) -> None:
        stack.callback(self.close)

    def close(self) -> None:
        raise self.error


class TerminalProbe:
    """Tries to message a delegated worker once its terminal is observed."""

    def __init__(self) -> None:
        self.outcomes: list[str] = []

    async def check(self, event: TaskCompleted) -> None:
        try:
            await event.origin.send(HaltEvent(reason="already finished", source="test"))
        except DeliveryError:
            self.outcomes.append("retired")
        else:
            self.outcomes.append("still addressable")


class StallingStorage(MemoryStorage):
    """A store whose first history read after the model reply hangs, like a stalled round-trip."""

    def __init__(self) -> None:
        super().__init__()
        self.replied = False
        self.stalled = asyncio.Event()

    async def save_event(self, event: BaseEvent, context: Context) -> None:
        if isinstance(event, ModelResponse):
            self.replied = True
        await super().save_event(event, context)

    async def get_history(self, stream_id: StreamId) -> list[BaseEvent]:
        if self.replied and not self.stalled.is_set():
            self.stalled.set()
            await asyncio.Event().wait()
        return list(await super().get_history(stream_id))


class UsageWritesFail(MemoryStorage):
    """A store that cannot record a delegated worker's usage."""

    async def save_event(self, event: BaseEvent, context: Context) -> None:
        if isinstance(event, UsageEvent):
            raise ConnectionError("usage store unavailable")
        await super().save_event(event, context)


class LosesThenFailsUsage(MemoryStorage):
    """A store that silently loses the first usage record, then cannot record usage at all."""

    def __init__(self) -> None:
        super().__init__()
        self.lost = False

    async def save_event(self, event: BaseEvent, context: Context) -> None:
        if isinstance(event, UsageEvent):
            if not self.lost:
                self.lost = True
                return
            raise ConnectionError("usage store unavailable")
        await super().save_event(event, context)


class Abort(BaseException):
    """An interrupt-style error that is not an ``Exception``."""


def abort() -> str:
    """Abort the model turn."""
    raise Abort


async def expire_on_start(event: TaskStarted, ctx: Context) -> None:
    await ctx.send(TaskExpired(task_id=event.task_id, agent_name=event.agent_name, objective=event.objective))


def fail_on_start(event: TaskStarted) -> None:
    raise ValueError("startup observer failed")


async def report_progress(task: TaskInject) -> str:
    """Report progress on the caller's task."""
    if task is not None:
        await task.progress({"from_tool": True})
    return "ok"


async def index_documents(ctx: Context, tool: SlowTool) -> None:
    async with Task(owner_id="indexer", spec=TaskSpec(title="index"), context=ctx):
        await tool.wait()


def delegating_lead(worker: Agent, *replies: str) -> Agent:
    return Agent(
        "lead",
        config=TestConfig(
            ToolCallEvent(id="call-1", name="task_worker", arguments='{"objective": "summarise the report"}'),
            *replies,
        ),
        tools=[worker.as_tool(description="Summarise documents")],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["ask", "run"])
async def test_untagged_calls_publish_no_task_events(entry: str) -> None:
    agent = Agent("analyst", config=TestConfig("done"))

    if entry == "ask":
        reply = await agent.ask("analyse the data")
    else:
        async with agent.run("analyse the data") as run:
            reply = await run.result()

    assert task_events(await reply.history.get_events()) == []


@pytest.mark.asyncio
async def test_tagged_ask_reports_its_task_without_leaking_the_prompt() -> None:
    tracking = TrackingConfig(TestConfig("done"))
    agent = Agent("analyst", config=tracking)

    reply = await agent.ask("confidential: customer list", capability="analysis")

    started, completed = task_events(await reply.history.get_events())
    assert started == IsInstance(TaskStarted) & HasAttributes(
        agent_name="analyst",
        objective="analyst",
        spec=TaskSpec(title="analyst", capability="analysis"),
        expires_at=None,
        parent_tool_call_id=None,
    )
    assert completed == IsInstance(TaskCompleted) & HasAttributes(
        task_id=started.task_id,
        result="done",
        task_stream=reply.context.stream.id,
    )
    [call] = tracking.calls
    assert task_events(list(call.messages)) == []


@pytest.mark.asyncio
async def test_ttl_publishes_a_deadline() -> None:
    agent = Agent("analyst", config=TestConfig("done"))

    reply = await agent.ask("analyse", ttl_seconds=60)

    [started, _] = task_events(await reply.history.get_events())
    assert started == HasAttributes(spec=TaskSpec(title="analyst"))
    assert datetime.fromisoformat(started.expires_at) > datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_expiry_reported_by_an_observer_is_final_and_does_not_stop_the_run() -> None:
    stream = MemoryStream()
    stream.where(TaskStarted).subscribe(expire_on_start, sync_to_thread=False)
    agent = Agent("analyst", config=TestConfig("done"))

    reply = await agent.ask("analyse", stream=stream, ttl_seconds=1)

    assert reply.body == "done"
    assert [type(event) for event in task_events(await stream.history.get_events())] == [TaskStarted, TaskExpired]


@pytest.mark.asyncio
async def test_run_publishes_the_terminal_after_the_block_exits() -> None:
    agent = Agent("analyst", config=TestConfig("done"))

    async with agent.run("analyse", capability="analysis") as run:
        reply = await run.result()
        assert [type(event) for event in task_events(await reply.history.get_events())] == [TaskStarted]

    assert [type(event) for event in task_events(await reply.history.get_events())] == [TaskStarted, TaskCompleted]


@pytest.mark.asyncio
async def test_run_left_without_driving_publishes_nothing() -> None:
    stream = MemoryStream()
    agent = Agent("analyst", config=TestConfig("unused"))

    async with agent.run("analyse", capability="analysis", stream=stream):
        pass

    assert task_events(await stream.history.get_events()) == []


@pytest.mark.asyncio
async def test_run_started_and_abandoned_is_cancelled_before_calling_the_model() -> None:
    stream = MemoryStream()
    tracking = TrackingConfig(TestConfig("unused"))
    agent = Agent("analyst", config=tracking)

    async with agent.run("analyse", capability="analysis", stream=stream) as run:
        run.start()

    assert tracking.calls == []
    assert [type(event) for event in task_events(await stream.history.get_events())] == [TaskStarted, TaskCancelled]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["ask", "run"])
async def test_failing_tool_fails_the_task_and_raises(entry: str) -> None:
    stream = MemoryStream()
    agent = Agent("analyst", config=TestConfig(ToolCallEvent(name="fail")), tools=[fail])

    with pytest.raises(ValueError, match="work failed"):
        if entry == "ask":
            await agent.ask("analyse", capability="analysis", stream=stream)
        else:
            async with agent.run("analyse", capability="analysis", stream=stream) as run:
                await run.result()

    started, failed = task_events(await stream.history.get_events())
    assert started == IsInstance(TaskStarted)
    assert failed == IsInstance(TaskFailed) & HasAttributes(task_id=started.task_id, error=IsInstance(ValueError))


@pytest.mark.asyncio
async def test_missing_model_config_fails_the_task() -> None:
    stream = MemoryStream()

    with pytest.raises(ConfigNotProvidedError):
        await Agent("analyst").ask("analyse", capability="analysis", stream=stream)

    assert [type(event) for event in task_events(await stream.history.get_events())] == [TaskStarted, TaskFailed]


@pytest.mark.asyncio
async def test_cancelled_ask_is_reported_after_tool_cleanup() -> None:
    stream = MemoryStream()
    tool = SlowTool()
    agent = Agent("analyst", config=TestConfig(ToolCallEvent(name="wait")), tools=[tool.wait])

    driver = asyncio.create_task(agent.ask("analyse", capability="analysis", stream=stream))
    await asyncio.wait_for(tool.entered.wait(), 1)
    driver.cancel()

    with pytest.raises(asyncio.CancelledError):
        await driver
    assert tool.cleaned_up.is_set()
    assert [type(event) for event in task_events(await stream.history.get_events())] == [TaskStarted, TaskCancelled]


@pytest.mark.asyncio
async def test_run_left_while_a_tool_runs_is_cancelled_after_tool_cleanup() -> None:
    stream = MemoryStream()
    tool = SlowTool()
    agent = Agent("analyst", config=TestConfig(ToolCallEvent(name="wait")), tools=[tool.wait])

    async with agent.run("analyse", capability="analysis", stream=stream) as run:
        run.start()
        await asyncio.wait_for(tool.entered.wait(), 1)

    assert tool.cleaned_up.is_set()
    assert [type(event) for event in task_events(await stream.history.get_events())] == [TaskStarted, TaskCancelled]


@pytest.mark.asyncio
async def test_cancelled_standalone_task_is_reported_as_cancelled() -> None:
    ctx = Context(MemoryStream())
    tool = SlowTool()

    driver = asyncio.create_task(index_documents(ctx, tool))
    await asyncio.wait_for(tool.entered.wait(), 1)
    driver.cancel()

    with pytest.raises(asyncio.CancelledError):
        await driver
    assert [type(event) for event in task_events(await ctx.stream.history.get_events())] == [
        TaskStarted,
        TaskCancelled,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["reply.ask", "reply.run", "resume"])
async def test_each_tagged_continuation_is_a_new_task(entry: str) -> None:
    agent = Agent("analyst", config=TestConfig("first", "second", "third"))
    first = await agent.ask("first", capability="analysis")

    if entry == "reply.ask":
        second = await first.ask("second", capability="followup")
    elif entry == "reply.run":
        async with first.run("second", capability="followup") as run:
            second = await run.result()
    else:
        second = await agent.resume(
            *await first.history.get_events(),
            ModelRequest.ensure_request(["second"]),
            stream=first.context.stream,
            capability="followup",
        )
    await second.ask("third")

    starts = [event for event in await second.history.get_events() if isinstance(event, TaskStarted)]
    assert [event.spec.capability for event in starts] == ["analysis", "followup"]
    assert starts[0].task_id != starts[1].task_id


@pytest.mark.asyncio
async def test_tagged_run_keeps_the_callers_task_visible_to_tools() -> None:
    ctx = Context(MemoryStream())
    agent = Agent(
        "analyst",
        config=TestConfig(ToolCallEvent(name="report_progress"), "done"),
        tools=[report_progress],
    )

    async with Task(owner_id="pipeline", spec=TaskSpec(title="pipeline"), context=ctx) as task:
        await agent.ask("analyse", stream=ctx.stream, dependencies=ctx.dependencies, capability="analysis")

    assert task.metadata.progress == {"from_tool": True}
    assert task.state == TaskState.COMPLETED


@pytest.mark.asyncio
async def test_delegation_reports_the_workers_task_to_the_parent() -> None:
    worker = Agent("worker", config=TestConfig(ModelResponse(ModelMessage("summary"), usage=Usage(prompt_tokens=3))))
    lead = delegating_lead(worker, "done")

    reply = await lead.ask("handle the report")

    started, completed = task_events(await reply.history.get_events())
    assert started == IsInstance(TaskStarted) & HasAttributes(
        agent_name="worker",
        objective="summarise the report",
        parent_tool_call_id="call-1",
    )
    assert completed == IsInstance(TaskCompleted) & HasAttributes(
        task_id=started.task_id,
        result="summary",
        usage=HasAttributes(prompt_tokens=3),
    )
    assert (await reply.usage()).total.prompt_tokens == 3


@pytest.mark.asyncio
async def test_tagged_parent_terminal_includes_the_workers_usage() -> None:
    worker = Agent("worker", config=TestConfig(ModelResponse(ModelMessage("summary"), usage=Usage(prompt_tokens=3))))
    lead = delegating_lead(worker, "done")

    reply = await lead.ask("handle the report", capability="triage")

    completed = [event for event in await reply.history.get_events() if isinstance(event, TaskCompleted)]
    assert completed == [
        HasAttributes(agent_name="worker", usage=HasAttributes(prompt_tokens=3)),
        HasAttributes(agent_name="lead", usage=HasAttributes(prompt_tokens=3)),
    ]


@pytest.mark.asyncio
async def test_failed_delegation_is_reported_and_the_parent_continues() -> None:
    worker = Agent("worker", config=TestConfig(ToolCallEvent(name="fail")), tools=[fail])
    lead = delegating_lead(worker, "recovered")

    reply = await lead.ask("handle the report")

    assert reply.body == "recovered"
    started, failed = task_events(await reply.history.get_events())
    assert failed == IsInstance(TaskFailed) & HasAttributes(task_id=started.task_id, error=IsInstance(ValueError))


@pytest.mark.asyncio
async def test_finished_worker_cannot_be_messaged_from_its_terminal() -> None:
    stream = MemoryStream()
    probe = TerminalProbe()
    stream.where(TaskCompleted).subscribe(probe.check, sync_to_thread=False)
    lead = delegating_lead(Agent("worker", config=TestConfig("summary")), "done")

    await lead.ask("handle the report", stream=stream)

    assert probe.outcomes == ["retired"]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["ask", "run", "reply.ask", "reply.run"])
async def test_observer_cleanup_failure_is_what_the_caller_and_observers_see(entry: str) -> None:
    error = ValueError("cleanup failed")
    stream = MemoryStream()
    agent = Agent("analyst", config=TestConfig("initial", "done"))
    options = {"capability": "analysis", "observers": [FailingCleanup(error)]}

    with pytest.raises(ValueError) as raised:
        if entry == "ask":
            await agent.ask("analyse", stream=stream, **options)
        elif entry == "run":
            async with agent.run("analyse", stream=stream, **options) as run:
                await run.result()
        else:
            previous = await agent.ask("initial", stream=stream)
            if entry == "reply.ask":
                await previous.ask("analyse", **options)
            else:
                async with previous.run("analyse", **options) as run:
                    await run.result()

    assert raised.value is error
    started, failed = task_events(await stream.history.get_events())
    assert failed == IsInstance(TaskFailed) & HasAttributes(task_id=started.task_id, error=error)


@pytest.mark.asyncio
async def test_failing_start_observer_fails_the_task_and_frees_the_stream() -> None:
    stream = MemoryStream()
    subscription = stream.where(TaskStarted).subscribe(fail_on_start, sync_to_thread=False)
    agent = Agent("analyst", config=TestConfig("done"))

    with pytest.raises(ValueError, match="startup observer failed"):
        await agent.ask("analyse", capability="analysis", stream=stream)
    stream.unsubscribe(subscription)

    assert [type(event) for event in task_events(await stream.history.get_events())] == [TaskStarted, TaskFailed]
    assert (await asyncio.wait_for(agent.ask("next", stream=stream), 1)).body == "done"


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_cancellation_while_the_store_reads_usage_is_still_reported() -> None:
    storage = StallingStorage()
    stream = MemoryStream(storage=storage)
    agent = Agent("analyst", config=TestConfig("done"))

    driver = asyncio.create_task(agent.ask("analyse", capability="analysis", stream=stream))
    await asyncio.wait_for(storage.stalled.wait(), 1)
    driver.cancel()

    with pytest.raises(asyncio.CancelledError):
        await driver
    assert [type(event) for event in task_events(await stream.history.get_events())] == [TaskStarted, TaskCancelled]


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize("storage", [UsageWritesFail, LosesThenFailsUsage])
async def test_unrecordable_worker_usage_fails_the_delegation_and_the_parent_continues(
    storage: type[MemoryStorage],
) -> None:
    stream = MemoryStream(storage=storage())
    worker = Agent("worker", config=TestConfig(ModelResponse(ModelMessage("summary"), usage=Usage(prompt_tokens=3))))
    lead = delegating_lead(worker, "recovered")

    reply = await asyncio.wait_for(lead.ask("handle the report", stream=stream), 1)

    assert reply.body == "recovered"
    started, failed = task_events(await stream.history.get_events())
    assert failed == IsInstance(TaskFailed) & HasAttributes(task_id=started.task_id, error=IsInstance(ConnectionError))


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_interrupt_raised_by_a_tool_fails_the_task_and_frees_the_stream() -> None:
    stream = MemoryStream()
    agent = Agent("analyst", config=TestConfig(ToolCallEvent(name="abort")), tools=[abort])

    with pytest.raises(Abort):
        async with agent.run("analyse", capability="analysis", stream=stream) as run:
            await run.result()

    assert [type(event) for event in task_events(await stream.history.get_events())] == [TaskStarted, TaskFailed]
    assert (
        await asyncio.wait_for(Agent("next", config=TestConfig("done")).ask("next", stream=stream), 1)
    ).body == "done"
