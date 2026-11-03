# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from datetime import datetime
from typing import Any

import pytest

from ag2 import Agent, AgentReply, Context
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
    TaskStarted,
    ToolCallEvent,
    Usage,
    UsageEvent,
)
from ag2.events.task_events import TaskEvent
from ag2.exceptions import ConfigNotProvidedError
from ag2.observers import observer
from ag2.stream import MemoryStream
from ag2.task import Task, TaskSpec, TaskState
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.subagents import DeliveryError
from ag2.tools.subagents.run_task import run_task
from ag2.usage import UsageReport


class Journal:
    def __init__(self) -> None:
        self.events: list[BaseEvent] = []
        self.completed = asyncio.Event()

    def record(self, event: BaseEvent) -> None:
        self.events.append(event)
        if isinstance(event, TaskCompleted):
            self.completed.set()


class Work:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cleaned = asyncio.Event()

    async def wait(self) -> str:
        """Wait until the invocation is cancelled."""
        self.entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cleaned.set()
        return "unreachable"


async def fail() -> str:
    """Fail the model turn."""
    raise ValueError("work failed")


async def expire_on_start(event: TaskStarted, ctx: Context) -> None:
    await ctx.send(TaskExpired(task_id=event.task_id, agent_name=event.agent_name, objective=event.objective))


async def issue_progress(ctx: Context) -> str:
    """Check that model runs do not replace an enclosing standalone task."""
    task = ctx.dependencies.get("ag2.task")
    if task is not None:
        await task.progress({"from_tool": True})
    return "ok"


def lifecycle(events: list[BaseEvent]) -> list[TaskEvent]:
    return [event for event in events if isinstance(event, TaskEvent)]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["ask", "run"])
@pytest.mark.parametrize("options", [{}, {"capability": "analysis"}, {"ttl_seconds": 60}, {"capability": ""}])
async def test_lifecycle_is_opt_in_and_metadata_matches(entry: str, options: dict) -> None:
    journal = Journal()
    tracking = TrackingConfig(TestConfig("done"))
    agent = Agent("analyst", config=tracking)
    stream = MemoryStream()
    stream.where(TaskEvent).subscribe(journal.record, sync_to_thread=False)
    if entry == "ask":
        reply = await agent.ask("analyse", "the data", stream=stream, **options)
    else:
        async with agent.run("analyse", "the data", stream=stream, **options) as run:
            assert journal.events == []
            reply = await run.result()
            assert await run.result() is reply
            assert not journal.completed.is_set()
    events = lifecycle(journal.events)
    if not options:
        assert events == []
    else:
        started, completed = events
        assert isinstance(started, TaskStarted)
        assert started.agent_name == "analyst"
        assert started.objective == "analyse\nthe data"
        assert started.spec == TaskSpec(title="analyse\nthe data", capability=options.get("capability"))
        assert (started.expires_at is not None) == ("ttl_seconds" in options)
        assert isinstance(completed, TaskCompleted)
        assert completed.task_id == started.task_id
        assert completed.task_stream == reply.context.stream.id
        assert completed.result == "done"
    [call] = tracking.calls
    assert not any(isinstance(event, TaskEvent) for event in call.messages)
    assert "ag2.task" not in reply.context.dependencies


@pytest.mark.asyncio
async def test_undriven_run_has_no_lifecycle() -> None:
    journal = Journal()
    stream = MemoryStream()
    stream.subscribe(journal.record)
    agent = Agent("runner", config=TestConfig("unused"))
    async with agent.run("work", capability="analysis", stream=stream):
        pass
    assert lifecycle(journal.events) == []


@pytest.mark.asyncio
async def test_started_run_emits_terminal_after_scope_exit() -> None:
    journal = Journal()
    agent = Agent("runner", config=TestConfig("done"))
    async with agent.run("work", capability="analysis") as run:
        run.stream.subscribe(journal.record)
        run.start()
        assert (await run.result()).body == "done"
        assert not journal.completed.is_set()
    assert journal.completed.is_set()
    assert [type(event) for event in lifecycle(journal.events)] == [TaskStarted, TaskCompleted]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["ask", "run", "delegation"])
async def test_model_failure_has_one_terminal(entry: str) -> None:
    stream = MemoryStream()
    agent = Agent("runner", config=TestConfig(ToolCallEvent(name="fail", arguments="{}")), tools=[fail])
    if entry == "delegation":
        result = await run_task(agent, "work", parent_context=Context(stream), task_id="assigned")
        assert not result.completed
        assert isinstance(result.error, ValueError)
    else:
        with pytest.raises(ValueError, match="work failed"):
            if entry == "ask":
                await agent.ask("work", capability="analysis", stream=stream)
            else:
                async with agent.run("work", capability="analysis", stream=stream) as run:
                    await run.result()
    events = lifecycle(await stream.history.get_events())
    assert [type(event) for event in events] == [TaskStarted, TaskFailed]
    assert events[0].task_id == events[1].task_id


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["ask", "run", "delegation"])
async def test_setup_failure_has_one_terminal(entry: str) -> None:
    stream = MemoryStream()
    agent = Agent("runner")
    if entry == "delegation":
        result = await run_task(agent, "work", parent_context=Context(stream))
        assert isinstance(result.error, ConfigNotProvidedError)
    else:
        with pytest.raises(ConfigNotProvidedError):
            if entry == "ask":
                await agent.ask("work", capability="analysis", stream=stream)
            else:
                async with agent.run("work", capability="analysis", stream=stream):
                    pass
    assert [type(event) for event in lifecycle(await stream.history.get_events())] == [TaskStarted, TaskFailed]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["ask", "run", "delegation", "standalone"])
async def test_cancellation_is_typed_after_work_cleanup(entry: str) -> None:
    work = Work()
    stream = MemoryStream()
    agent = Agent("runner", config=TestConfig(ToolCallEvent(name="wait", arguments="{}")), tools=[work.wait])
    if entry == "standalone":
        driver = asyncio.create_task(standalone_wait(work, stream))
    elif entry == "delegation":
        driver = asyncio.create_task(run_task(agent, "work", parent_context=Context(stream)))
    elif entry == "run":
        async with agent.run("work", capability="analysis", stream=stream) as run:
            run.start()
            await asyncio.wait_for(work.entered.wait(), 1)
        assert work.cleaned.is_set()
        assert [type(event) for event in lifecycle(await stream.history.get_events())] == [TaskStarted, TaskCancelled]
        return
    else:
        driver = asyncio.create_task(agent.ask("work", capability="analysis", stream=stream))
    await asyncio.wait_for(work.entered.wait(), 1)
    driver.cancel()
    with pytest.raises(asyncio.CancelledError):
        await driver
    assert work.cleaned.is_set()
    assert [type(event) for event in lifecycle(await stream.history.get_events())] == [TaskStarted, TaskCancelled]


async def standalone_wait(work: Work, stream: MemoryStream) -> None:
    async with Task(owner_id="runner", spec=TaskSpec(title="work"), context=Context(stream)):
        await work.wait()


@pytest.mark.asyncio
async def test_external_expiry_does_not_produce_a_second_terminal() -> None:
    stream = MemoryStream()
    stream.where(TaskStarted).subscribe(expire_on_start, sync_to_thread=False)
    agent = Agent("runner", config=TestConfig("done"))
    reply = await agent.ask("work", stream=stream, ttl_seconds=1)
    assert reply.body == "done"  # TTL is observation metadata, not a local execution timeout.
    events = lifecycle(await stream.history.get_events())
    assert sorted(type(event).__name__ for event in events) == ["TaskExpired", "TaskStarted"]
    assert events[0].task_id == events[1].task_id


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["ask", "run", "resume"])
async def test_continuations_have_fresh_opt_in_identity(entry: str) -> None:
    agent = Agent("runner", config=TestConfig("first", "second", "third"))
    first = await agent.ask("first", capability="analysis")
    stream = first.context.stream
    if entry == "ask":
        second = await first.ask("second", capability="followup")
    elif entry == "run":
        async with first.run("second", capability="followup") as run:
            second = await run.result()
    else:
        second = await agent.resume(
            *await stream.history.get_events(),
            ModelRequest.ensure_request(["second"]),
            stream=stream,
            capability="followup",
        )
    events = lifecycle(await second.context.stream.history.get_events())
    starts = [event for event in events if isinstance(event, TaskStarted)]
    assert len(starts) == 2
    assert starts[0].task_id != starts[1].task_id
    assert [event.spec.capability for event in starts] == ["analysis", "followup"]
    await second.ask("third")
    assert len(lifecycle(await stream.history.get_events())) == len(events)


@pytest.mark.asyncio
async def test_run_keeps_enclosing_task_injection() -> None:
    ctx = Context(MemoryStream())
    agent = Agent(
        "runner",
        config=TestConfig(ToolCallEvent(name="issue_progress", arguments="{}"), "done"),
        tools=[issue_progress],
    )
    async with Task(owner_id="owner", spec=TaskSpec(title="pipeline"), context=ctx) as task:
        reply = await agent.ask("work", dependencies=ctx.dependencies, stream=ctx.stream, capability="analysis")
        assert reply.context.dependencies["ag2.task"] is task
        assert task.metadata.progress == {"from_tool": True}
    assert task.state == TaskState.COMPLETED


@pytest.mark.asyncio
async def test_root_and_delegation_each_have_one_lifecycle_and_source_usage() -> None:
    child = Agent("worker", config=TestConfig(ModelResponse(ModelMessage("result"), usage=Usage(prompt_tokens=3))))
    root = Agent(
        "root",
        config=TestConfig(ToolCallEvent(name="task_worker", arguments='{"objective":"work"}'), "done"),
        tools=[child.as_tool(description="Work")],
    )
    reply = await root.ask("delegate", capability="analysis")
    events = await reply.history.get_events()
    starts = [event for event in events if isinstance(event, TaskStarted)]
    terminals = [event for event in events if isinstance(event, TaskCompleted)]
    assert sorted(event.agent_name for event in starts) == ["root", "worker"]
    assert sorted(event.agent_name for event in terminals) == ["root", "worker"]
    child_start = next(event for event in starts if event.agent_name == "worker")
    child_terminal = next(event for event in terminals if event.agent_name == "worker")
    assert child_start.path == child_terminal.path == (child_start.task_id,)
    usage = [event for event in events if isinstance(event, UsageEvent)]
    assert len(usage) == 1
    assert usage[0].path == (child_start.task_id,)
    assert (await reply.usage()).total.prompt_tokens == 3
    assert child_terminal.usage.prompt_tokens == 3
    assert next(event for event in terminals if event.agent_name == "root").usage.prompt_tokens == 3


@pytest.mark.asyncio
async def test_ttl_is_carried_on_start_without_a_local_timer() -> None:
    agent = Agent("runner", config=TestConfig("done"))
    reply = await agent.ask("work", ttl_seconds=60)
    started = next(event for event in await reply.history.get_events() if isinstance(event, TaskStarted))
    assert started.expires_at is not None
    datetime.fromisoformat(started.expires_at)


class TaggedAgent(Agent):
    async def ask(self, *msg: Any, **kwargs: Any) -> AgentReply:
        return await super().ask(*msg, capability="analysis", ttl_seconds=60, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("emit_events", [True, False])
async def test_delegation_opt_in_uses_its_identity_and_respects_emission_switch(emit_events: bool) -> None:
    parent = Context(MemoryStream())
    child = TaggedAgent("worker", config=TestConfig(ModelResponse(ModelMessage("done"), usage=Usage(prompt_tokens=3))))
    result = await run_task(child, "work", parent_context=parent, emit_events=emit_events, task_id="delegated")
    assert result.completed
    events = await parent.stream.history.get_events()
    tasks = lifecycle(events)
    if emit_events:
        started, completed = tasks
        assert isinstance(started, TaskStarted)
        assert started.spec == TaskSpec(title="work", capability="analysis")
        assert started.expires_at is not None
        assert isinstance(completed, TaskCompleted)
        assert started.task_id == completed.task_id == result.task_id == "delegated"
    else:
        assert tasks == []
    [usage] = [event for event in events if isinstance(event, UsageEvent)]
    assert usage.task_id == "delegated"
    assert usage.usage.prompt_tokens == 3


class BackgroundWork:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.model_finished = asyncio.Event()

    async def wait(self) -> str:
        """Wait for the caller to release the worker."""
        self.entered.set()
        await self.release.wait()
        return "child done"

    def observe(self, event: ModelResponse) -> None:
        if event.content == "root done":
            self.model_finished.set()

    async def delegate(self, ctx: Context) -> str:
        """Start a worker that finishes after the parent's model."""
        child = Agent(
            "worker", config=TestConfig(ToolCallEvent(name="wait", arguments="{}"), "child done"), tools=[self.wait]
        )
        ctx.stream.spawn_background(self.run_child(child, ctx))
        await self.entered.wait()
        return "started"

    async def run_child(self, child: Agent, ctx: Context) -> None:
        await run_task(child, "work", parent_context=ctx)


@pytest.mark.asyncio
async def test_root_terminal_waits_for_background_descendant() -> None:
    work = BackgroundWork()
    stream = MemoryStream()
    stream.where(ModelResponse).subscribe(work.observe, sync_to_thread=False)
    root = Agent(
        "root", config=TestConfig(ToolCallEvent(name="delegate", arguments="{}"), "root done"), tools=[work.delegate]
    )
    driver = asyncio.create_task(root.ask("work", stream=stream, capability="analysis"))
    try:
        await asyncio.wait_for(work.model_finished.wait(), 1)
        assert not driver.done()
    finally:
        work.release.set()
    await asyncio.wait_for(driver, 1)
    terminals = [event.agent_name for event in await stream.history.get_events() if isinstance(event, TaskCompleted)]
    assert terminals == ["worker", "root"]


class RememberingAgent(Agent):
    def __init__(self) -> None:
        super().__init__("worker", config=TestConfig("first", "second"))
        self.replies: list[AgentReply] = []

    async def ask(self, *msg: Any, **kwargs: Any) -> AgentReply:
        reply = await super().ask(*msg, **kwargs)
        self.replies.append(reply)
        return reply


@pytest.mark.asyncio
async def test_continuation_of_finished_delegation_has_new_identity_and_no_stale_origin() -> None:
    parent = Context(MemoryStream())
    child = RememberingAgent()
    result = await run_task(child, "first", parent_context=parent, task_id="original")
    assert result.completed
    [first] = child.replies
    second = await first.ask("second", capability="analysis")
    events = lifecycle(await second.history.get_events())
    assert [type(event) for event in events] == [TaskStarted, TaskCompleted, TaskStarted, TaskCompleted]
    assert events[0].task_id == events[1].task_id == "original"
    assert events[2].task_id == events[3].task_id != "original"
    assert len(lifecycle(await parent.stream.history.get_events())) == 2


def fail_start(event: TaskStarted) -> None:
    raise ValueError("startup observer failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["ask", "standalone"])
async def test_start_failure_has_one_terminal_and_cleans_up_injection(entry: str) -> None:
    ctx = Context(MemoryStream())
    ctx.stream.where(TaskStarted).subscribe(fail_start, sync_to_thread=False)
    with pytest.raises(ValueError, match="startup observer failed"):
        if entry == "standalone":
            async with Task(owner_id="owner", spec=TaskSpec(title="work"), context=ctx):
                pass
        else:
            await Agent("runner", config=TestConfig("unused")).ask("work", capability="analysis", stream=ctx.stream)
    assert [type(event) for event in lifecycle(await ctx.stream.history.get_events())] == [TaskStarted, TaskFailed]
    assert "ag2.task" not in ctx.dependencies


@pytest.mark.asyncio
async def test_start_then_immediate_exit_emits_cancellation_without_calling_model() -> None:
    stream = MemoryStream()
    tracking = TrackingConfig(TestConfig("unused"))
    agent = Agent("runner", config=tracking)
    async with agent.run("work", capability="analysis", stream=stream) as run:
        run.start()
    assert tracking.calls == []
    assert [type(event) for event in lifecycle(await stream.history.get_events())] == [TaskStarted, TaskCancelled]


@pytest.mark.asyncio
async def test_rejected_start_observer_failure_releases_the_stream() -> None:
    stream = MemoryStream()
    subscription = stream.where(TaskStarted).subscribe(fail_start, sync_to_thread=False)
    agent = Agent("runner", config=TestConfig("done"))
    with pytest.raises(ValueError, match="startup observer failed"):
        async with agent.run("work", capability="analysis", stream=stream) as run:
            run.start()
    stream.unsubscribe(subscription)
    reply = await asyncio.wait_for(agent.ask("next", stream=stream), 1)
    assert reply.body == "done"


class TerminalAddress:
    def __init__(self) -> None:
        self.checked: list[str] = []

    async def check(self, event: TaskEvent) -> None:
        with pytest.raises(DeliveryError):
            await event.origin.send(HaltEvent(reason="already finished", source="test"))
        self.checked.append(event.task_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_delegation_address_is_retired_before_terminal_observers(fails: bool) -> None:
    parent = Context(MemoryStream())
    address = TerminalAddress()
    parent.stream.where(TaskCompleted | TaskFailed).subscribe(address.check, sync_to_thread=False)
    child = Agent(
        "worker", config=TestConfig(ToolCallEvent(name="fail", arguments="{}") if fails else "done"), tools=[fail]
    )
    result = await run_task(child, "work", parent_context=parent, task_id="worker")
    assert result.completed == (not fails)
    assert address.checked == ["worker"]


class FailingCleanup:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.cleaned = False

    def register(self, stack: ExitStack, context: Context) -> None:
        stack.callback(self.close)

    def close(self) -> None:
        self.cleaned = True
        raise self.error


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["ask", "run", "reply_ask", "reply_run", "delegation"])
@pytest.mark.parametrize("cancel_cleanup", [False, True])
async def test_cleanup_failure_matches_the_returned_outcome(entry: str, cancel_cleanup: bool) -> None:
    error = asyncio.CancelledError("cleanup cancelled") if cancel_cleanup else ValueError("cleanup failed")
    cleanup = FailingCleanup(error)
    stream = MemoryStream()
    options = {"stream": stream, "capability": "analysis", "observers": [cleanup]}
    agent = Agent("runner", config=TestConfig("initial", "done"))
    if entry == "delegation":
        agent = Agent("worker", config=TestConfig("done"), observers=[cleanup])
        if cancel_cleanup:
            with pytest.raises(asyncio.CancelledError):
                await run_task(agent, "work", parent_context=Context(stream))
        else:
            result = await run_task(agent, "work", parent_context=Context(stream))
            assert not result.completed
            assert result.error is error
    else:
        with pytest.raises(type(error)) as exc:
            if entry == "ask":
                await agent.ask("work", **options)
            elif entry == "run":
                async with agent.run("work", **options) as run:
                    await run.result()
            else:
                previous = await agent.ask("initial", stream=stream)
                options.pop("stream")
                if entry == "reply_ask":
                    await previous.ask("work", **options)
                else:
                    async with previous.run("work", **options) as run:
                        await run.result()
        assert exc.value is error
    assert cleanup.cleaned
    events = lifecycle(await stream.history.get_events())
    assert [type(event) for event in events] == [TaskStarted, TaskCancelled if cancel_cleanup else TaskFailed]
    if not cancel_cleanup:
        assert events[-1].error is error


@pytest.mark.asyncio
@pytest.mark.parametrize("driver_cancel", [False, True])
@pytest.mark.parametrize("cleanup_cancel", [False, True])
async def test_cleanup_exception_overrides_the_driver_failure(driver_cancel: bool, cleanup_cancel: bool) -> None:
    failure = asyncio.CancelledError("model cancelled") if driver_cancel else RuntimeError("model failed")
    error = asyncio.CancelledError("cleanup cancelled") if cleanup_cancel else ValueError("cleanup failed")
    stream = MemoryStream()
    agent = Agent("runner", config=TestConfig(failure), observers=[FailingCleanup(error)])
    with pytest.raises(type(error)) as exc:
        await agent.ask("work", stream=stream, capability="analysis")
    assert exc.value is error
    events = lifecycle(await stream.history.get_events())
    assert [type(event) for event in events] == [TaskStarted, TaskCancelled if cleanup_cancel else TaskFailed]
    if not cleanup_cancel:
        assert events[-1].error is error


async def cancel_on_start(event: TaskStarted, ctx: Context) -> None:
    await ctx.send(TaskCancelled(task_id=event.task_id, agent_name=event.agent_name, objective=event.objective))


async def fail_on_start(event: TaskStarted, ctx: Context) -> None:
    await ctx.send(
        TaskFailed(
            task_id=event.task_id,
            agent_name=event.agent_name,
            objective=event.objective,
            error=ValueError("externally failed"),
        )
    )


async def complete_on_start(event: TaskStarted, ctx: Context) -> None:
    await ctx.send(
        TaskCompleted(
            task_id=event.task_id,
            agent_name=event.agent_name,
            objective=event.objective,
            task_stream=ctx.stream.id,
        )
    )


async def cancel_usage_delivery(event: UsageEvent) -> None:
    raise asyncio.CancelledError("usage delivery interrupted")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal",
    [expire_on_start, cancel_on_start, fail_on_start, complete_on_start],
    ids=["expired", "cancelled", "failed", "completed"],
)
async def test_external_terminal_does_not_disable_source_accounting_retention(
    terminal: Callable[[TaskStarted, Context], Awaitable[None]],
) -> None:
    parent = Context(MemoryStream())
    parent.stream.where(UsageEvent).subscribe(cancel_usage_delivery, sync_to_thread=False)
    source = MemoryStream(storage=parent.stream.history.storage)
    child = Agent(
        "worker",
        config=TestConfig(ModelResponse(ModelMessage("done"), usage=Usage(total_tokens=30))),
        observers=[observer(TaskStarted, callback=terminal)],
    )
    with pytest.raises(asyncio.CancelledError):
        await run_task(child, "work", parent_context=parent, stream=source)
    source_events = await source.history.get_events()
    parent_events = await parent.stream.history.get_events()
    [source_usage] = [event for event in source_events if isinstance(event, UsageEvent)]
    [parent_usage] = [event for event in parent_events if isinstance(event, UsageEvent)]
    assert source_usage.event_id == parent_usage.event_id
    assert UsageReport.from_events(source_events).total.total_tokens == 30
    assert UsageReport.from_events(parent_events).total.total_tokens == 30
    assert len(lifecycle(source_events)) == len(lifecycle(parent_events)) == 2
