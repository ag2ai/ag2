# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from dirty_equals import IsPartialDict

from ag2 import Agent, Context, KnowledgeConfig, MemoryStream, Scope, ScopedValue, observer
from ag2.compact import CompactTrigger, TailWindowCompact
from ag2.events import (
    BaseEvent,
    ChildEvent,
    DeliveryFailed,
    DownEvent,
    HaltEvent,
    HumanInputRequest,
    HumanMessage,
    ModelMessage,
    ModelMessageChunk,
    ModelResponse,
    TaskCancelled,
    TaskStarted,
    ToolCallEvent,
    Usage,
    UsageEvent,
)
from ag2.knowledge import MemoryKnowledgeStore
from ag2.testing import TestConfig
from ag2.tools.subagents import DeliveryError, background_agent_tool
from ag2.tools.subagents.run_task import run_task
from ag2.usage import UsageReport


@dataclass
class Journal:
    events: list[BaseEvent] = field(default_factory=list)

    async def record(self, event: BaseEvent) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_root_observes_original_leaf_and_counts_each_call_once() -> None:
    leaf = Agent(
        "leaf",
        config=TestConfig(
            ModelMessageChunk("working"),
            ModelResponse(ModelMessage("leaf done"), usage=Usage(total_tokens=30), model="leaf-model"),
        ),
    )
    worker = Agent(
        "worker",
        config=TestConfig(
            ToolCallEvent(name="task_leaf", arguments='{"objective": "dig"}'),
            ModelResponse(ModelMessage("worker done"), usage=Usage(total_tokens=20), model="worker-model"),
        ),
        tools=[leaf.as_tool(description="Dig")],
    )
    root = Agent(
        "root",
        config=TestConfig(
            ToolCallEvent(name="task_worker", arguments='{"objective": "organize"}'),
            ModelResponse(ModelMessage("root done"), usage=Usage(total_tokens=10), model="root-model"),
        ),
        tools=[worker.as_tool(description="Organize")],
    )
    journal = Journal()
    reply = await root.ask("start", stream=MemoryStream(), observers=[observer(callback=journal.record)])

    chunks = [e for e in journal.events if isinstance(e, ChildEvent) and isinstance(e.inner, ModelMessageChunk)]
    assert len(chunks) == 1
    assert chunks[0].agent_name == "leaf"
    assert len(chunks[0].path) == 2
    assert chunks[0].origin.agent_name == "leaf"
    history = list(await reply.history.get_events())
    assert not any(isinstance(e, ChildEvent) and isinstance(e.inner, ModelMessageChunk) for e in history)
    assert not any(isinstance(e, UsageEvent) and e.kind == "subtask" for e in history)
    report = await reply.usage()
    assert report.total.total_tokens == 60
    assert {model: usage.total_tokens for model, usage in report.by_model.items()} == {
        "leaf-model": 30,
        "worker-model": 20,
        "root-model": 10,
    }


class WorkSignal(BaseEvent):
    __conversational__ = False


class Directive(BaseEvent):
    __conversational__ = False
    text: str


@dataclass
class Guidance:
    origins: list[Any] = field(default_factory=list)

    async def monitor(self, event: BaseEvent) -> None:
        if isinstance(event, ChildEvent) and isinstance(event.inner, WorkSignal):
            self.origins.append(event.origin)
            await event.origin.send(Directive(text="keep digging"))

    async def receive(self, event: BaseEvent, ctx: Context) -> None:
        if isinstance(event, Directive):
            ctx.variables["guidance"] = event.text


async def seek_guidance(ctx: Context) -> str:
    """Receive guidance during this tool's execution."""
    await ctx.send(WorkSignal())
    return ctx.variables["guidance"]


@pytest.mark.asyncio
async def test_origin_addresses_the_leaf_without_echo_and_expires_after_completion() -> None:
    guidance, journal = Guidance(), Journal()
    leaf = Agent(
        "leaf",
        config=TestConfig(ToolCallEvent(name="seek_guidance", arguments="{}"), "done"),
        tools=[seek_guidance],
        observers=[observer(callback=guidance.receive)],
    )
    worker = Agent(
        "worker",
        config=TestConfig(ToolCallEvent(name="task_leaf", arguments='{"objective":"dig"}'), "done"),
        tools=[leaf.as_tool(description="Dig")],
    )
    root = Agent(
        "root",
        config=TestConfig(ToolCallEvent(name="task_worker", arguments='{"objective":"organize"}'), "done"),
        tools=[worker.as_tool(description="Organize")],
    )
    stream = MemoryStream()
    stream.subscribe(journal.record)
    await root.ask("start", stream=stream, observers=[observer(callback=guidance.monitor)])
    assert [origin.agent_name for origin in guidance.origins] == ["leaf"]
    assert not any(isinstance(e, ChildEvent) and isinstance(e.inner, Directive) for e in journal.events)
    with pytest.raises(DeliveryError):
        await guidance.origins[0].send(Directive(text="too late"))
    assert any(
        isinstance(e, ChildEvent) and isinstance(e.inner, DeliveryFailed) or isinstance(e, DeliveryFailed)
        for e in journal.events
    )


async def long_work(ctx: Context) -> str:
    """A cancellable operation that must finish cleanup before its terminal."""
    try:
        await ctx.send(WorkSignal())
        await asyncio.Event().wait()
    finally:
        ctx.variables["cleanup"].append("cleaned")
    return "unreachable"


async def stop_on_signal(event: BaseEvent, ctx: Context) -> None:
    if isinstance(event, ChildEvent) and isinstance(event.inner, WorkSignal):
        sibling = ctx.dependencies["sibling_gate"]
        await sibling.entered.wait()
        try:
            await event.origin.send(HaltEvent(reason="budget", source="test"))
        finally:
            sibling.release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("same_name", [False, True])
async def test_targeted_halt_cancels_running_tool_without_aborting_parent_or_sibling(same_name: bool) -> None:
    child = Agent(
        "worker" if same_name else "child",
        config=TestConfig(ToolCallEvent(name="long_work", arguments="{}")),
        tools=[long_work],
    )
    sibling_gate = Gate()
    sibling = Agent(
        "worker" if same_name else "sibling",
        config=TestConfig(ToolCallEvent(name="run", arguments="{}"), "sibling done"),
        tools=[sibling_gate.run],
    )
    root = Agent(
        "root",
        config=TestConfig(
            [
                ToolCallEvent(name="cancel_work", arguments='{"objective":"work"}'),
                ToolCallEvent(name="keep_work", arguments='{"objective":"work"}'),
            ],
            "root survived",
        ),
        tools=[
            child.as_tool(description="Work", name="cancel_work"),
            sibling.as_tool(description="Work", name="keep_work"),
        ],
        dependencies={"sibling_gate": sibling_gate},
        variables={"cleanup": ScopedValue([], scope=Scope.SHARED)},
    )
    journal = Journal()
    reply = await asyncio.wait_for(
        root.ask("start", observers=[observer(callback=stop_on_signal), observer(callback=journal.record)]), 2
    )
    assert reply.body == "root survived"
    assert sibling_gate.received == [False]
    assert reply.context.variables["cleanup"] == ["cleaned"]
    cancelled = [e.inner if isinstance(e, ChildEvent) else e for e in journal.events]
    assert [e.agent_name for e in cancelled if isinstance(e, TaskCancelled)] == [child.name]


async def ask_human(ctx: Context) -> str:
    """Ask the owner of the human-input channel."""
    return await ctx.input("May I continue?", timeout=1)


def human_answer(event: HumanInputRequest) -> str:
    return "yes"


@pytest.mark.asyncio
@pytest.mark.parametrize("own_hook", [False, True])
async def test_root_audits_one_question_and_answer_from_leaf_with_either_hook_owner(own_hook: bool) -> None:
    leaf = Agent(
        "leaf",
        config=TestConfig(ToolCallEvent(name="ask_human", arguments="{}"), "done"),
        tools=[ask_human],
        hitl_hook=human_answer if own_hook else None,
    )
    worker = Agent(
        "worker",
        config=TestConfig(ToolCallEvent(name="task_leaf", arguments='{"objective":"ask"}'), "done"),
        tools=[leaf.as_tool(description="Ask")],
    )
    root = Agent(
        "root",
        config=TestConfig(ToolCallEvent(name="task_worker", arguments='{"objective":"ask"}'), "done"),
        tools=[worker.as_tool(description="Ask")],
        hitl_hook=human_answer,
    )
    journal = Journal()
    await root.ask("start", observers=[observer(callback=journal.record)])
    requests = [e.inner for e in journal.events if isinstance(e, ChildEvent) and isinstance(e.inner, HumanInputRequest)]
    answers = [e.inner for e in journal.events if isinstance(e, ChildEvent) and isinstance(e.inner, HumanMessage)]
    assert len(requests) == len(answers) == 1
    assert answers[0].parent_id == requests[0].id
    assert any(
        isinstance(e, ChildEvent) and isinstance(e.inner, TaskStarted) and e.agent_name == "leaf"
        for e in journal.events
    )


@dataclass
class Gate:
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    received: list[bool] = field(default_factory=list)

    async def run(self, ctx: Context) -> str:
        """Wait until the test releases this public tool."""
        self.entered.set()
        await self.release.wait()
        self.received.append("guidance" in ctx.variables)
        return "done"


@dataclass
class HaltCleanup:
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    cleaned: asyncio.Event = field(default_factory=asyncio.Event)

    async def work(self) -> str:
        """Finish cleanup after an externally delivered Halt."""
        self.entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cleaned.set()
        return "unreachable"

    async def observe(self, event: BaseEvent) -> None:
        if isinstance(event, HaltEvent):
            await self.cleaned.wait()


@pytest.mark.asyncio
async def test_successfully_delivered_halt_remains_successful_during_recipient_cleanup() -> None:
    cleanup, journal = HaltCleanup(), Journal()
    ctx = Context(MemoryStream())
    ctx.stream.subscribe(journal.record)
    child = Agent(
        "child",
        config=TestConfig(ToolCallEvent(name="work", arguments="{}")),
        tools=[cleanup.work],
        observers=[observer(callback=cleanup.observe), observer(callback=journal.record)],
    )
    task = asyncio.create_task(run_task(child, "work", parent_context=ctx, task_id="child"))
    await cleanup.entered.wait()
    handle = next(event.origin for event in journal.events if isinstance(event, ChildEvent))
    await handle.send(HaltEvent(reason="stop", source="test"))
    result = await task
    assert result.cancelled
    assert cleanup.cleaned.is_set()
    assert not any(isinstance(event, DeliveryFailed) for event in await ctx.stream.history.get_events())


@pytest.mark.asyncio
async def test_queued_delegation_has_no_origin_and_does_not_capture_active_turn_delivery() -> None:
    gate = Gate()
    guidance = Guidance()
    worker = Agent(
        "worker",
        config=TestConfig(ToolCallEvent(name="run", arguments="{}"), "done"),
        tools=[gate.run],
        observers=[observer(callback=guidance.receive)],
    )
    ctx = Context(MemoryStream())
    journal = Journal()
    ctx.stream.subscribe(journal.record)
    shared_stream = MemoryStream(storage=ctx.stream.history.storage)
    first = asyncio.create_task(run_task(worker, "first", parent_context=ctx, stream=shared_stream, task_id="first"))
    await gate.entered.wait()
    second = asyncio.create_task(run_task(worker, "second", parent_context=ctx, stream=shared_stream, task_id="second"))
    await asyncio.sleep(0)
    try:
        observations = [event for event in journal.events if isinstance(event, ChildEvent)]
        assert {event.task_id for event in observations} == {"first"}
        await observations[0].origin.send(Directive(text="first only"))
    finally:
        gate.release.set()
        results = await asyncio.gather(first, second)
    assert all(result.completed for result in results)
    assert gate.received == [True, False]


@pytest.mark.asyncio
async def test_descendant_accounting_survives_compaction_and_serialized_replay() -> None:
    child = Agent(
        "child", config=TestConfig(ModelResponse(ModelMessage("done"), usage=Usage(total_tokens=30), model="child"))
    )
    root = Agent(
        "root",
        config=TestConfig(ToolCallEvent(name="task_child", arguments='{"objective":"work"}'), "done"),
        tools=[child.as_tool(description="Work")],
        knowledge=KnowledgeConfig(
            store=MemoryKnowledgeStore(),
            compact=TailWindowCompact(target=1),
            compact_trigger=CompactTrigger(max_events=1),
            expose_tool=False,
            write_event_log=False,
        ),
    )
    reply = await root.ask("start")
    history = list(await reply.history.get_events())
    replayed = [type(event).from_dict(event.to_dict()) for event in history]
    assert UsageReport.from_events(replayed).total.total_tokens == 30
    assert UsageReport.from_events([*history, *replayed]).total.total_tokens == 30


@dataclass
class CancellationState:
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    context: Context | None = None
    signals: int = 0

    async def record(self, event: BaseEvent, ctx: Context) -> None:
        inner = event.inner if isinstance(event, ChildEvent) else event
        if isinstance(inner, WorkSignal):
            self.context = ctx
            self.signals += 1
            if self.signals == 2:
                self.ready.set()


@pytest.mark.asyncio
async def test_parent_cancellation_settles_background_descendants_before_returning() -> None:
    child = Agent("child", config=TestConfig(ToolCallEvent(name="long_work", arguments="{}")), tools=[long_work])
    root = Agent(
        "root",
        config=TestConfig([
            ToolCallEvent(name="background_task_child", arguments='{"objective":"work"}'),
            ToolCallEvent(name="long_work", arguments="{}"),
        ]),
        tools=[background_agent_tool(child, description="Work"), long_work],
        variables={"cleanup": ScopedValue([], scope=Scope.SHARED)},
    )
    state = CancellationState()
    turn = asyncio.create_task(root.ask("start", observers=[observer(callback=state.record)]))
    await asyncio.wait_for(state.ready.wait(), 2)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    assert state.context is not None
    assert state.context.variables["cleanup"] == ["cleaned", "cleaned"]


@pytest.mark.asyncio
async def test_republishing_a_usage_template_creates_distinct_source_publications() -> None:
    template = UsageEvent(Usage(total_tokens=10))
    child = Agent("child", config=TestConfig(template, "done"))
    root = Agent(
        "root",
        config=TestConfig(
            [
                ToolCallEvent(name="task_child", arguments='{"objective":"one"}'),
                ToolCallEvent(name="task_child", arguments='{"objective":"two"}'),
            ],
            "done",
        ),
        tools=[child.as_tool(description="Work")],
    )
    reply = await root.ask("start")
    assert (await reply.usage()).total.total_tokens == 20


@dataclass
class TurnRace:
    raw: bool = False
    first_entered: asyncio.Event = field(default_factory=asyncio.Event)
    second_entered: asyncio.Event = field(default_factory=asyncio.Event)
    release_first: asyncio.Event = field(default_factory=asyncio.Event)
    release_second: asyncio.Event = field(default_factory=asyncio.Event)

    async def hold(self, ctx: Context) -> str:
        """Hold each persistent turn at a public tool boundary."""
        if ctx.variables["number"] == 1:
            self.first_entered.set()
            await self.release_first.wait()
        else:
            self.second_entered.set()
            await self.release_second.wait()
        return "done"

    async def advance_during_audit(self, event: BaseEvent) -> BaseEvent:
        if (self.raw and isinstance(event, HaltEvent)) or (
            not self.raw and isinstance(event, DownEvent) and event.task_id == "first"
        ):
            self.release_first.set()
            await self.second_entered.wait()
        return event


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["sender", "recipient_audit", "recipient_event"])
async def test_downlink_revalidates_its_recipient_after_an_awaiting_audit(stage: str) -> None:
    race = TurnRace(raw=stage == "recipient_event")
    ctx = Context(MemoryStream(), variables={"number": 1})
    journal = Journal()
    ctx.stream.subscribe(journal.record)
    stream = MemoryStream(storage=ctx.stream.history.storage)
    if stage == "sender":
        ctx.stream.subscribe(race.advance_during_audit)
    else:
        stream.subscribe(race.advance_during_audit, interrupt=True)
    worker = Agent("worker", config=TestConfig(ToolCallEvent(name="hold", arguments="{}"), "done"), tools=[race.hold])
    first = asyncio.create_task(run_task(worker, "first", parent_context=ctx, stream=stream, task_id="first"))
    await race.first_entered.wait()
    handle = next(
        event.origin
        for event in journal.events
        if isinstance(event, ChildEvent) and isinstance(event.source_event, TaskStarted) and event.task_id == "first"
    )
    assert handle is not None
    ctx.variables["number"] = 2
    second = asyncio.create_task(run_task(worker, "second", parent_context=ctx, stream=stream, task_id="second"))
    try:
        with pytest.raises(DeliveryError):
            await handle.send(HaltEvent(reason="stop first", source="test"))
    finally:
        race.release_first.set()
        race.release_second.set()
        results = await asyncio.gather(first, second)
    assert all(result.completed for result in results)


async def halt_on_spend(event: UsageEvent) -> None:
    await event.origin.send(HaltEvent(reason="budget spent", source="test"))


@pytest.mark.asyncio
@pytest.mark.parametrize("emit_events", [True, False])
async def test_usage_retained_at_source_when_upward_delivery_cancels_it(emit_events: bool) -> None:
    ctx = Context(MemoryStream())
    ctx.stream.where(UsageEvent).subscribe(halt_on_spend)
    child = Agent(
        "child", config=TestConfig(ModelResponse(ModelMessage("done"), usage=Usage(total_tokens=30), model="child"))
    )
    result = await run_task(child, "work", parent_context=ctx, emit_events=emit_events)
    assert result.cancelled
    assert result.usage == Usage(total_tokens=30)
    assert UsageReport.from_events(await result.stream.history.get_events()).total == Usage(total_tokens=30)
    assert UsageReport.from_events(await ctx.stream.history.get_events()).total == Usage(total_tokens=30)


@dataclass
class UsagePause:
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def pause(self, event: UsageEvent) -> UsageEvent:
        self.entered.set()
        await self.release.wait()
        return event


@pytest.mark.asyncio
async def test_parent_cancellation_retains_usage_while_its_interrupter_is_awaiting() -> None:
    pause = UsagePause()
    ctx = Context(MemoryStream())
    ctx.stream.where(UsageEvent).subscribe(pause.pause, interrupt=True)
    stream = MemoryStream(storage=ctx.stream.history.storage)
    child = Agent("child", config=TestConfig(UsageEvent(Usage(total_tokens=30)), "done"))
    task = asyncio.create_task(run_task(child, "work", parent_context=ctx, stream=stream))
    await pause.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert UsageReport.from_events(await stream.history.get_events()).total == Usage(total_tokens=30)
    assert UsageReport.from_events(await ctx.stream.history.get_events()).total == Usage(total_tokens=30)


class MiddleSender:
    async def guide(self, event: BaseEvent) -> None:
        if isinstance(event, ChildEvent) and isinstance(event.inner, WorkSignal):
            await event.origin.send(Directive(text="from middle"))


@pytest.mark.asyncio
async def test_nested_delivery_records_the_calling_ancestor_as_author() -> None:
    guidance = Guidance()
    leaf = Agent(
        "leaf",
        config=TestConfig(ToolCallEvent(name="seek_guidance", arguments="{}"), "done"),
        tools=[seek_guidance],
        observers=[observer(callback=guidance.receive)],
    )
    sender = MiddleSender()
    middle = Agent(
        "middle",
        config=TestConfig(ToolCallEvent(name="task_leaf", arguments='{"objective":"work"}'), "done"),
        tools=[leaf.as_tool(description="Work")],
        observers=[observer(callback=sender.guide)],
    )
    ctx = Context(MemoryStream())
    result = await run_task(middle, "organize", parent_context=ctx, task_id="middle")
    assert result.completed
    audits = [
        event.inner
        for event in await ctx.stream.history.get_events()
        if isinstance(event, ChildEvent) and isinstance(event.inner, DownEvent)
    ]
    assert len(audits) == 1
    assert audits[0].to_dict() == IsPartialDict({
        "origin_task_id": "middle",
        "origin_path": ["middle"],
        "origin_stream_id": str(result.stream.id),
    })
