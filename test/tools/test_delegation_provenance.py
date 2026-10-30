# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from dataclasses import dataclass, field

import pytest

from ag2 import Agent, Context, KnowledgeConfig, MemoryStream, observer
from ag2.compact import CompactTrigger, TailWindowCompact
from ag2.events import (
    BaseEvent,
    ChildEvent,
    HaltEvent,
    HumanInputRequest,
    ModelMessage,
    ModelResponse,
    TaskCancelled,
    TaskCompleted,
    TaskStarted,
    ToolCallEvent,
    Usage,
    UsageEvent,
)
from ag2.events.task_events import TaskEvent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.observers import BaseObserver
from ag2.testing import TestConfig
from ag2.usage import UsageReport
from ag2.watch import EventWatch


@dataclass
class Journal:
    events: list[BaseEvent] = field(default_factory=list)

    async def record(self, event: BaseEvent) -> None:
        self.events.append(event)


def spending(name: str, tokens: int, *calls: ToolCallEvent, tools: list | None = None) -> Agent:
    return Agent(
        name,
        config=TestConfig(
            *calls,
            ModelResponse(ModelMessage(f"{name} done"), usage=Usage(total_tokens=tokens), model=f"{name}-model"),
        ),
        tools=tools or [],
    )


def three_levels() -> Agent:
    leaf = spending("leaf", 30)
    worker = spending(
        "worker",
        20,
        ToolCallEvent(name="task_leaf", arguments='{"objective": "dig"}'),
        tools=[leaf.as_tool(description="Dig")],
    )
    return spending(
        "root",
        10,
        ToolCallEvent(name="task_worker", arguments='{"objective": "organize"}'),
        tools=[worker.as_tool(description="Organize")],
    )


def reload(events: list[BaseEvent]) -> list[BaseEvent]:
    return [type(event).from_dict(json.loads(json.dumps(event.to_dict()))) for event in events]


def task_ids(events: list[BaseEvent]) -> dict[str, str]:
    return {event.agent_name: event.task_id for event in events if isinstance(event, TaskStarted)}


@pytest.mark.asyncio
async def test_stored_usage_names_its_spender_at_every_level() -> None:
    reply = await three_levels().ask("start")

    history = reload(list(await reply.history.get_events()))

    ids = task_ids(history)
    spent = sorted(
        (event.agent_name or "", event.task_id or "", list(event.path), event.usage.total_tokens)
        for event in history
        if isinstance(event, UsageEvent)
    )
    assert spent == sorted([
        ("", "", [], 10),
        ("worker", ids["worker"], [ids["worker"]], 20),
        ("leaf", ids["leaf"], [ids["worker"], ids["leaf"]], 30),
    ])
    assert UsageReport.from_events(history).total.total_tokens == 60
    assert UsageReport.from_events([*history, *reload(history)]).total.total_tokens == 60


@pytest.mark.asyncio
async def test_descendant_usage_keeps_its_spender_through_compaction() -> None:
    compaction = KnowledgeConfig(
        store=MemoryKnowledgeStore(),
        compact=TailWindowCompact(target=1),
        compact_trigger=CompactTrigger(max_events=1),
        expose_tool=False,
        write_event_log=False,
    )
    leaf = spending("leaf", 30)
    worker = Agent(
        "worker",
        config=TestConfig(ToolCallEvent(name="task_leaf", arguments='{"objective": "dig"}'), "worker done"),
        tools=[leaf.as_tool(description="Dig")],
        knowledge=compaction,
    )
    root = Agent(
        "root",
        config=TestConfig(ToolCallEvent(name="task_worker", arguments='{"objective": "organize"}'), "root done"),
        tools=[worker.as_tool(description="Organize")],
        knowledge=compaction,
    )

    reply = await root.ask("start")

    history = reload(list(await reply.history.get_events()))
    spent = [event for event in history if isinstance(event, UsageEvent)]
    assert [(event.agent_name, len(event.path), event.usage.total_tokens) for event in spent] == [("leaf", 2, 30)]
    assert UsageReport.from_events(history).total.total_tokens == 30


@pytest.mark.asyncio
async def test_typed_lifecycle_observer_on_root_sees_child_and_grandchild() -> None:
    journal = Journal()

    await three_levels().ask("start", observers=[observer(TaskStarted | TaskCompleted, callback=journal.record)])

    ids = task_ids(journal.events)
    assert [(type(event).__name__, event.agent_name, list(event.path)) for event in journal.events] == [
        ("TaskStarted", "worker", [ids["worker"]]),
        ("TaskStarted", "leaf", [ids["worker"], ids["leaf"]]),
        ("TaskCompleted", "leaf", [ids["worker"], ids["leaf"]]),
        ("TaskCompleted", "worker", [ids["worker"]]),
    ]


@pytest.mark.asyncio
async def test_lifecycle_events_are_not_wrapped_at_any_ancestor() -> None:
    reply = await three_levels().ask("start")

    history = list(await reply.history.get_events())
    assert not any(isinstance(event, ChildEvent) and isinstance(event.inner, TaskEvent) for event in history)
    assert {(type(event).__name__, event.agent_name) for event in history if isinstance(event, TaskEvent)} == {
        ("TaskStarted", "worker"),
        ("TaskStarted", "leaf"),
        ("TaskCompleted", "leaf"),
        ("TaskCompleted", "worker"),
    }


class LeafStopper:
    async def stop(self, event: TaskStarted) -> None:
        if event.agent_name == "leaf":
            await event.origin.send(HaltEvent(reason="not needed", source="root"))


@pytest.mark.asyncio
async def test_root_halts_grandchild_through_its_forwarded_lifecycle_event() -> None:
    journal = Journal()
    stopper = LeafStopper()

    reply = await three_levels().ask(
        "start",
        observers=[observer(TaskStarted, callback=stopper.stop), observer(TaskCancelled, callback=journal.record)],
    )

    assert reply.body == "root done"
    assert [(event.agent_name, len(event.path), event.reason) for event in journal.events] == [
        ("leaf", 2, "not needed")
    ]


class SpendByModel(BaseObserver):
    """Sums every ``UsageEvent`` it is shown, with no deduplication, like a cost observer."""

    def __init__(self) -> None:
        super().__init__("spend-by-model", watch=EventWatch(UsageEvent))
        self.tokens: dict[str | None, int] = {}

    async def process(self, events: list[BaseEvent], ctx: Context) -> None:
        for event in events:
            if isinstance(event, UsageEvent):
                self.tokens[event.model] = self.tokens.get(event.model, 0) + (event.usage.total_tokens or 0)


@pytest.mark.asyncio
async def test_usage_watch_on_root_counts_each_descendant_call_once() -> None:
    spend = SpendByModel()

    await three_levels().ask("start", observers=[spend])

    assert spend.tokens == {"root-model": 10, "worker-model": 20, "leaf-model": 30}


@dataclass
class Calls:
    count: int = 0

    async def probe(self) -> str:
        """Count a call."""
        self.count += 1
        return "probed"


@dataclass
class Hook:
    questions: list[str] = field(default_factory=list)

    async def answer(self, event: HumanInputRequest) -> str:
        self.questions.append(event.content)
        return "yes"


@pytest.mark.asyncio
async def test_forwarded_events_do_not_run_the_roots_tools_model_or_hitl() -> None:
    leaf_calls, root_calls, hook = Calls(), Calls(), Hook()
    leaf = Agent(
        "leaf",
        config=TestConfig(
            ToolCallEvent(name="probe", arguments="{}"),
            ModelResponse(ModelMessage("leaf done"), usage=Usage(total_tokens=30)),
        ),
        tools=[leaf_calls.probe],
    )
    worker = Agent(
        "worker",
        config=TestConfig(ToolCallEvent(name="task_leaf", arguments='{"objective": "dig"}'), "worker done"),
        tools=[leaf.as_tool(description="Dig")],
    )
    root = Agent(
        "root",
        config=TestConfig(ToolCallEvent(name="task_worker", arguments='{"objective": "organize"}'), "root done"),
        tools=[worker.as_tool(description="Organize"), root_calls.probe],
        hitl_hook=hook.answer,
    )

    reply = await root.ask("start")

    assert reply.body == "root done"
    assert (leaf_calls.count, root_calls.count, hook.questions) == (1, 0, [])
    history = list(await reply.history.get_events())
    assert len([event for event in history if isinstance(event, ModelResponse)]) == 2


@dataclass
class UsageHold:
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    context: Context | None = None

    async def hold(self, event: UsageEvent, ctx: Context) -> None:
        self.context = ctx
        self.entered.set()
        await self.release.wait()


@pytest.mark.asyncio
async def test_root_cancel_keeps_grandchild_usage_once_at_every_level() -> None:
    hold = UsageHold()
    leaf = Agent("leaf", config=TestConfig(UsageEvent(Usage(total_tokens=30)), "leaf done"))
    middle = Agent(
        "middle",
        config=TestConfig(ToolCallEvent(name="task_leaf", arguments='{"objective": "dig"}'), "middle done"),
        tools=[leaf.as_tool(description="Dig")],
        observers=[observer(UsageEvent, callback=hold.hold, interrupt=True)],
    )
    root = Agent(
        "root",
        config=TestConfig(ToolCallEvent(name="task_middle", arguments='{"objective": "organize"}'), "root done"),
        tools=[middle.as_tool(description="Organize")],
    )
    stream = MemoryStream()

    turn = asyncio.create_task(root.ask("start", stream=stream))
    await asyncio.wait_for(hold.entered.wait(), 2)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert hold.context is not None
    for history, cancelled in (
        (list(await stream.history.get_events()), "middle"),
        (list(await hold.context.stream.history.get_events()), "leaf"),
    ):
        records = [index for index, event in enumerate(history) if isinstance(event, UsageEvent)]
        terminals = [
            index
            for index, event in enumerate(history)
            if isinstance(event, TaskCancelled) and event.agent_name == cancelled
        ]
        assert len(records) == 1, cancelled
        assert len(terminals) == 1, cancelled
        assert records[0] < terminals[0], cancelled
        assert UsageReport.from_events(history).total == Usage(total_tokens=30)
