# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import pytest

from ag2 import Agent, MemoryStream, observer
from ag2.events import (
    BaseEvent,
    ChildEvent,
    ModelMessage,
    ModelResponse,
    ObserverAlert,
    ToolCallEvent,
    Usage,
    UsageEvent,
)
from ag2.observers import LoopDetector, TokenMonitor
from ag2.testing import TestConfig
from ag2.usage import UsageReport


class WorkSignal(BaseEvent):
    __conversational__ = False


@dataclass
class Journal:
    events: list[BaseEvent] = field(default_factory=list)

    async def record(self, event: BaseEvent) -> None:
        self.events.append(event)


@dataclass
class UsageJournal:
    events: list[UsageEvent] = field(default_factory=list)

    async def record(self, event: UsageEvent) -> None:
        self.events.append(event)


@pytest.mark.asyncio
@pytest.mark.parametrize("subscription", ["stream", "observer"])
async def test_usage_subscriber_receives_unwrapped_descendant_costs_with_source_identity(subscription: str) -> None:
    leaf_stream, middle_stream, root_stream = MemoryStream(), MemoryStream(), MemoryStream()
    leaf = Agent(
        "leaf",
        config=TestConfig(
            UsageEvent(Usage(total_tokens=30), kind="compaction", model="leaf", provider="leaf-provider"), "done"
        ),
    )
    middle = Agent(
        "middle",
        config=TestConfig(
            ToolCallEvent(name="task_leaf", arguments='{"objective":"work"}'),
            ModelResponse(ModelMessage("done"), usage=Usage(total_tokens=20), model="middle"),
        ),
        tools=[leaf.as_tool(description="Work", stream=leaf_stream)],
    )
    root = Agent(
        "root",
        config=TestConfig(
            ToolCallEvent(name="task_middle", arguments='{"objective":"work"}'),
            ModelResponse(ModelMessage("done"), usage=Usage(total_tokens=10), model="root"),
        ),
        tools=[middle.as_tool(description="Work", stream=middle_stream)],
    )
    usage, journal = UsageJournal(), Journal()
    observers = []
    if subscription == "stream":
        root_stream.where(UsageEvent).subscribe(usage.record)
    else:
        observers.append(observer(UsageEvent, usage.record))
    root_stream.subscribe(journal.record)

    reply = await root.ask("start", stream=root_stream, observers=observers)

    assert [(event.model, event.usage.total_tokens) for event in usage.events] == [
        ("leaf", 30),
        ("middle", 20),
        ("root", 10),
    ]
    assert (usage.events[0].kind, usage.events[0].provider) == ("compaction", "leaf-provider")
    assert not any(isinstance(event, ChildEvent) and isinstance(event.inner, UsageEvent) for event in journal.events)
    leaf_history = list(await leaf_stream.history.get_events())
    source = next(event for event in leaf_history if isinstance(event, UsageEvent))
    assert usage.events[0].event_id == source.event_id
    histories = [*leaf_history, *await middle_stream.history.get_events(), *await root_stream.history.get_events()]
    assert UsageReport.from_events(histories).total.total_tokens == 60
    assert (await reply.usage()).total.total_tokens == 60


@dataclass
class BudgetProbe:
    monitor: TokenMonitor
    observed: list[int] = field(default_factory=list)

    async def record(self, event: BaseEvent) -> None:
        if isinstance(event, ChildEvent) and isinstance(event.inner, WorkSignal):
            self.observed.append(self.monitor.total_tokens)


@pytest.mark.asyncio
async def test_root_monitor_counts_source_usage_before_child_finishes() -> None:
    monitor = TokenMonitor()
    probe = BudgetProbe(monitor)
    child = Agent("child", config=TestConfig(UsageEvent(Usage(total_tokens=30), model="leaf"), WorkSignal(), "done"))
    root = Agent(
        "root",
        config=TestConfig(ToolCallEvent(name="task_child", arguments='{"objective":"work"}'), "done"),
        tools=[child.as_tool(description="Work")],
    )
    reply = await root.ask("start", observers=[monitor, observer(callback=probe.record)])
    assert probe.observed == [30]
    assert (await reply.usage()).total.total_tokens == 30


async def do_nothing() -> str:
    """A repeated observable operation."""
    return "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("repeats, alerts", [(1, 0), (3, 3)])
async def test_loop_detector_observes_descendants_without_combining_sibling_histories(
    repeats: int, alerts: int
) -> None:
    children = [
        Agent(
            f"worker-{i}",
            config=TestConfig(*(ToolCallEvent(name="do_nothing", arguments="{}") for _ in range(repeats)), "done"),
            tools=[do_nothing],
        )
        for i in range(3)
    ]
    root = Agent(
        "root",
        config=TestConfig(
            [ToolCallEvent(name=f"task_worker-{i}", arguments='{"objective":"work"}') for i in range(3)], "done"
        ),
        tools=[child.as_tool(description="Work") for child in children],
    )
    journal = Journal()
    await root.ask("start", observers=[LoopDetector(), observer(callback=journal.record)])
    assert len([event for event in journal.events if isinstance(event, ObserverAlert)]) == alerts
