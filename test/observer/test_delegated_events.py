# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import pytest

from ag2 import Agent, observer
from ag2.events import BaseEvent, ChildEvent, ObserverAlert, ToolCallEvent, Usage, UsageEvent
from ag2.observers import LoopDetector, TokenMonitor
from ag2.testing import TestConfig


class WorkSignal(BaseEvent):
    __conversational__ = False


@dataclass
class Journal:
    events: list[BaseEvent] = field(default_factory=list)

    async def record(self, event: BaseEvent) -> None:
        self.events.append(event)


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
