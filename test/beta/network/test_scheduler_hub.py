# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Scheduler operating in Hub mode (delegation via hub._delegate)."""

import asyncio
import logging

import pytest

from autogen.beta.network.events import SchedulerTriggerFired
from autogen.beta.network.hub import Hub
from autogen.beta.network.primitives.watch import IntervalWatch
from autogen.beta.network.scheduler import Scheduler, WatchStatus


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class _AskableAgent:
    """Minimal mock agent that can handle ask() calls without an LLM."""

    def __init__(self, name: str, result: str = "done"):
        self.name = name
        self._result = result
        self.asked: list[str] = []

    async def ask(self, message, **kwargs):
        self.asked.append(message)
        return type("Reply", (), {"content": self._result})()


class _FailingAgent:
    """Mock agent whose ask() always raises."""

    def __init__(self, name: str):
        self.name = name

    async def ask(self, message, **kwargs):
        raise RuntimeError("agent exploded")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSchedulerWithHub:
    """Scheduler behaviour when connected to a Hub."""

    @pytest.mark.asyncio
    async def test_interval_watch_delegates_to_target(self) -> None:
        """IntervalWatch fires and the Scheduler delegates the task to the
        target agent through the Hub."""
        agent = _AskableAgent("monitor", result="health ok")
        hub = Hub()
        await hub.register(agent)

        scheduler = Scheduler(hub=hub)
        scheduler.add(
            IntervalWatch(0.05),
            target="monitor",
            task="Check health",
        )
        await scheduler.start()

        # Wait long enough for at least one interval tick
        await asyncio.sleep(0.25)
        await scheduler.stop()

        assert len(agent.asked) >= 1
        assert agent.asked[0] == "Check health"

    @pytest.mark.asyncio
    async def test_task_factory_produces_dynamic_task(self) -> None:
        """When a task_factory is provided, the Scheduler uses its return
        value as the delegation task instead of the static ``task`` string."""
        agent = _AskableAgent("worker", result="ok")
        hub = Hub()
        await hub.register(agent)

        call_count = 0

        def factory(events):
            nonlocal call_count
            call_count += 1
            return f"dynamic task #{call_count}"

        scheduler = Scheduler(hub=hub)
        scheduler.add(
            IntervalWatch(0.05),
            target="worker",
            task_factory=factory,
        )
        await scheduler.start()

        await asyncio.sleep(0.25)
        await scheduler.stop()

        assert len(agent.asked) >= 1
        # The first delegated task should come from the factory
        assert agent.asked[0] == "dynamic task #1"

    @pytest.mark.asyncio
    async def test_trigger_fired_event_emitted_on_hub_stream(self) -> None:
        """A SchedulerTriggerFired event should appear on hub.stream when an
        IntervalWatch fires in Hub mode."""
        agent = _AskableAgent("monitor", result="ok")
        hub = Hub()
        await hub.register(agent)

        captured: list[SchedulerTriggerFired] = []

        async def _capture(event: SchedulerTriggerFired) -> None:
            captured.append(event)

        hub.stream.subscribe(
            _capture,
            condition=None,  # subscribe to all; we filter manually below
        )

        scheduler = Scheduler(hub=hub)
        watch_id = scheduler.add(
            IntervalWatch(0.05),
            target="monitor",
            task="ping",
        )
        await scheduler.start()

        await asyncio.sleep(0.25)
        await scheduler.stop()

        trigger_events = [e for e in captured if isinstance(e, SchedulerTriggerFired)]
        assert len(trigger_events) >= 1
        evt = trigger_events[0]
        assert evt.watch_id == watch_id
        assert evt.target == "monitor"
        assert evt.task == "ping"

    @pytest.mark.asyncio
    async def test_delegation_failure_is_caught(self, caplog) -> None:
        """When the target agent's ask() raises, the Scheduler should log the
        error and NOT crash (other watches keep running)."""
        agent = _FailingAgent("broken")
        hub = Hub()
        await hub.register(agent)

        scheduler = Scheduler(hub=hub)
        scheduler.add(
            IntervalWatch(0.05),
            target="broken",
            task="will fail",
        )

        with caplog.at_level(logging.ERROR):
            await scheduler.start()
            await asyncio.sleep(0.25)
            await scheduler.stop()

        # Scheduler should still be in a valid state (not crashed)
        watches = scheduler.watches
        assert len(watches) == 1
        assert watches[0][2] == WatchStatus.PAUSED  # stopped cleanly

    @pytest.mark.asyncio
    async def test_callback_mode_works_with_hub_present(self) -> None:
        """When a watch is registered with a callback (standalone mode) but
        the Scheduler is connected to a Hub, the callback should still fire
        directly — no delegation should occur."""
        agent = _AskableAgent("unused", result="should not be called")
        hub = Hub()
        await hub.register(agent)

        received: list[bool] = []

        async def my_callback(events, ctx):
            received.append(True)

        scheduler = Scheduler(hub=hub)
        scheduler.add(IntervalWatch(0.05), callback=my_callback)
        await scheduler.start()

        await asyncio.sleep(0.25)
        await scheduler.stop()

        # Callback was invoked
        assert len(received) >= 2
        # The agent was NOT asked (callback mode bypasses delegation)
        assert len(agent.asked) == 0
