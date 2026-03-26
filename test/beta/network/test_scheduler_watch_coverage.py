# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Additional Scheduler + Watch coverage tests.

Covers gaps identified during review:
- EventWatch with Scheduler + Hub (reactive path from design doc)
- task_factory raising with EventWatch (Bug 1 regression)
- resume() when scheduler is not running
- add() with no callback and no target (silent no-op)
- stop() idempotency
- cancel() on an armed watch mid-flight
- Sequence with > 2 watches
- WindowWatch spanning multiple flush cycles
- CronWatch arm + fire cycle
"""

import asyncio
import logging

import pytest

from autogen.beta.context import Context
from autogen.beta.events import ModelMessage, ToolCallEvent
from autogen.beta.events.base import BaseEvent
from autogen.beta.network.events import SchedulerTriggerFired
from autogen.beta.network.hub import Hub
from autogen.beta.network.primitives.watch import (
    CronWatch,
    EventWatch,
    IntervalWatch,
    Sequence,
    WindowWatch,
)
from autogen.beta.network.scheduler import Scheduler, WatchStatus
from autogen.beta.stream import MemoryStream

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class _AskableAgent:
    def __init__(self, name: str, result: str = "done"):
        self.name = name
        self._result = result
        self.asked: list[str] = []

    async def ask(self, message, **kwargs):
        self.asked.append(message)
        return type("Reply", (), {"content": self._result, "body": self._result})()


# ---------------------------------------------------------------------------
# EventWatch + Scheduler + Hub (reactive delegation path)
# ---------------------------------------------------------------------------


class TestEventWatchWithSchedulerHub:
    @pytest.mark.asyncio
    async def test_event_watch_triggers_delegation(self) -> None:
        """EventWatch on hub stream fires scheduler → delegates to target.
        This is the core reactive pattern from the design doc."""
        agent = _AskableAgent("reporter", result="report done")
        hub = Hub()
        await hub.register(agent)

        scheduler = Scheduler(hub=hub)
        scheduler.add(
            EventWatch(ModelMessage),
            target="reporter",
            task="Summarize the message",
        )
        await scheduler.start()

        # Send a ModelMessage on the hub stream to trigger the EventWatch
        hub_ctx = Context(stream=hub.stream)
        await hub.stream.send(ModelMessage(content="hello world"), hub_ctx)

        # Give async tasks time to complete delegation
        await asyncio.sleep(0.1)
        await scheduler.stop()

        assert len(agent.asked) >= 1
        assert agent.asked[0] == "Summarize the message"

    @pytest.mark.asyncio
    async def test_event_watch_with_task_factory(self) -> None:
        """EventWatch + task_factory: the factory receives the triggering events."""
        agent = _AskableAgent("auditor", result="audit done")
        hub = Hub()
        await hub.register(agent)

        received_events: list[list[BaseEvent]] = []

        def factory(events):
            received_events.append(events)
            return f"Audit: {len(events)} event(s)"

        scheduler = Scheduler(hub=hub)
        scheduler.add(
            EventWatch(ModelMessage),
            target="auditor",
            task_factory=factory,
        )
        await scheduler.start()

        hub_ctx = Context(stream=hub.stream)
        await hub.stream.send(ModelMessage(content="trigger"), hub_ctx)

        await asyncio.sleep(0.1)
        await scheduler.stop()

        assert len(agent.asked) >= 1
        assert "Audit: 1 event(s)" in agent.asked[0]
        assert len(received_events) == 1
        assert isinstance(received_events[0][0], ModelMessage)

    @pytest.mark.asyncio
    async def test_trigger_event_emitted_for_event_watch(self) -> None:
        """SchedulerTriggerFired should appear on hub stream for EventWatch too."""
        from autogen.beta.events.conditions import TypeCondition

        agent = _AskableAgent("worker", result="ok")
        hub = Hub()
        await hub.register(agent)

        captured: list[SchedulerTriggerFired] = []

        async def _capture(event: SchedulerTriggerFired) -> None:
            captured.append(event)

        hub.stream.subscribe(_capture, condition=TypeCondition(SchedulerTriggerFired))

        scheduler = Scheduler(hub=hub)
        watch_id = scheduler.add(
            EventWatch(ModelMessage),
            target="worker",
            task="react",
        )
        await scheduler.start()

        hub_ctx = Context(stream=hub.stream)
        await hub.stream.send(ModelMessage(content="go"), hub_ctx)

        await asyncio.sleep(0.1)
        await scheduler.stop()

        assert len(captured) >= 1
        assert captured[0].watch_id == watch_id
        assert captured[0].target == "worker"


# ---------------------------------------------------------------------------
# Bug 1 regression: task_factory raising with EventWatch
# ---------------------------------------------------------------------------


class TestTaskFactoryErrorWithEventWatch:
    @pytest.mark.asyncio
    async def test_task_factory_exception_caught_with_event_watch(self, caplog) -> None:
        """task_factory raising with EventWatch should be caught, not crash the stream."""
        agent = _AskableAgent("worker")
        hub = Hub()
        await hub.register(agent)

        def bad_factory(events):
            raise ValueError("factory boom")

        scheduler = Scheduler(hub=hub)
        scheduler.add(
            EventWatch(ModelMessage),
            target="worker",
            task_factory=bad_factory,
        )
        await scheduler.start()

        hub_ctx = Context(stream=hub.stream)
        with caplog.at_level(logging.ERROR):
            await hub.stream.send(ModelMessage(content="trigger"), hub_ctx)
            await asyncio.sleep(0.1)

        await scheduler.stop()

        # Agent should NOT have been asked (factory failed before delegation)
        assert len(agent.asked) == 0
        # Scheduler should still be alive
        assert len(scheduler.watches) == 1

        # Stream should still be functional after the error
        received: list = []

        async def _check(event: ModelMessage) -> None:
            received.append(event)

        hub.stream.subscribe(_check, condition=None)
        await hub.stream.send(ModelMessage(content="after error"), hub_ctx)
        assert any(isinstance(e, ModelMessage) and e.content == "after error" for e in received)


# ---------------------------------------------------------------------------
# Scheduler edge cases
# ---------------------------------------------------------------------------


class TestSchedulerEdgeCases:
    @pytest.mark.asyncio
    async def test_resume_when_not_running(self) -> None:
        """resume() should be a no-op when scheduler is not running."""
        scheduler = Scheduler()

        async def noop(events, ctx):
            pass

        wid = scheduler.add(IntervalWatch(999), callback=noop)
        await scheduler.start()
        await scheduler.stop()

        # Now stopped — resume should not re-arm
        scheduler.resume(wid)
        assert scheduler.watches[0][2] == WatchStatus.PAUSED

    @pytest.mark.asyncio
    async def test_add_with_no_callback_and_no_target(self) -> None:
        """add() with neither callback nor target: fires silently do nothing."""
        scheduler = Scheduler()
        scheduler.add(IntervalWatch(0.05))
        await scheduler.start()

        # Let it fire a few times — should not crash
        await asyncio.sleep(0.2)
        await scheduler.stop()

        # Watch was armed and now paused — no crash
        assert scheduler.watches[0][2] == WatchStatus.PAUSED

    @pytest.mark.asyncio
    async def test_stop_idempotent(self) -> None:
        """Calling stop() twice should not raise."""
        scheduler = Scheduler()

        async def noop(events, ctx):
            pass

        scheduler.add(IntervalWatch(999), callback=noop)
        await scheduler.start()
        await scheduler.stop()
        await scheduler.stop()  # Second stop — should be a no-op

        assert scheduler.watches[0][2] == WatchStatus.PAUSED

    @pytest.mark.asyncio
    async def test_cancel_armed_watch(self) -> None:
        """cancel() on an armed watch should disarm it cleanly."""
        scheduler = Scheduler()
        call_count = 0

        async def callback(events, ctx):
            nonlocal call_count
            call_count += 1

        wid = scheduler.add(IntervalWatch(0.05), callback=callback)
        await scheduler.start()

        # Let it fire at least once
        await asyncio.sleep(0.1)
        count_before = call_count

        # Cancel mid-flight
        result = scheduler.cancel(wid)
        assert result is True
        assert scheduler.status(wid) == WatchStatus.CANCELLED

        # After cancel, no more fires
        await asyncio.sleep(0.15)
        assert call_count == count_before

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self) -> None:
        """Calling start() twice should not double-arm watches."""
        scheduler = Scheduler()
        call_count = 0

        async def callback(events, ctx):
            nonlocal call_count
            call_count += 1

        scheduler.add(IntervalWatch(0.05), callback=callback)
        await scheduler.start()
        await scheduler.start()  # Second start — should be a no-op

        await asyncio.sleep(0.15)
        await scheduler.stop()

        # Should have fired a reasonable number of times, not double
        assert call_count >= 2
        assert call_count < 10  # Not doubled


# ---------------------------------------------------------------------------
# Sequence with > 2 watches
# ---------------------------------------------------------------------------


class TestSequenceThreeSteps:
    @pytest.mark.asyncio
    async def test_three_step_sequence(self) -> None:
        """Sequence with 3 watches: must fire in exact order."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        received: list = []

        async def callback(events, _ctx):
            received.append(events)

        w = Sequence(
            EventWatch(ToolCallEvent),
            EventWatch(ModelMessage),
            EventWatch(ToolCallEvent),
        )
        w.arm(stream, callback)

        # Step 1: ToolCallEvent
        await stream.send(ToolCallEvent(name="t1", arguments="{}"), ctx)
        assert len(received) == 0

        # Step 2: ModelMessage
        await stream.send(ModelMessage(content="m1"), ctx)
        assert len(received) == 0

        # Step 3: ToolCallEvent — completes the sequence
        await stream.send(ToolCallEvent(name="t2", arguments="{}"), ctx)
        assert len(received) == 1
        assert len(received[0]) == 3  # All 3 events collected

    @pytest.mark.asyncio
    async def test_three_step_sequence_resets(self) -> None:
        """After a 3-step sequence completes, it resets and can fire again."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        received: list = []

        async def callback(events, _ctx):
            received.append(events)

        w = Sequence(
            EventWatch(ToolCallEvent),
            EventWatch(ModelMessage),
            EventWatch(ToolCallEvent),
        )
        w.arm(stream, callback)

        # First cycle
        await stream.send(ToolCallEvent(name="t1", arguments="{}"), ctx)
        await stream.send(ModelMessage(content="m1"), ctx)
        await stream.send(ToolCallEvent(name="t2", arguments="{}"), ctx)
        assert len(received) == 1

        # Second cycle
        await stream.send(ToolCallEvent(name="t3", arguments="{}"), ctx)
        await stream.send(ModelMessage(content="m2"), ctx)
        await stream.send(ToolCallEvent(name="t4", arguments="{}"), ctx)
        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_three_step_wrong_order_ignored(self) -> None:
        """Events out of order should not advance the sequence."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        received: list = []

        async def callback(events, _ctx):
            received.append(events)

        w = Sequence(
            EventWatch(ToolCallEvent),
            EventWatch(ModelMessage),
            EventWatch(ToolCallEvent),
        )
        w.arm(stream, callback)

        # Wrong: start with ModelMessage (expects ToolCallEvent first)
        await stream.send(ModelMessage(content="m1"), ctx)
        assert len(received) == 0

        # Correct step 1
        await stream.send(ToolCallEvent(name="t1", arguments="{}"), ctx)
        # Wrong: another ToolCallEvent (expects ModelMessage)
        await stream.send(ToolCallEvent(name="t2", arguments="{}"), ctx)
        # Still no fire — wrong order at step 2
        assert len(received) == 0


# ---------------------------------------------------------------------------
# WindowWatch spanning multiple flush cycles
# ---------------------------------------------------------------------------


class TestWindowWatchMultipleWindows:
    @pytest.mark.asyncio
    async def test_two_separate_windows(self) -> None:
        """Events in separate time windows produce separate batches."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        batches: list = []

        async def callback(events, _ctx):
            batches.append(events)

        watch = WindowWatch(0.1, condition=ToolCallEvent)
        watch.arm(stream, callback)

        # First window
        await stream.send(ToolCallEvent(name="t1", arguments="{}"), ctx)
        await stream.send(ToolCallEvent(name="t2", arguments="{}"), ctx)

        # Wait for first window to flush
        await asyncio.sleep(0.2)
        assert len(batches) == 1
        assert len(batches[0]) == 2

        # Second window
        await stream.send(ToolCallEvent(name="t3", arguments="{}"), ctx)

        # Wait for second window to flush
        await asyncio.sleep(0.2)
        assert len(batches) == 2
        assert len(batches[1]) == 1

        watch.disarm()


# ---------------------------------------------------------------------------
# CronWatch arm + fire cycle
# ---------------------------------------------------------------------------


class TestCronWatchArmFire:
    @pytest.mark.asyncio
    async def test_cron_watch_fires_callback(self) -> None:
        """CronWatch should actually fire its callback when armed.

        We use a trick: patch _next_fire_time to return a time very close to now
        so we don't have to wait for real cron alignment.
        """
        import datetime

        stream = MemoryStream()
        call_count = 0

        async def callback(events, _ctx):
            nonlocal call_count
            call_count += 1

        watch = CronWatch("* * * * *")  # Every minute

        # Monkey-patch to fire almost immediately

        def _fast_next(now):
            return now + datetime.timedelta(milliseconds=50)

        watch._next_fire_time = _fast_next

        watch.arm(stream, callback)
        assert watch.is_armed

        await asyncio.sleep(0.2)
        watch.disarm()
        assert not watch.is_armed

        # Should have fired at least once
        assert call_count >= 1
