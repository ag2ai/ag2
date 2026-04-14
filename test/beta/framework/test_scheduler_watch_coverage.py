# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Additional Scheduler + Watch coverage tests.

Covers gaps identified during review:
- resume() when scheduler is not running
- add() with no callback and no target (silent no-op)
- stop() idempotency
- cancel() on an armed watch mid-flight
- Sequence with > 2 watches
- WindowWatch spanning multiple flush cycles
- CronWatch arm + fire cycle

Note: the V2 network tests (EventWatch + Hub reactive delegation, Bug 1
regression on task_factory raising) were dropped during the V3 rewrite
along with the rest of the V2 Hub. They will be re-added when network
Tasks land in Phase 4.
"""

import asyncio

import pytest

from autogen.beta.context import ConversationContext as Context
from autogen.beta.events import ModelMessage, ToolCallEvent
from autogen.beta.events.base import BaseEvent
from autogen.beta.scheduler import Scheduler, WatchStatus
from autogen.beta.stream import MemoryStream
from autogen.beta.watch import (
    CronWatch,
    EventWatch,
    IntervalWatch,
    Sequence,
    WindowWatch,
)

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
