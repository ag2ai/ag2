# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import datetime

import pytest

from autogen.beta.context import ConversationContext as Context
from autogen.beta.events import ModelMessage, ToolCallEvent
from autogen.beta.stream import MemoryStream
from autogen.beta.watch import (
    AllOf,
    AnyOf,
    CadenceWatch,
    CronWatch,
    DelayWatch,
    EventWatch,
    IntervalWatch,
    Sequence,
)


class TestEventWatch:
    @pytest.mark.asyncio
    async def test_fires_on_matching_event(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        received: list = []

        async def callback(events, _ctx):
            received.extend(events)

        watch = EventWatch(ToolCallEvent)
        watch.arm(stream, callback)
        assert watch.is_armed

        event = ToolCallEvent(name="search", arguments="{}")
        await stream.send(event, ctx)
        assert len(received) == 1
        assert received[0] is event

    @pytest.mark.asyncio
    async def test_does_not_fire_on_non_matching(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        received: list = []

        async def callback(events, _ctx):
            received.extend(events)

        watch = EventWatch(ToolCallEvent)
        watch.arm(stream, callback)

        await stream.send(ModelMessage(content="hello"), ctx)
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_condition_filter(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        received: list = []

        async def callback(events, _ctx):
            received.extend(events)

        watch = EventWatch(ToolCallEvent.name == "search")
        watch.arm(stream, callback)

        await stream.send(ToolCallEvent(name="search", arguments="{}"), ctx)
        await stream.send(ToolCallEvent(name="other", arguments="{}"), ctx)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_disarm_stops_firing(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        received: list = []

        async def callback(events, _ctx):
            received.extend(events)

        watch = EventWatch(ToolCallEvent)
        watch.arm(stream, callback)
        watch.disarm()
        assert not watch.is_armed

        await stream.send(ToolCallEvent(name="search", arguments="{}"), ctx)
        assert len(received) == 0


class TestCadenceWatch:
    @pytest.mark.asyncio
    async def test_fires_after_n_events(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        batches: list = []

        async def callback(events, _ctx):
            batches.append(events)

        watch = CadenceWatch(n=3, condition=ToolCallEvent)
        watch.arm(stream, callback)

        for i in range(5):
            await stream.send(ToolCallEvent(name=f"t{i}", arguments="{}"), ctx)

        # 5 events, batch size 3 -> 1 batch of 3, 2 remaining
        assert len(batches) == 1
        assert len(batches[0]) == 3

    @pytest.mark.asyncio
    async def test_multiple_batches(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        batches: list = []

        async def callback(events, _ctx):
            batches.append(events)

        watch = CadenceWatch(n=2)
        watch.arm(stream, callback)

        for i in range(4):
            await stream.send(ModelMessage(content=f"m{i}"), ctx)

        assert len(batches) == 2

    @pytest.mark.asyncio
    async def test_disarm_clears_buffer(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        batches: list = []

        async def callback(events, _ctx):
            batches.append(events)

        watch = CadenceWatch(n=5)
        watch.arm(stream, callback)

        await stream.send(ModelMessage(content="m1"), ctx)
        watch.disarm()
        assert len(batches) == 0

    @pytest.mark.asyncio
    async def test_fires_on_timeout(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        batches: list = []

        async def callback(events, _ctx):
            batches.append(events)

        watch = CadenceWatch(max_wait=0.1, condition=ToolCallEvent)
        watch.arm(stream, callback)

        await stream.send(ToolCallEvent(name="t1", arguments="{}"), ctx)
        await stream.send(ToolCallEvent(name="t2", arguments="{}"), ctx)

        await asyncio.sleep(0.2)

        assert len(batches) == 1
        assert len(batches[0]) == 2

    @pytest.mark.asyncio
    async def test_n_wins_when_reached_before_timeout(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        batches: list = []

        async def callback(events, _ctx):
            batches.append(events)

        watch = CadenceWatch(n=2, max_wait=1.0)
        watch.arm(stream, callback)

        await stream.send(ModelMessage(content="m1"), ctx)
        await stream.send(ModelMessage(content="m2"), ctx)

        # Count trigger fires immediately; no need to wait for timeout
        assert len(batches) == 1
        assert len(batches[0]) == 2

    @pytest.mark.asyncio
    async def test_timeout_wins_when_n_not_reached(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        batches: list = []

        async def callback(events, _ctx):
            batches.append(events)

        watch = CadenceWatch(n=10, max_wait=0.1)
        watch.arm(stream, callback)

        await stream.send(ModelMessage(content="m1"), ctx)
        await asyncio.sleep(0.2)

        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_requires_at_least_one_trigger(self) -> None:
        with pytest.raises(ValueError, match="at least one of 'n' or 'max_wait'"):
            CadenceWatch()

    def test_rejects_non_positive_n(self) -> None:
        with pytest.raises(ValueError, match="'n' must be positive"):
            CadenceWatch(n=0)

    def test_rejects_non_positive_max_wait(self) -> None:
        with pytest.raises(ValueError, match="'max_wait' must be positive"):
            CadenceWatch(max_wait=0)

    @pytest.mark.asyncio
    async def test_count_trigger_cancels_pending_timer(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        batches: list = []

        async def callback(events, _ctx):
            batches.append(events)

        watch = CadenceWatch(n=2, max_wait=0.2)
        watch.arm(stream, callback)

        # Count trigger fires
        await stream.send(ModelMessage(content="m1"), ctx)
        await stream.send(ModelMessage(content="m2"), ctx)
        assert len(batches) == 1

        # Wait past max_wait; cancelled timer must not fire a phantom batch
        await asyncio.sleep(0.3)
        assert len(batches) == 1

    @pytest.mark.asyncio
    async def test_timer_restarts_after_count_flush(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        batches: list = []

        async def callback(events, _ctx):
            batches.append(events)

        watch = CadenceWatch(n=2, max_wait=0.1)
        watch.arm(stream, callback)

        # Count-trigger the first batch
        await stream.send(ModelMessage(content="m1"), ctx)
        await stream.send(ModelMessage(content="m2"), ctx)
        assert len(batches) == 1

        # A single follow-up event should start a fresh timer and flush on timeout
        await stream.send(ModelMessage(content="m3"), ctx)
        await asyncio.sleep(0.2)

        assert len(batches) == 2
        assert len(batches[1]) == 1

    @pytest.mark.asyncio
    async def test_events_during_slow_callback_are_not_stranded(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        delivered: list[str] = []
        callback_sleep = 0.15
        max_wait = 0.05

        async def slow_callback(events, _ctx):
            await asyncio.sleep(callback_sleep)
            delivered.extend(e.content for e in events)

        watch = CadenceWatch(n=5, max_wait=max_wait)
        watch.arm(stream, slow_callback)

        # Wave 1: count-trigger fires batch 1 (callback now awaiting).
        for i in range(1, 6):
            await stream.send(ModelMessage(content=f"m{i}"), ctx)

        # Wave 2: while batch 1's callback is in flight, queue 3 events.
        # max_wait elapses, _wait_and_fire fires batch 2 (also slow).
        await asyncio.sleep(0.01)
        for i in range(6, 9):
            await stream.send(ModelMessage(content=f"m{i}"), ctx)
        await asyncio.sleep(max_wait + 0.02)

        # Wave 3: lands while batch 2's callback is awaiting. Pre-fix, the
        # max_wait timer task is alive (in callback) so no fresh timer is
        # scheduled and these events sit in the buffer forever.
        for i in range(9, 12):
            await stream.send(ModelMessage(content=f"m{i}"), ctx)

        # Wait for all in-flight callbacks plus one more max_wait cycle.
        await asyncio.sleep(2 * callback_sleep + max_wait + 0.1)

        assert set(delivered) == {f"m{i}" for i in range(1, 12)}

    @pytest.mark.asyncio
    async def test_count_fires_after_timer_flush(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        batches: list = []

        async def callback(events, _ctx):
            batches.append(events)

        watch = CadenceWatch(n=3, max_wait=0.1)
        watch.arm(stream, callback)

        # Timer-trigger the first batch with a single event
        await stream.send(ModelMessage(content="m1"), ctx)
        await asyncio.sleep(0.2)
        assert len(batches) == 1
        assert len(batches[0]) == 1

        # Count trigger must still work on the next batch
        for i in range(3):
            await stream.send(ModelMessage(content=f"x{i}"), ctx)

        assert len(batches) == 2
        assert len(batches[1]) == 3


class TestIntervalWatch:
    @pytest.mark.asyncio
    async def test_fires_periodically(self) -> None:
        stream = MemoryStream()
        call_count = 0

        async def callback(events, _ctx):
            nonlocal call_count
            call_count += 1

        watch = IntervalWatch(0.05)
        watch.arm(stream, callback)
        assert watch.is_armed

        await asyncio.sleep(0.18)
        watch.disarm()
        assert not watch.is_armed
        # Should have fired ~3 times in 0.18s at 0.05s interval
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_disarm_stops_timer(self) -> None:
        stream = MemoryStream()
        call_count = 0

        async def callback(events, _ctx):
            nonlocal call_count
            call_count += 1

        watch = IntervalWatch(0.05)
        watch.arm(stream, callback)
        watch.disarm()

        await asyncio.sleep(0.15)
        assert call_count == 0


class TestDelayWatch:
    @pytest.mark.asyncio
    async def test_fires_once_after_delay(self) -> None:
        stream = MemoryStream()
        call_count = 0

        async def callback(events, _ctx):
            nonlocal call_count
            call_count += 1

        watch = DelayWatch(0.05)
        watch.arm(stream, callback)

        await asyncio.sleep(0.15)
        assert call_count == 1
        assert not watch.is_armed  # Auto-disarmed

    @pytest.mark.asyncio
    async def test_disarm_before_fire(self) -> None:
        stream = MemoryStream()
        call_count = 0

        async def callback(events, _ctx):
            nonlocal call_count
            call_count += 1

        watch = DelayWatch(0.1)
        watch.arm(stream, callback)
        watch.disarm()

        await asyncio.sleep(0.2)
        assert call_count == 0


class TestAllOf:
    @pytest.mark.asyncio
    async def test_fires_when_all_sub_watches_fired(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        received: list = []

        async def callback(events, _ctx):
            received.append(events)

        w = AllOf(
            EventWatch(ToolCallEvent),
            EventWatch(ModelMessage),
        )
        w.arm(stream, callback)

        # Fire only one — should not trigger
        await stream.send(ToolCallEvent(name="t", arguments="{}"), ctx)
        assert len(received) == 0

        # Fire the other — now both have fired
        await stream.send(ModelMessage(content="m"), ctx)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_collects_events_from_all_sub_watches(self) -> None:
        """AllOf should include events from ALL sub-watches in callback, not just the last."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        received: list = []

        async def callback(events, _ctx):
            received.append(events)

        w = AllOf(
            EventWatch(ToolCallEvent),
            EventWatch(ModelMessage),
        )
        w.arm(stream, callback)

        tool_event = ToolCallEvent(name="search", arguments="{}")
        msg_event = ModelMessage(content="hello")

        await stream.send(tool_event, ctx)
        await stream.send(msg_event, ctx)

        assert len(received) == 1
        combined = received[0]
        # Should contain events from both sub-watches
        assert len(combined) == 2
        assert tool_event in combined
        assert msg_event in combined

    @pytest.mark.asyncio
    async def test_resets_after_firing(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        received: list = []

        async def callback(events, _ctx):
            received.append(events)

        w = AllOf(
            EventWatch(ToolCallEvent),
            EventWatch(ModelMessage),
        )
        w.arm(stream, callback)

        # First cycle
        await stream.send(ToolCallEvent(name="t", arguments="{}"), ctx)
        await stream.send(ModelMessage(content="m"), ctx)
        assert len(received) == 1

        # Second cycle
        await stream.send(ToolCallEvent(name="t2", arguments="{}"), ctx)
        await stream.send(ModelMessage(content="m2"), ctx)
        assert len(received) == 2


class TestAnyOf:
    @pytest.mark.asyncio
    async def test_fires_on_either_watch(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        received: list = []

        async def callback(events, _ctx):
            received.append(events)

        w = AnyOf(
            EventWatch(ToolCallEvent),
            EventWatch(ModelMessage),
        )
        w.arm(stream, callback)

        await stream.send(ToolCallEvent(name="t", arguments="{}"), ctx)
        assert len(received) == 1

        await stream.send(ModelMessage(content="m"), ctx)
        assert len(received) == 2


class TestSequence:
    @pytest.mark.asyncio
    async def test_fires_in_order(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        received: list = []

        async def callback(events, _ctx):
            received.append(events)

        w = Sequence(
            EventWatch(ToolCallEvent),
            EventWatch(ModelMessage),
        )
        w.arm(stream, callback)

        # Wrong order — message first, then tool call → should not fire
        await stream.send(ModelMessage(content="m"), ctx)
        assert len(received) == 0

        # Right order — tool call (matches first watch)
        await stream.send(ToolCallEvent(name="t", arguments="{}"), ctx)
        assert len(received) == 0  # Only first step done

        # Second step — message
        await stream.send(ModelMessage(content="m2"), ctx)
        assert len(received) == 1  # Sequence complete

    @pytest.mark.asyncio
    async def test_resets_after_completion(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        received: list = []

        async def callback(events, _ctx):
            received.append(events)

        w = Sequence(
            EventWatch(ToolCallEvent),
            EventWatch(ModelMessage),
        )
        w.arm(stream, callback)

        # Complete first sequence
        await stream.send(ToolCallEvent(name="t1", arguments="{}"), ctx)
        await stream.send(ModelMessage(content="m1"), ctx)
        assert len(received) == 1

        # Complete second sequence
        await stream.send(ToolCallEvent(name="t2", arguments="{}"), ctx)
        await stream.send(ModelMessage(content="m2"), ctx)
        assert len(received) == 2


class TestCronWatchExpressions:
    def test_range_expression(self) -> None:
        cron = CronWatch("1-5 * * * *")
        now = datetime.datetime(2026, 3, 22, 10, 0, 0)
        next_fire = cron._next_fire_time(now)
        assert next_fire.minute in {1, 2, 3, 4, 5}

    def test_list_expression(self) -> None:
        cron = CronWatch("0,15,30,45 * * * *")
        now = datetime.datetime(2026, 3, 22, 10, 0, 0)
        next_fire = cron._next_fire_time(now)
        assert next_fire.minute in {0, 15, 30, 45}

    def test_step_with_range(self) -> None:
        cron = CronWatch("*/10 * * * *")
        now = datetime.datetime(2026, 3, 22, 10, 0, 0)
        next_fire = cron._next_fire_time(now)
        assert next_fire.minute in {0, 10, 20, 30, 40, 50}

    def test_specific_hour_and_minute(self) -> None:
        cron = CronWatch("30 14 * * *")
        now = datetime.datetime(2026, 3, 22, 10, 0, 0)
        next_fire = cron._next_fire_time(now)
        assert next_fire.hour == 14
        assert next_fire.minute == 30

    def test_invalid_field_count_raises(self) -> None:
        cron = CronWatch("* * *")
        with pytest.raises(ValueError, match="5 fields"):
            cron._next_fire_time(datetime.datetime.now())

    def test_step_five_minutes(self) -> None:
        cron = CronWatch("*/5 * * * *")
        now = datetime.datetime(2026, 3, 21, 10, 3, 0)
        nxt = cron._next_fire_time(now)
        assert nxt.minute == 5
        assert nxt.hour == 10

    def test_day_of_week_name(self) -> None:
        cron = CronWatch("0 9 * * MON")
        # 2026-03-21 is a Saturday; next Monday is 2026-03-23
        now = datetime.datetime(2026, 3, 21, 10, 0, 0)
        nxt = cron._next_fire_time(now)
        assert nxt.weekday() == 0
        assert nxt.hour == 9
        assert nxt.minute == 0

    def test_invalid_expression_raises(self) -> None:
        cron = CronWatch("bad")
        with pytest.raises(ValueError, match="Invalid cron"):
            cron._next_fire_time(datetime.datetime.now())

    def test_numeric_dow_sunday_zero(self) -> None:
        cron = CronWatch("0 9 * * 0")
        # 2026-03-21 is a Saturday; next Sunday is 2026-03-22
        now = datetime.datetime(2026, 3, 21, 10, 0, 0)
        nxt = cron._next_fire_time(now)
        assert nxt.isoweekday() % 7 == 0
        assert nxt.day == 22
        assert nxt.hour == 9

    def test_numeric_dow_saturday_six(self) -> None:
        cron = CronWatch("0 9 * * 6")
        # 2026-03-21 (Saturday) at 10:00 — past 9:00; next Saturday is 2026-03-28
        now = datetime.datetime(2026, 3, 21, 10, 0, 0)
        nxt = cron._next_fire_time(now)
        assert nxt.isoweekday() % 7 == 6
        assert nxt.day == 28
        assert nxt.hour == 9

    def test_numeric_dow_seven_is_sunday_alias(self) -> None:
        watch_seven = CronWatch("0 9 * * 7")
        watch_zero = CronWatch("0 9 * * 0")
        now = datetime.datetime(2026, 3, 21, 10, 0, 0)
        assert watch_seven._next_fire_time(now) == watch_zero._next_fire_time(now)
