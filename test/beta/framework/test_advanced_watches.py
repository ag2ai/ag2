# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from autogen.beta.context import ConversationContext as Context
from autogen.beta.events import ModelMessage, ToolCallEvent
from autogen.beta.stream import MemoryStream
from autogen.beta.watch import CronWatch, WindowWatch


class TestWindowWatch:
    @pytest.mark.asyncio
    async def test_collects_events_in_window(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        batches: list = []

        async def callback(events, _ctx):
            batches.append(events)

        watch = WindowWatch(0.1, condition=ToolCallEvent)
        watch.arm(stream, callback)

        # Send events within the window
        await stream.send(ToolCallEvent(name="t1", arguments="{}"), ctx)
        await stream.send(ToolCallEvent(name="t2", arguments="{}"), ctx)

        # Wait for window to flush
        await asyncio.sleep(0.2)

        assert len(batches) == 1
        assert len(batches[0]) == 2

    @pytest.mark.asyncio
    async def test_ignores_non_matching(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        batches: list = []

        async def callback(events, _ctx):
            batches.append(events)

        watch = WindowWatch(0.1, condition=ToolCallEvent)
        watch.arm(stream, callback)

        # ModelMessage doesn't match
        await stream.send(ModelMessage(content="hello"), ctx)
        await asyncio.sleep(0.2)

        assert len(batches) == 0

    @pytest.mark.asyncio
    async def test_disarm_cancels_timer(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        batches: list = []

        async def callback(events, _ctx):
            batches.append(events)

        watch = WindowWatch(0.1)
        watch.arm(stream, callback)

        await stream.send(ModelMessage(content="hello"), ctx)
        watch.disarm()

        await asyncio.sleep(0.2)
        assert len(batches) == 0

    def test_disarm_resets_armed_flag(self) -> None:
        """WindowWatch.disarm() must set is_armed to False."""
        stream = MemoryStream()

        async def callback(events, ctx):
            pass

        watch = WindowWatch(1.0)
        watch.arm(stream, callback)
        assert watch.is_armed

        watch.disarm()
        assert not watch.is_armed


class TestCronWatch:
    def test_next_fire_time_basic(self) -> None:
        import datetime

        watch = CronWatch("*/5 * * * *")  # Every 5 minutes
        now = datetime.datetime(2026, 3, 21, 10, 3, 0)
        nxt = watch._next_fire_time(now)

        # Next 5-minute mark after 10:03 should be 10:05
        assert nxt.minute == 5
        assert nxt.hour == 10

    def test_next_fire_time_day_of_week(self) -> None:
        import datetime

        watch = CronWatch("0 9 * * MON")  # Monday at 9:00
        # 2026-03-21 is a Saturday
        now = datetime.datetime(2026, 3, 21, 10, 0, 0)
        nxt = watch._next_fire_time(now)

        # Next Monday is 2026-03-23
        assert nxt.weekday() == 0  # Monday
        assert nxt.hour == 9
        assert nxt.minute == 0

    def test_invalid_expression_raises(self) -> None:
        watch = CronWatch("bad")
        import datetime

        with pytest.raises(ValueError, match="Invalid cron"):
            watch._next_fire_time(datetime.datetime.now())

    def test_numeric_dow_sunday_zero(self) -> None:
        """Numeric 0 = Sunday in standard cron."""
        import datetime

        watch = CronWatch("0 9 * * 0")
        # 2026-03-21 is a Saturday
        now = datetime.datetime(2026, 3, 21, 10, 0, 0)
        nxt = watch._next_fire_time(now)

        # Next Sunday is 2026-03-22
        assert nxt.isoweekday() % 7 == 0  # Sunday
        assert nxt.day == 22
        assert nxt.hour == 9

    def test_numeric_dow_saturday_six(self) -> None:
        """Numeric 6 = Saturday in standard cron."""
        import datetime

        watch = CronWatch("0 9 * * 6")
        # 2026-03-21 (Saturday) at 10:00 — already past 9:00
        now = datetime.datetime(2026, 3, 21, 10, 0, 0)
        nxt = watch._next_fire_time(now)

        # Next Saturday at 9:00 is 2026-03-28
        assert nxt.isoweekday() % 7 == 6  # Saturday
        assert nxt.day == 28
        assert nxt.hour == 9

    def test_numeric_dow_seven_means_sunday(self) -> None:
        """Numeric 7 is an alias for Sunday (0) in standard cron."""
        import datetime

        watch_seven = CronWatch("0 9 * * 7")
        watch_zero = CronWatch("0 9 * * 0")
        now = datetime.datetime(2026, 3, 21, 10, 0, 0)

        nxt_seven = watch_seven._next_fire_time(now)
        nxt_zero = watch_zero._next_fire_time(now)

        # Both should resolve to the same Sunday
        assert nxt_seven == nxt_zero

    @pytest.mark.asyncio
    async def test_arm_and_disarm(self) -> None:
        stream = MemoryStream()

        async def callback(events, ctx):
            pass

        watch = CronWatch("0 9 * * MON")
        watch.arm(stream, callback)
        assert watch.is_armed

        watch.disarm()
        assert not watch.is_armed
