# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from autogen.beta.context import Context
from autogen.beta.network.scheduler import Scheduler, WatchStatus
from autogen.beta.network.primitives.watch import DelayWatch, EventWatch, IntervalWatch


class TestSchedulerStandalone:
    @pytest.mark.asyncio
    async def test_add_and_list_watches(self) -> None:
        scheduler = Scheduler()
        w1 = scheduler.add(IntervalWatch(60), callback=self._noop)
        w2 = scheduler.add(DelayWatch(30), callback=self._noop)

        watches = scheduler.watches
        assert len(watches) == 2
        assert watches[0][0] == w1
        assert watches[1][0] == w2

    @pytest.mark.asyncio
    async def test_start_arms_watches(self) -> None:
        scheduler = Scheduler()
        scheduler.add(IntervalWatch(999), callback=self._noop)

        await scheduler.start()
        watches = scheduler.watches
        assert watches[0][2] == WatchStatus.ARMED

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_pauses_watches(self) -> None:
        scheduler = Scheduler()
        scheduler.add(IntervalWatch(999), callback=self._noop)

        await scheduler.start()
        await scheduler.stop()

        watches = scheduler.watches
        assert watches[0][2] == WatchStatus.PAUSED

    @pytest.mark.asyncio
    async def test_pause_and_resume(self) -> None:
        scheduler = Scheduler()
        wid = scheduler.add(IntervalWatch(999), callback=self._noop)

        await scheduler.start()
        scheduler.pause(wid)
        assert scheduler.watches[0][2] == WatchStatus.PAUSED

        scheduler.resume(wid)
        assert scheduler.watches[0][2] == WatchStatus.ARMED

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_cancel_removes_watch(self) -> None:
        scheduler = Scheduler()
        wid = scheduler.add(IntervalWatch(999), callback=self._noop)

        result = scheduler.cancel(wid)
        assert result is True
        assert scheduler.status(wid) == WatchStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_returns_false(self) -> None:
        scheduler = Scheduler()
        result = scheduler.cancel("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_callback_fires(self) -> None:
        scheduler = Scheduler()
        received: list = []

        async def callback(events, ctx):
            received.append(True)

        scheduler.add(IntervalWatch(0.05), callback=callback)
        await scheduler.start()

        await asyncio.sleep(0.15)
        await scheduler.stop()

        assert len(received) >= 2

    @pytest.mark.asyncio
    async def test_delay_watch_fires_once(self) -> None:
        scheduler = Scheduler()
        received: list = []

        async def callback(events, ctx):
            received.append(True)

        scheduler.add(DelayWatch(0.05), callback=callback)
        await scheduler.start()

        await asyncio.sleep(0.2)
        await scheduler.stop()

        assert len(received) == 1

    @staticmethod
    async def _noop(events, ctx):
        pass


class TestSchedulerWithAddAfterStart:
    @pytest.mark.asyncio
    async def test_add_after_start_arms_immediately(self) -> None:
        scheduler = Scheduler()
        await scheduler.start()

        received: list = []

        async def callback(events, ctx):
            received.append(True)

        scheduler.add(IntervalWatch(0.05), callback=callback)

        await asyncio.sleep(0.15)
        await scheduler.stop()

        assert len(received) >= 2
