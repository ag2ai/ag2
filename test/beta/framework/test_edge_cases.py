# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Framework-core edge cases extracted from the former
``test/beta/network/test_edge_cases.py``.

The original file was a grab-bag that exercised both framework-core and
V2-network code paths. This file preserves the framework-core portions —
Observer exception handling and CronWatch expression parsing. The
V2-network edge cases (multi-hop delegation, topology reroute, concurrent
delegation, self-delegation, Network convenience cleanup) were dropped
with the rest of V2 and will get reinstated against V3 types in Phase 2 /
Phase 4.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import ExitStack

import pytest

from autogen.beta import BaseObserver
from autogen.beta.context import ConversationContext as Context
from autogen.beta.events import ModelMessage
from autogen.beta.events.alert import ObserverAlert
from autogen.beta.stream import MemoryStream
from autogen.beta.watch import CronWatch, EventWatch


class _CrashingObserver(BaseObserver):
    """Observer whose ``process()`` always raises."""

    def __init__(self) -> None:
        super().__init__("crasher", watch=EventWatch(ModelMessage))

    async def process(self, events, ctx):
        raise RuntimeError("observer exploded")


class TestObserverExceptionHandling:
    @pytest.mark.asyncio
    async def test_observer_process_exception_is_caught(self, caplog) -> None:
        observer = _CrashingObserver()
        stream = MemoryStream()
        ctx = Context(stream=stream)

        with ExitStack() as stack:
            observer.register(stack, ctx)

            with caplog.at_level(logging.ERROR):
                await stream.send(ModelMessage(content="trigger"), ctx)
                await asyncio.sleep(0.01)

        assert any("process() failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_observer_returns_none_no_signal(self) -> None:
        from autogen.beta.annotations import Context as AnnContext

        class _NullObserver(BaseObserver):
            def __init__(self) -> None:
                super().__init__("null", watch=EventWatch(ModelMessage))

            async def process(self, events, ctx):
                return None

        observer = _NullObserver()
        stream = MemoryStream()
        ctx = Context(stream=stream)

        signals: list[ObserverAlert] = []

        async def _capture(event: ObserverAlert, _ctx: AnnContext) -> None:
            signals.append(event)

        from autogen.beta.events.conditions import TypeCondition

        stream.subscribe(_capture, condition=TypeCondition(ObserverAlert))

        with ExitStack() as stack:
            observer.register(stack, ctx)
            await stream.send(ModelMessage(content="test"), ctx)
            await asyncio.sleep(0.01)

        assert len(signals) == 0


class TestCronWatchExpressions:
    def test_range_expression(self) -> None:
        import datetime

        cron = CronWatch("1-5 * * * *")
        now = datetime.datetime(2026, 3, 22, 10, 0, 0)
        next_fire = cron._next_fire_time(now)
        assert next_fire.minute in {1, 2, 3, 4, 5}

    def test_list_expression(self) -> None:
        import datetime

        cron = CronWatch("0,15,30,45 * * * *")
        now = datetime.datetime(2026, 3, 22, 10, 0, 0)
        next_fire = cron._next_fire_time(now)
        assert next_fire.minute in {0, 15, 30, 45}

    def test_step_with_range(self) -> None:
        import datetime

        cron = CronWatch("*/10 * * * *")
        now = datetime.datetime(2026, 3, 22, 10, 0, 0)
        next_fire = cron._next_fire_time(now)
        assert next_fire.minute in {0, 10, 20, 30, 40, 50}

    def test_specific_hour_and_minute(self) -> None:
        import datetime

        cron = CronWatch("30 14 * * *")
        now = datetime.datetime(2026, 3, 22, 10, 0, 0)
        next_fire = cron._next_fire_time(now)
        assert next_fire.hour == 14
        assert next_fire.minute == 30

    def test_invalid_field_count_raises(self) -> None:
        import datetime

        cron = CronWatch("* * *")
        with pytest.raises(ValueError, match="5 fields"):
            cron._next_fire_time(datetime.datetime.now())
