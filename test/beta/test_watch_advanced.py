# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from dirty_equals import IsList

from autogen.beta import Agent, Context, MemoryStream, testing
from autogen.beta.events import BaseEvent, ToolCallEvent
from autogen.beta.watch import CadenceWatch, CronWatch


@pytest.mark.asyncio
class TestCadenceWatchTimeTrigger:
    async def test_collects_events_in_window(self) -> None:
        # arrange stream
        stream = MemoryStream()
        batches: list[BaseEvent] = []

        async def callback(events: BaseEvent, ctx: Context) -> None:
            batches.append(events)

        watch = CadenceWatch(max_wait=0.01, condition=ToolCallEvent)
        watch.arm(stream, callback)

        # arrange agent
        tool_calls = [
            ToolCallEvent(name="t1", arguments="{}"),
            ToolCallEvent(name="t2", arguments="{}"),
        ]

        agent = Agent(
            "test-agent",
            config=testing.TestConfig(tool_calls, "Done"),
        )

        @agent.tool
        def t1(): ...
        @agent.tool
        def t2(): ...

        # act
        await agent.ask("Hello, world!", stream=stream)
        await asyncio.sleep(0.02)

        # assert
        assert batches == [
            IsList(*tool_calls, check_order=False),
        ]

    async def test_ignores_non_matching(self) -> None:
        stream = MemoryStream()
        batches: list = []

        async def callback(events, ctx):
            batches.append(events)

        watch = CadenceWatch(max_wait=0.01, condition=ToolCallEvent)
        watch.arm(stream, callback)

        agent = Agent("test-agent", config=testing.TestConfig("Done"))
        await agent.ask("Hello", stream=stream)
        await asyncio.sleep(0.02)

        assert len(batches) == 0

    async def test_disarm_cancels_timer(self) -> None:
        stream = MemoryStream()
        batches: list = []

        async def callback(events, ctx):
            batches.append(events)

        watch = CadenceWatch(max_wait=0.01)
        watch.arm(stream, callback)
        watch.disarm()

        agent = Agent("test-agent", config=testing.TestConfig("Done"))
        await agent.ask("Hello", stream=stream)
        await asyncio.sleep(0.02)

        assert len(batches) == 0

    def test_disarm_resets_armed_flag(self) -> None:
        stream = MemoryStream()

        async def callback(events, ctx):
            pass

        watch = CadenceWatch(max_wait=1.0)
        watch.arm(stream, callback)
        assert watch.is_armed

        watch.disarm()
        assert not watch.is_armed


@pytest.mark.asyncio
async def test_cron_watch_arm_and_disarm() -> None:
    stream = MemoryStream()

    async def callback(events, ctx):
        pass

    watch = CronWatch("0 9 * * MON")
    watch.arm(stream, callback)
    assert watch.is_armed

    watch.disarm()
    assert not watch.is_armed
