# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from autogen.beta.context import ConversationContext as Context
from autogen.beta.events import ModelMessage, ToolCallEvent
from autogen.beta.stream import MemoryStream
from autogen.beta.watch import (
    AllOf,
    AnyOf,
    BatchWatch,
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


class TestBatchWatch:
    @pytest.mark.asyncio
    async def test_fires_after_n_events(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        batches: list = []

        async def callback(events, _ctx):
            batches.append(events)

        watch = BatchWatch(3, condition=ToolCallEvent)
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

        watch = BatchWatch(2)
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

        watch = BatchWatch(5)
        watch.arm(stream, callback)

        await stream.send(ModelMessage(content="m1"), ctx)
        watch.disarm()
        assert len(batches) == 0


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
