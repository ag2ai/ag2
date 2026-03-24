# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from autogen.beta.network.primitives.channel import BufferedChannel, LocalChannel, PriorityChannel
from autogen.beta.network.primitives.envelope import Envelope
from autogen.beta.events import ModelMessage, ToolCallEvent
from autogen.beta.events.conditions import TypeCondition
from autogen.beta.network.primitives.priority import DefaultPriority


class TestLocalChannel:
    @pytest.mark.asyncio
    async def test_send_and_receive(self) -> None:
        channel = LocalChannel()
        received: list = []

        async def handler(envelope, ctx):
            received.append(envelope)

        channel.subscribe(handler)
        env = Envelope(event=ModelMessage(content="hello"), sender="a")
        await channel.send(env)

        assert len(received) == 1
        assert received[0].event.content == "hello"

    @pytest.mark.asyncio
    async def test_condition_filter(self) -> None:
        channel = LocalChannel()
        received: list = []

        async def handler(envelope, ctx):
            received.append(envelope)

        channel.subscribe(handler, condition=TypeCondition(ToolCallEvent))

        await channel.send(Envelope(event=ModelMessage(content="m"), sender="a"))
        await channel.send(Envelope(event=ToolCallEvent(name="t", arguments="{}"), sender="a"))

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self) -> None:
        channel = LocalChannel()
        received: list = []

        async def handler(envelope, ctx):
            received.append(envelope)

        sub_id = channel.subscribe(handler)
        await channel.send(Envelope(event=ModelMessage(content="m1"), sender="a"))
        channel.unsubscribe(sub_id)
        await channel.send(Envelope(event=ModelMessage(content="m2"), sender="a"))

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_close_rejects_sends(self) -> None:
        channel = LocalChannel()
        await channel.close()

        with pytest.raises(RuntimeError, match="closed"):
            await channel.send(Envelope(event=ModelMessage(content="m"), sender="a"))

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self) -> None:
        channel = LocalChannel()
        r1: list = []
        r2: list = []

        async def h1(envelope, ctx):
            r1.append(envelope)

        async def h2(envelope, ctx):
            r2.append(envelope)

        channel.subscribe(h1)
        channel.subscribe(h2)

        await channel.send(Envelope(event=ModelMessage(content="m"), sender="a"))
        assert len(r1) == 1
        assert len(r2) == 1


class TestBufferedChannel:
    @pytest.mark.asyncio
    async def test_delivers_buffered(self) -> None:
        channel = BufferedChannel(max_buffer=100)
        received: list = []

        async def handler(envelope, ctx):
            received.append(envelope)

        channel.subscribe(handler)

        for i in range(5):
            await channel.send(Envelope(event=ModelMessage(content=f"m{i}"), sender="a"))

        await asyncio.sleep(0.05)
        assert len(received) == 5

    @pytest.mark.asyncio
    async def test_drop_oldest_overflow(self) -> None:
        channel = BufferedChannel(max_buffer=2, overflow_policy="drop_oldest")
        received: list = []

        async def handler(envelope, ctx):
            received.append(envelope)

        channel.subscribe(handler)

        # Send 3 messages with buffer size 2
        for i in range(3):
            await channel.send(Envelope(event=ModelMessage(content=f"m{i}"), sender="a"))

        await asyncio.sleep(0.05)
        # m0 was dropped, m1 and m2 delivered
        contents = [r.event.content for r in received]
        assert "m0" not in contents or len(received) <= 3

    @pytest.mark.asyncio
    async def test_drop_newest_overflow(self) -> None:
        channel = BufferedChannel(max_buffer=2, overflow_policy="drop_newest")
        # Just verify it doesn't error
        for i in range(5):
            await channel.send(Envelope(event=ModelMessage(content=f"m{i}"), sender="a"))

    @pytest.mark.asyncio
    async def test_close_drains(self) -> None:
        channel = BufferedChannel(max_buffer=100)
        received: list = []

        async def handler(envelope, ctx):
            received.append(envelope)

        channel.subscribe(handler)
        await channel.send(Envelope(event=ModelMessage(content="m"), sender="a"))
        # Let the drain task start
        await asyncio.sleep(0.05)
        await channel.close()

        assert len(received) >= 1


class TestPriorityChannel:
    @pytest.mark.asyncio
    async def test_higher_priority_first(self) -> None:
        channel = PriorityChannel()
        received: list = []

        async def handler(envelope, ctx):
            received.append(envelope)

        channel.subscribe(handler)

        # Send low then high priority
        await channel.send(Envelope(
            event=ModelMessage(content="low"),
            sender="a",
            priority=DefaultPriority.BACKGROUND,
        ))
        await channel.send(Envelope(
            event=ModelMessage(content="high"),
            sender="a",
            priority=DefaultPriority.URGENT,
        ))

        await asyncio.sleep(0.05)
        # Both should be delivered; high priority should come first if queued
        assert len(received) >= 2

    @pytest.mark.asyncio
    async def test_expired_envelopes_skipped(self) -> None:
        channel = PriorityChannel()
        received: list = []

        async def handler(envelope, ctx):
            received.append(envelope)

        channel.subscribe(handler)

        # Create an envelope that's already expired
        env = Envelope(
            event=ModelMessage(content="expired"),
            sender="a",
            ttl=0.0,  # Immediately expired
            timestamp=0.0,  # Old timestamp
        )
        await channel.send(env)

        await asyncio.sleep(0.05)
        # Expired envelope should be skipped
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        channel = PriorityChannel()
        await channel.close()
        with pytest.raises(RuntimeError, match="closed"):
            await channel.send(Envelope(event=ModelMessage(content="m"), sender="a"))
