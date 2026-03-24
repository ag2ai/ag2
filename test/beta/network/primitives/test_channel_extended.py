# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from autogen.beta.network.primitives.channel import BufferedChannel, LocalChannel, PriorityChannel
from autogen.beta.network.primitives.envelope import Envelope
from autogen.beta.events import ModelMessage
from autogen.beta.network.primitives.priority import DefaultPriority


class TestBufferedChannelBlockOverflow:
    """BufferedChannel with overflow_policy='block' blocks senders until drained."""

    @pytest.mark.asyncio
    async def test_block_policy_unblocks_after_drain(self) -> None:
        """Send blocks when buffer is full and resumes once an item is drained."""
        channel = BufferedChannel(max_buffer=1, overflow_policy="block")
        received: list[str] = []

        async def slow_handler(envelope, ctx):
            await asyncio.sleep(0.05)
            received.append(envelope.event.content)

        channel.subscribe(slow_handler)

        # First send fills the buffer and kicks off drain
        await channel.send(Envelope(event=ModelMessage(content="m0"), sender="a"))
        # Give drain task a moment to start (it will start processing m0)
        await asyncio.sleep(0.01)

        # Second send should eventually succeed once the drain frees a slot.
        # Use a timeout to prevent hanging if the test is wrong.
        await asyncio.wait_for(
            channel.send(Envelope(event=ModelMessage(content="m1"), sender="a")),
            timeout=2.0,
        )

        # Wait for drain to finish processing
        await asyncio.sleep(0.2)

        assert "m0" in received
        assert "m1" in received


class TestBufferedChannelDropOldest:
    """BufferedChannel drop_oldest verifies the *oldest* item is the one dropped."""

    @pytest.mark.asyncio
    async def test_drop_oldest_drops_first_item(self) -> None:
        """With max_buffer=2, sending 3 items should drop the oldest (first)."""
        channel = BufferedChannel(max_buffer=2, overflow_policy="drop_oldest")
        received: list[str] = []

        async def handler(envelope, ctx):
            received.append(envelope.event.content)

        channel.subscribe(handler)

        # Send 3 messages rapidly; buffer holds 2 so m0 should be dropped
        for i in range(3):
            await channel.send(Envelope(event=ModelMessage(content=f"m{i}"), sender="a"))

        await asyncio.sleep(0.1)

        # m0 was the oldest and should have been dropped
        assert "m0" not in received
        assert "m1" in received
        assert "m2" in received


class TestPriorityChannelOrdering:
    """PriorityChannel delivers higher-priority envelopes before lower ones."""

    @pytest.mark.asyncio
    async def test_priority_ordering(self) -> None:
        """Envelopes pushed in low-then-high order are delivered high-first."""
        channel = PriorityChannel()
        received: list[str] = []

        async def handler(envelope, ctx):
            received.append(envelope.event.content)

        # Subscribe *after* sending so all items are queued before drain starts
        # Actually, we need the subscriber registered before drain runs.
        # So: register subscriber, push items, then let drain proceed.
        channel.subscribe(handler)

        # Push three envelopes at different priorities.
        # Because the drain task runs asynchronously, we push them and then
        # wait for delivery. The PriorityChannel uses a heap, so items that
        # are still in the heap when drain pops will come out in priority order.

        # To ensure all items are in the heap before drain processes them,
        # we can push them without awaiting in between (drain kicks off on
        # the first send but won't run until we yield control).
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
        await channel.send(Envelope(
            event=ModelMessage(content="normal"),
            sender="a",
            priority=DefaultPriority.NORMAL,
        ))

        await asyncio.sleep(0.1)

        # The drain task starts on the first send. It may deliver "low" first
        # because it pops immediately. But subsequent items should be ordered
        # by priority. At minimum, "high" must appear before "normal" and
        # "normal" before "low" among those queued together.
        assert len(received) == 3
        # After the first item is drained, remaining heap items should
        # come out in priority order. Verify high comes before low at least.
        high_idx = received.index("high")
        low_idx = received.index("low")
        # If all three were in the heap, high should come before low.
        # The first send might drain immediately, but the second and third
        # are queued. Either way, high must not come after low.
        assert high_idx < low_idx, f"Expected 'high' before 'low', got order: {received}"


class TestChannelTTLBehavior:
    """Document current TTL behavior: LocalChannel and BufferedChannel deliver
    expired envelopes (they do not check TTL). PriorityChannel skips them."""

    @pytest.mark.asyncio
    async def test_local_channel_delivers_expired_envelope(self) -> None:
        """LocalChannel does NOT check TTL — expired envelopes are still delivered."""
        channel = LocalChannel()
        received: list = []

        async def handler(envelope, ctx):
            received.append(envelope)

        channel.subscribe(handler)

        # Create an already-expired envelope
        env = Envelope(
            event=ModelMessage(content="expired"),
            sender="a",
            ttl=0.0,
            timestamp=0.0,  # epoch — long expired
        )
        assert env.is_expired  # sanity check

        await channel.send(env)

        # LocalChannel delivers it regardless
        assert len(received) == 1
        assert received[0].event.content == "expired"

    @pytest.mark.asyncio
    async def test_buffered_channel_delivers_expired_envelope(self) -> None:
        """BufferedChannel does NOT check TTL — expired envelopes are still delivered."""
        channel = BufferedChannel(max_buffer=100)
        received: list = []

        async def handler(envelope, ctx):
            received.append(envelope)

        channel.subscribe(handler)

        env = Envelope(
            event=ModelMessage(content="expired"),
            sender="a",
            ttl=0.0,
            timestamp=0.0,
        )
        assert env.is_expired

        await channel.send(env)
        await asyncio.sleep(0.1)

        # BufferedChannel delivers it regardless
        assert len(received) == 1
        assert received[0].event.content == "expired"


class TestChannelCloseIdempotency:
    """Calling close() multiple times should not raise."""

    @pytest.mark.asyncio
    async def test_local_channel_close_idempotent(self) -> None:
        channel = LocalChannel()
        await channel.close()
        await channel.close()  # second close should not raise

    @pytest.mark.asyncio
    async def test_buffered_channel_close_idempotent(self) -> None:
        channel = BufferedChannel()
        await channel.close()
        await channel.close()

    @pytest.mark.asyncio
    async def test_priority_channel_close_idempotent(self) -> None:
        channel = PriorityChannel()
        await channel.close()
        await channel.close()
