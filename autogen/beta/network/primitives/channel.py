# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Channel primitives — typed event transport with delivery semantics.

A Channel abstracts over how Envelopes move between actors. It sits at the
network level and provides delivery guarantees appropriate to the backend.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import uuid4

logger = logging.getLogger(__name__)

from autogen.beta.context import Context, SubId
from autogen.beta.events.conditions import Condition
from autogen.beta.stream import MemoryStream

from .envelope import Envelope
from .priority import DefaultPriority, DefaultPriorityScheme, PriorityScheme

ChannelCallback = Callable[[Envelope, Context], Awaitable[None]]


@runtime_checkable
class Channel(Protocol):
    """Transport layer for envelopes between actors."""

    async def send(self, envelope: Envelope) -> None:
        """Send an envelope. Delivery semantics depend on implementation."""
        ...

    def subscribe(
        self,
        callback: ChannelCallback,
        *,
        condition: Condition | None = None,
    ) -> SubId:
        """Subscribe to incoming envelopes, optionally filtered."""
        ...

    def unsubscribe(self, sub_id: SubId) -> None: ...

    async def close(self) -> None:
        """Graceful shutdown. Flush pending, stop accepting."""
        ...


class LocalChannel:
    """In-process channel. Ordered delivery. At-most-once semantics.

    Wraps MemoryStream. Envelope.event is extracted and sent on the stream.
    Envelope metadata is available to subscribers. This introduces zero risk
    to existing core behavior.

    Example::

        channel = LocalChannel()
        channel.subscribe(my_handler)
        await channel.send(envelope)
    """

    def __init__(self) -> None:
        self._stream = MemoryStream()
        self._subscribers: dict[SubId, tuple[Condition | None, ChannelCallback]] = {}
        self._closed = False

    async def send(self, envelope: Envelope) -> None:
        if self._closed:
            raise RuntimeError("Channel is closed")

        # Deliver to subscribers
        ctx = Context(stream=self._stream)
        for condition, callback in tuple(self._subscribers.values()):
            if condition and not condition(envelope.event):
                continue
            await callback(envelope, ctx)

        # Also send the raw event on the stream for stream-level subscribers
        await self._stream.send(envelope.event, ctx)

    def subscribe(
        self,
        callback: ChannelCallback,
        *,
        condition: Condition | None = None,
    ) -> SubId:
        sub_id = uuid4()
        self._subscribers[sub_id] = (condition, callback)
        return sub_id

    def unsubscribe(self, sub_id: SubId) -> None:
        self._subscribers.pop(sub_id, None)

    async def close(self) -> None:
        self._closed = True
        self._subscribers.clear()


class BufferedChannel:
    """Bounded buffer with configurable backpressure policy.

    When observers can't keep up, the buffer absorbs bursts.

    Example::

        channel = BufferedChannel(max_buffer=1000, overflow_policy="drop_oldest")
    """

    def __init__(
        self,
        max_buffer: int = 1000,
        overflow_policy: Literal["drop_oldest", "drop_newest", "block"] = "drop_oldest",
    ) -> None:
        self._max_buffer = max_buffer
        self._overflow_policy = overflow_policy
        self._buffer: deque[Envelope] = deque()
        self._subscribers: dict[SubId, tuple[Condition | None, ChannelCallback]] = {}
        self._closed = False
        self._drain_task: asyncio.Task[Any] | None = None
        self._not_full = asyncio.Condition()  # Used by "block" policy
        self._shared_stream = MemoryStream()  # Shared context stream

    async def send(self, envelope: Envelope) -> None:
        if self._closed:
            raise RuntimeError("Channel is closed")

        if len(self._buffer) >= self._max_buffer:
            if self._overflow_policy == "drop_oldest":
                self._buffer.popleft()
            elif self._overflow_policy == "drop_newest":
                return  # Drop the incoming envelope
            elif self._overflow_policy == "block":
                async with self._not_full:
                    while len(self._buffer) >= self._max_buffer:
                        await self._not_full.wait()

        self._buffer.append(envelope)

        # Start drain loop if not running
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.ensure_future(self._drain())

    async def _drain(self) -> None:
        """Process buffered envelopes."""
        while self._buffer and not self._closed:
            envelope = self._buffer.popleft()
            # Notify blocked senders
            async with self._not_full:
                self._not_full.notify()
            ctx = Context(stream=self._shared_stream)
            for condition, callback in tuple(self._subscribers.values()):
                if condition and not condition(envelope.event):
                    continue
                await callback(envelope, ctx)

    def subscribe(
        self,
        callback: ChannelCallback,
        *,
        condition: Condition | None = None,
    ) -> SubId:
        sub_id = uuid4()
        self._subscribers[sub_id] = (condition, callback)
        return sub_id

    def unsubscribe(self, sub_id: SubId) -> None:
        self._subscribers.pop(sub_id, None)

    async def close(self) -> None:
        self._closed = True
        # Drain remaining
        await self._drain()
        self._subscribers.clear()
        if self._drain_task and not self._drain_task.done():
            self._drain_task.cancel()


class PriorityChannel:
    """In-process channel with priority-ordered delivery.

    Uses a heap to deliver higher-priority envelopes first. Envelopes
    without an explicit priority use ``default_priority`` (defaults to
    ``DefaultPriority.NORMAL``).  Pass a different value when using a
    custom priority scheme so that un-tagged envelopes sort correctly.

    Example::

        channel = PriorityChannel(scheme=DefaultPriorityScheme())

        # With a custom scheme whose "normal" baseline is 10:
        channel = PriorityChannel(scheme=my_scheme, default_priority=10)
    """

    def __init__(
        self,
        scheme: PriorityScheme | None = None,
        default_priority: Any = None,
    ) -> None:
        self._scheme = scheme or DefaultPriorityScheme()
        self._default_priority = default_priority if default_priority is not None else DefaultPriority.NORMAL
        self._heap: list[tuple[int, int, Envelope]] = []  # (neg_priority, sequence, envelope)
        self._seq = 0
        self._subscribers: dict[SubId, tuple[Condition | None, ChannelCallback]] = {}
        self._closed = False
        self._drain_task: asyncio.Task[Any] | None = None
        self._event = asyncio.Event()
        self._shared_stream = MemoryStream()  # Shared context stream

    async def send(self, envelope: Envelope) -> None:
        if self._closed:
            raise RuntimeError("Channel is closed")

        priority = envelope.priority if envelope.priority is not None else self._default_priority
        # Use scheme.compare for ordering: negate so higher priorities come first
        score = self._scheme.compare(priority, 0)  # compare against zero baseline
        heapq.heappush(self._heap, (-score, self._seq, envelope))
        self._seq += 1
        self._event.set()

        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.ensure_future(self._drain())

    async def _drain(self) -> None:
        """Process envelopes in priority order."""
        while self._heap and not self._closed:
            _, _, envelope = heapq.heappop(self._heap)

            if envelope.is_expired:
                continue

            ctx = Context(stream=self._shared_stream)
            for condition, callback in tuple(self._subscribers.values()):
                if condition and not condition(envelope.event):
                    continue
                await callback(envelope, ctx)

    def subscribe(
        self,
        callback: ChannelCallback,
        *,
        condition: Condition | None = None,
    ) -> SubId:
        sub_id = uuid4()
        self._subscribers[sub_id] = (condition, callback)
        return sub_id

    def unsubscribe(self, sub_id: SubId) -> None:
        self._subscribers.pop(sub_id, None)

    async def close(self) -> None:
        self._closed = True
        await self._drain()
        self._subscribers.clear()
        if self._drain_task and not self._drain_task.done():
            self._drain_task.cancel()
