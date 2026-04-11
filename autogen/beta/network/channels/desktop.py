# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""DesktopChannel — bridges a Hub to local agents on a user's desktop.

This channel is designed for the AgentOS architecture where the cloud Hub
communicates with desktop-hosted agents (Claude Code, MiniMax, ElevenLabs,
etc.) via an existing WebSocket connection. The channel itself does NOT
own the WSS connection — it accepts a transport callback and a delivery
inbox so it can plug into any existing duplex transport.

Wire model:

    Cloud Hub                              Desktop
    ─────────                              ───────
    network(target="claude-code", ...)
        │
        ▼
    DesktopProxyAgent.ask()
        │
        ▼
    DesktopChannel.send(envelope)  ───►   AgentManager.handleDelegation()
                                                │
                                                ▼
    DesktopChannel.deliver(envelope) ◄───  ClaudeCodeAdapter.execute()
        │
        ▼
    callback fires → Future resolves
        │
        ▼
    network() returns result

Concurrency: multiple delegations can be in flight at once. Each is keyed
by `correlation_id`, and the channel maintains a `correlation_id → Future`
map so results route back to the right caller.

This channel does not perform retries — at-least-once delivery is the
WSS layer's responsibility (the cloud has reconnect/replay; if a result
comes back twice, the second deliver is a no-op since the Future is
already resolved).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from autogen.beta.context import Context, SubId
from autogen.beta.events.conditions import Condition
from autogen.beta.stream import MemoryStream

from ..primitives.channel import ChannelCallback
from ..primitives.envelope import Envelope

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# A SendCallback is what the channel uses to push an envelope toward the
# desktop. The cloud's WSS handler provides this — it serializes the
# envelope and writes a `agent_delegation` frame on the WebSocket.
SendCallback = Callable[[Envelope], Awaitable[None]]


class DesktopChannel:
    """Channel implementation for cloud ↔ desktop agent communication.

    Conforms to the AG2 Channel protocol (send / subscribe / unsubscribe /
    close) but ALSO exposes ``deliver()`` for the transport to push
    received envelopes back into the channel.

    Lifecycle:
        1. Cloud creates the channel with a `send_callback` that knows how
           to write to the user's WSS connection.
        2. The DesktopProxyAgent subscribes for incoming results.
        3. ``send(envelope)`` invokes ``send_callback(envelope)`` which
           serializes and writes to the WSS.
        4. When the desktop replies, the WSS handler calls
           ``channel.deliver(envelope)`` which fires subscribers AND
           resolves any pending Future awaiting that correlation_id.

    Example::

        async def transport(env: Envelope) -> None:
            await ws.send_json({"type": "agent_delegation", "payload": env.to_dict()})

        channel = DesktopChannel(send_callback=transport)
        proxy = DesktopProxyAgent("claude-code", channel=channel)

        # When desktop replies, ws handler calls:
        await channel.deliver(result_envelope)
    """

    def __init__(
        self,
        send_callback: SendCallback,
        *,
        default_timeout: float = 600.0,
    ) -> None:
        self._send_callback = send_callback
        self._default_timeout = default_timeout
        self._subscribers: dict[SubId, tuple[Condition | None, ChannelCallback]] = {}
        # correlation_id → Future awaiting the result for that delegation
        self._pending: dict[str, asyncio.Future[Envelope]] = {}
        self._closed = False
        # Shared stream for context delivery to subscribers (mirrors LocalChannel)
        self._stream = MemoryStream()

    # ------------------------------------------------------------------
    # Channel protocol
    # ------------------------------------------------------------------

    async def send(self, envelope: Envelope) -> None:
        """Push an envelope toward the desktop via the transport callback."""
        if self._closed:
            raise RuntimeError("DesktopChannel is closed")
        await self._send_callback(envelope)

    def subscribe(
        self,
        callback: ChannelCallback,
        *,
        condition: Condition | None = None,
    ) -> SubId:
        sub_id: SubId = uuid4()
        self._subscribers[sub_id] = (condition, callback)
        return sub_id

    def unsubscribe(self, sub_id: SubId) -> None:
        self._subscribers.pop(sub_id, None)

    async def close(self) -> None:
        """Cancel all pending awaits and clear subscribers."""
        self._closed = True
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        self._subscribers.clear()

    # ------------------------------------------------------------------
    # Delivery (called by WSS handler when desktop sends a reply)
    # ------------------------------------------------------------------

    async def deliver(self, envelope: Envelope) -> None:
        """Deliver an envelope received from the desktop.

        Two side effects:
        1. Fires all subscribers whose condition matches.
        2. If a Future is registered for ``envelope.correlation_id``,
           resolves it with the envelope so the original caller can wake.
        """
        if self._closed:
            return

        # Wake any caller awaiting this correlation_id
        fut = self._pending.pop(envelope.correlation_id, None)
        if fut is not None and not fut.done():
            fut.set_result(envelope)

        # Fire subscribers
        ctx = Context(stream=self._stream)
        for condition, callback in tuple(self._subscribers.values()):
            try:
                if condition and not condition(envelope.event):
                    continue
                await callback(envelope, ctx)
            except Exception:
                logger.exception(
                    "DesktopChannel subscriber failed for envelope %s",
                    envelope.correlation_id,
                )

        # Also publish on the stream for stream-level subscribers
        try:
            await self._stream.send(envelope.event, ctx)
        except Exception:
            logger.debug(
                "DesktopChannel stream publish failed for %s",
                envelope.correlation_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Request/response convenience
    # ------------------------------------------------------------------

    async def request(
        self,
        envelope: Envelope,
        *,
        timeout: float | None = None,
    ) -> Envelope:
        """Send an envelope and wait for the matching reply.

        The reply is matched by ``correlation_id``. The desktop side MUST
        echo the same correlation_id on its reply envelope.

        Raises:
            asyncio.TimeoutError: if no reply arrives within ``timeout``.
            RuntimeError: if the channel is closed before the reply.
        """
        if self._closed:
            raise RuntimeError("DesktopChannel is closed")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Envelope] = loop.create_future()
        self._pending[envelope.correlation_id] = future

        try:
            await self._send_callback(envelope)
            return await asyncio.wait_for(
                future,
                timeout=timeout if timeout is not None else self._default_timeout,
            )
        finally:
            # Clean up regardless of outcome (success, timeout, cancel)
            self._pending.pop(envelope.correlation_id, None)

    # ------------------------------------------------------------------
    # Introspection (test helpers)
    # ------------------------------------------------------------------

    @property
    def pending_count(self) -> int:
        """How many requests are awaiting a response."""
        return len(self._pending)

    @property
    def is_closed(self) -> bool:
        return self._closed
