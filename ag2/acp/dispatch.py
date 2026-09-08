# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Make inbound ``session/update`` notifications handled in wire order.

The SDK spawns each inbound notification as its own task: the receive loop reads
a message and calls ``create_task`` on the handler, so two ``session/update``
handlers that both await — and AG2's does, it sends an event to the run's stream —
can interleave. Chunks then land out of the order the agent sent them, and the
assembled reply is scrambled rather than short.

Completeness, the other half of this, is the SDK's own since 0.12.1: a
``ClientSideConnection`` tracks the ``session/update`` handlers it has in flight
per session and ``session/prompt`` does not return until they have all returned.
So a caller that reads the turn out after the prompt response sees every update
that preceded it on the wire. That is what this module used to arrange itself, by
handing ``Connection`` a queue it could ``join()``; the queue is gone from the SDK
and the guarantee is now upstream, but the ordering is not.

Ordering is recovered with a lock rather than a queue, because task *creation* is
already in wire order: the receive loop is sequential, so handler tasks start in
the order their messages were read, and each one takes this lock as its first
await. ``asyncio.Lock`` hands off to waiters in the order they arrived, so the
lock's order is the wire's order. The trade-off is that a slow handler holds up
the updates behind it — bounded by the run's own subscribers, and by the turn
timeout above it, since ``prompt`` is waiting on the same handlers.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

__all__ = ["InOrderUpdates"]


class InOrderUpdates:
    """The per-bridge lock that makes update handling sequential.

    One of these is owned by the bridge (:class:`~.bridge.BridgeState`) and taken
    by its ``session_update`` route. A bridge with no connection — a test double
    that calls ``handle_update`` directly — simply never contends on it.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def in_order(self) -> AsyncGenerator[None]:
        """Hold off the updates read after this one until this one is handled."""
        async with self._lock:
            yield
