# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``LocalLink`` — in-memory duplex implementing the ``Link`` Protocol.

Each ``LocalLink.client()`` call creates a paired ``LocalLinkClient`` /
``LocalLinkEndpoint`` sharing two ``asyncio.Queue``s (client→hub and
hub→client). The hub is notified of the new endpoint via
``LocalLink.on_connect`` so it can spawn a frame-processor task.

Tests run real protocol traffic through the same Frame vocabulary that
a wire transport would use — no socket, no serialization overhead, but
identical protocol coverage.

``LocalLinkClient.open()`` is a no-op (connection is implicit at
``client()`` time). Heartbeat refresh is synchronous: every frame
operation refreshes ``last_heartbeat``; no wire ping is sent.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ..ids import make_id
from .frames import Frame

if TYPE_CHECKING:
    from ..hub import Hub

__all__ = ("LocalLink", "LocalLinkClient", "LocalLinkEndpoint")


class LocalLinkClient:
    """Tenant-side half of an in-memory duplex.

    Holds the two queues; ``send_frame`` writes to the client→hub queue,
    ``frames()`` reads from the hub→client queue. ``close()`` pushes a
    sentinel so the iterator terminates cleanly on both ends.
    """

    def __init__(
        self,
        endpoint_id: str,
        client_to_hub: "asyncio.Queue[Frame | None]",
        hub_to_client: "asyncio.Queue[Frame | None]",
    ) -> None:
        # __init__ stores params; no side effects.
        self.endpoint_id = endpoint_id
        self._c2h = client_to_hub
        self._h2c = hub_to_client
        self._closed = False

    async def open(self) -> None:
        """No-op — connection is implicit at ``LocalLink.client()`` time."""
        return None

    async def send_frame(self, frame: Frame) -> None:
        if self._closed:
            return
        await self._c2h.put(frame)

    def frames(self) -> AsyncIterator[Frame]:
        return self._frames_impl()

    async def _frames_impl(self) -> AsyncIterator[Frame]:
        while True:
            frame = await self._h2c.get()
            if frame is None:
                return
            yield frame

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Sentinel both directions so handlers on both sides stop iterating.
        await self._c2h.put(None)
        await self._h2c.put(None)


class LocalLinkEndpoint:
    """Hub-side half of an in-memory duplex.

    Mirror of ``LocalLinkClient``: ``send_frame`` writes hub→client,
    ``frames()`` reads client→hub. ``agent_id`` is set by the hub once
    a ``HelloFrame`` (or direct in-process binding) associates this
    endpoint with an identity.
    """

    def __init__(
        self,
        endpoint_id: str,
        client_to_hub: "asyncio.Queue[Frame | None]",
        hub_to_client: "asyncio.Queue[Frame | None]",
    ) -> None:
        # __init__ stores params; no side effects.
        self.endpoint_id = endpoint_id
        self.agent_id: str | None = None
        self._c2h = client_to_hub
        self._h2c = hub_to_client
        self._closed = False

    async def send_frame(self, frame: Frame) -> None:
        if self._closed:
            return
        await self._h2c.put(frame)

    def frames(self) -> AsyncIterator[Frame]:
        return self._frames_impl()

    async def _frames_impl(self) -> AsyncIterator[Frame]:
        while True:
            frame = await self._c2h.get()
            if frame is None:
                return
            yield frame

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._c2h.put(None)
        await self._h2c.put(None)


class LocalLink:
    """Factory creating in-process duplex pairs against one ``Hub``.

    Pass to ``HubClient(link=LocalLink(hub), hub=hub)``. Each
    ``HubClient`` requests one ``LocalLinkClient`` (held for the life
    of the process); multiple ``AgentClient``s registered through that
    ``HubClient`` share the connection.

    Each ``client()`` call constructs a fresh endpoint and immediately
    hands it to ``Hub.attach_endpoint`` so the hub spawns its
    frame-processor task. A wire transport follows the same shape —
    its connect handler calls ``Hub.attach_endpoint`` once the
    connection is up.
    """

    def __init__(self, hub: "Hub") -> None:
        # __init__ stores params; side effects deferred to client().
        self._hub = hub

    @property
    def hub(self) -> "Hub":
        return self._hub

    def client(self) -> LocalLinkClient:
        """Create a fresh client+endpoint pair; attach to the hub."""
        endpoint_id = make_id()
        c2h: asyncio.Queue[Frame | None] = asyncio.Queue()
        h2c: asyncio.Queue[Frame | None] = asyncio.Queue()
        endpoint = LocalLinkEndpoint(endpoint_id=endpoint_id, client_to_hub=c2h, hub_to_client=h2c)
        client = LocalLinkClient(endpoint_id=endpoint_id, client_to_hub=c2h, hub_to_client=h2c)
        self._hub.attach_endpoint(endpoint)
        return client
