# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""LocalLink — in-process duplex transport.

Two ``asyncio.Queue`` objects form a bidirectional pipe between a
``LinkClient`` and a ``LinkEndpoint`` (hub-side). No encoding or decoding —
frames are passed by reference. This is the transport used by all Phase 1
in-process tests and runtime, and its frame exchange matches what
``WsLink`` will do in Phase 3, so application code above the transport is
transport-agnostic.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable

from ..errors import LinkClosedError
from ..ids import new_id
from .frames import Frame

__all__ = ("LocalLink",)


_SENTINEL = object()


class _ClientSide:
    """Client handle returned by :meth:`LocalLink.client`."""

    def __init__(self, peer_inbox: asyncio.Queue[object]) -> None:
        self._peer_inbox = peer_inbox
        self._inbox: asyncio.Queue[object] = asyncio.Queue()
        self._closed = False

    async def send_frame(self, frame: Frame) -> None:
        if self._closed:
            raise LinkClosedError("link is closed")
        await self._peer_inbox.put(frame)

    def frames(self) -> AsyncIterator[Frame]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Frame]:
        while True:
            item = await self._inbox.get()
            if item is _SENTINEL:
                return
            yield item  # type: ignore[misc]

    async def _deliver(self, frame: Frame) -> None:
        if self._closed:
            return
        await self._inbox.put(frame)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._inbox.put(_SENTINEL)
        await self._peer_inbox.put(_SENTINEL)

    @property
    def closed(self) -> bool:
        return self._closed


class _EndpointSide:
    """Hub-side peer handle passed to the hub's connection handler.

    ``binding`` exposes the transport flavor to the hub so
    ``_handle_hello`` can stamp the actor's ``runtime.json`` with the
    right ``binding`` field (§3.4). ``LocalLink`` is always ``"local"``;
    ``WsLink`` will set ``"ws"`` and carry ``ws_url`` / ``http_url``
    alongside, matching the design's runtime-binding shape.
    """

    binding: str = "local"
    ws_url: str | None = None
    http_url: str | None = None

    def __init__(self, client: _ClientSide) -> None:
        self.endpoint_id = new_id()
        self.actor_id: str | None = None
        self._client = client
        self._inbox: asyncio.Queue[object] = asyncio.Queue()
        self._closed = False

    async def send_frame(self, frame: Frame) -> None:
        if self._closed:
            raise LinkClosedError("endpoint is closed")
        await self._client._deliver(frame)

    def frames(self) -> AsyncIterator[Frame]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Frame]:
        while True:
            item = await self._inbox.get()
            if item is _SENTINEL:
                return
            yield item  # type: ignore[misc]

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._inbox.put(_SENTINEL)
        if not self._client.closed:
            await self._client._deliver(_SENTINEL)  # type: ignore[arg-type]
            self._client._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


ConnectionCallback = Callable[[_EndpointSide], Awaitable[None]]


class LocalLink:
    """In-process Link implementation.

    The hub registers a connection handler via :meth:`on_connection`. Every
    :meth:`client` call spawns a new client / endpoint pair, hands the
    endpoint to the hub's handler (in a background task), and returns the
    client-side handle to the caller.
    """

    def __init__(self) -> None:
        self._handler: ConnectionCallback | None = None
        self._pending: list[asyncio.Task[None]] = []
        self._closed = False

    def on_connection(self, handler: ConnectionCallback) -> None:
        self._handler = handler

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in list(self._pending):
            if not task.done():
                task.cancel()
        for task in list(self._pending):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    def client(self) -> _ClientSide:
        if self._closed:
            raise LinkClosedError("link is closed")
        handler = self._handler
        if handler is None:
            raise LinkClosedError("no connection handler registered on link")

        endpoint_inbox: asyncio.Queue[object] = asyncio.Queue()
        client = _ClientSide(peer_inbox=endpoint_inbox)
        endpoint = _EndpointSide(client=client)
        endpoint._inbox = endpoint_inbox  # share the queue the client sends into

        async def _run() -> None:
            try:
                await handler(endpoint)
            finally:
                await endpoint.close()

        task = asyncio.get_event_loop().create_task(_run())
        self._pending.append(task)
        task.add_done_callback(
            lambda t: self._pending.remove(t) if t in self._pending else None
        )
        return client
