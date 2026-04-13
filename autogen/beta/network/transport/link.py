# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Link protocol — the single bidirectional pipe between a client and a hub.

A ``Link`` is created in two roles:

* **Client role** — owned by a ``HubClient`` / ``ActorClient``. Exposes
  :meth:`send_frame` to push frames *to* the hub and an async iterator of
  :meth:`frames` to receive frames *from* the hub.
* **Server role** — owned by the hub's ``LinkServer``. The hub receives a
  :class:`LinkEndpoint` per connected client and sends frames back via
  :meth:`LinkEndpoint.send_frame`.

Phase 1 ships only :class:`LocalLink`, an in-process duplex that short-
circuits network I/O. Phase 3 adds :class:`WsLink`. Both honor this exact
protocol.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from .frames import Frame


@runtime_checkable
class LinkClient(Protocol):
    """Client side of a Link. Owned by HubClient / ActorClient."""

    async def send_frame(self, frame: Frame) -> None:
        """Send a frame to the hub. Raises :class:`LinkClosedError` if closed."""
        ...

    def frames(self) -> AsyncIterator[Frame]:
        """Async iterator over frames from the hub."""
        ...

    async def close(self) -> None:
        """Close the link. Idempotent."""
        ...

    @property
    def closed(self) -> bool: ...


@runtime_checkable
class LinkEndpoint(Protocol):
    """Server-side peer handle held by the hub."""

    endpoint_id: str
    actor_id: str | None

    async def send_frame(self, frame: Frame) -> None: ...

    def frames(self) -> AsyncIterator[Frame]:
        """Async iterator over frames from the client.

        The hub's :meth:`~autogen.beta.network.hub.core.Hub.connection_handler`
        drives this iterator for the life of the connection.
        """
        ...

    async def close(self) -> None: ...

    @property
    def closed(self) -> bool: ...


@runtime_checkable
class LinkServer(Protocol):
    """Hub-side listener. Accepts client connections and hands them to the hub.

    The hub registers a single connection handler via :meth:`on_connection`
    and is responsible for driving the framing loop of each
    :class:`LinkEndpoint` it receives.
    """

    def on_connection(self, handler: ConnectionHandler) -> None: ...

    async def close(self) -> None: ...


class ConnectionHandler(Protocol):
    async def __call__(self, endpoint: LinkEndpoint) -> None: ...


@runtime_checkable
class Link(Protocol):
    """Umbrella protocol for transports that expose both sides.

    :class:`LocalLink` implements this — it is a server *and* dispenses
    client handles from the same process. Remote transports will implement
    it asymmetrically (hub-side ``Link`` is the server; client-side ``Link``
    is a client), but the same base class is used so callers can bind the
    ``Link`` to either role.
    """

    def client(self) -> LinkClient:
        """Return a new client handle."""
        ...
