# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``Link`` Protocol — wire abstraction between ``HubClient`` and ``Hub``.

Two roles:

* ``LinkClient`` — tenant-side handle to the hub. ``HubClient`` holds
  one per process per hub.
* ``LinkEndpoint`` — hub-side handle to one connected client.

``LocalLink`` (see ``.local``) is the in-process implementation.
Cross-process transports satisfy the same Protocol surface.
"""

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from .frames import Frame

__all__ = ("LinkClient", "LinkEndpoint", "LinkFactory")


class LinkClient(Protocol):
    """Tenant-side handle to the hub.

    The ``HubClient`` opens one ``LinkClient`` per process per hub and
    multiplexes any number of registered ``AgentClient``s through it.
    """

    endpoint_id: str

    async def open(self) -> None:
        """Connect, perform ``hello`` handshake, await ``welcome``.

        ``LocalLink`` makes this a no-op — connection is implicit when
        the link is created.
        """
        ...

    async def send_frame(self, frame: Frame) -> None:
        """Push a frame towards the hub."""
        ...

    def frames(self) -> AsyncIterator[Frame]:
        """Async iterator of inbound frames from the hub.

        Iteration ends when ``close()`` is called.
        """
        ...

    async def close(self) -> None:
        """Drain queues, signal close to the hub-side handler."""
        ...


class LinkEndpoint(Protocol):
    """Hub-side handle to one connected client.

    Lives for the duration of a single connection; replaced on
    reconnect (the ``HubClient`` re-opens after a transport drop).
    """

    endpoint_id: str
    agent_id: str | None  # set after the hub binds an identity to this connection

    async def send_frame(self, frame: Frame) -> None:
        """Push a frame towards the client."""
        ...

    def frames(self) -> AsyncIterator[Frame]:
        """Async iterator of inbound frames from the client."""
        ...

    async def close(self) -> None:
        """Drain queues, signal close to the client-side handler."""
        ...


@runtime_checkable
class LinkFactory(Protocol):
    """Produces ``LinkClient`` connections to one hub.

    ``HubClient`` holds a factory (``LocalLink`` in-process, ``WsLink``
    over WebSocket) and calls :meth:`client` once to open its
    connection. ``LocalLink`` additionally exposes a ``hub`` attribute
    so an in-process ``HubClient`` can default to direct hub calls; a
    wire factory has no such attribute and the ``HubClient`` routes
    control-plane operations through ``RequestFrame`` RPC instead.
    """

    def client(self) -> LinkClient:
        """Open (or allocate) a fresh ``LinkClient`` to the hub."""
        ...
