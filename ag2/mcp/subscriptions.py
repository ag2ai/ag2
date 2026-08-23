# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Resource-change notifications for a served agent, across both protocol eras.

A served agent's resource *list* is fixed at construction, but a resource's
*body* is whatever its ``read`` returns, so the one change worth announcing is
"the resource at this URI is stale". :class:`ResourceNotifier` is the single
place that announcement enters the system; the two eras differ only in how it
leaves:

* **Modern era** (2026-07-28) — the client opens a ``subscriptions/listen``
  stream and ``mcp``'s own :class:`~mcp.server.subscriptions.ListenHandler`
  fans bus events onto it.
* **Handshake era** (≤ 2025-11-25) — the client calls ``resources/subscribe``
  and the server pushes ``notifications/resources/updated`` down that session.
  :class:`_HandshakeDelivery` keeps that bookkeeping.

Both hang off the same :class:`~mcp.server.subscriptions.SubscriptionBus`, so a
single publish reaches both, and swapping the bus for a cross-process one
(Redis, NATS) moves both eras at once rather than only the modern one.
"""

import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import anyio
from mcp.server.subscriptions import InMemorySubscriptionBus, SubscriptionBus
from mcp.shared.exceptions import MCPError
from mcp.shared.subscriptions import ResourceUpdated, ServerEvent
from mcp.types import INTERNAL_ERROR, INVALID_REQUEST

from .errors import MCPResourceNotFoundError
from .sessions import STDIO_SESSION

if TYPE_CHECKING:
    from mcp.server.context import ServerRequestContext
    from mcp.server.session import ServerSession

logger = logging.getLogger(__name__)

# ``ListenHandler``'s own defaults: the handshake registry and the modern listen
# streams should fall over at the same scale, not at two.
_MAX_SUBSCRIBERS = 1024
_MAX_BUFFERED_EVENTS = 1024


class ResourceNotifier:
    """Announces that a served resource's body changed.

    Construct one *before* the :class:`~ag2.mcp.MCPServer` that carries it and
    pass it as ``subscriptions=``. Building it first is what lets a served tool
    reach it — the tool closes over the notifier, where it could not close over
    a server that does not exist until its tools do::

        notifier = ResourceNotifier()


        @mcp_tool
        async def bump() -> str:
            counter["n"] += 1
            await notifier.notify_resource_updated("mem://counter")
            return "bumped"


        server = MCPServer(agent, resources=[...], tools=[bump], subscriptions=notifier)

    ``bus`` swaps the fan-out seam; the default is in-process. Give it a
    cross-process bus to serve subscribers spread over several replicas.

    ``max_subscribers`` bounds concurrent subscribers in each era — listen
    streams and handshake-era connections are counted separately, since they
    cost different things. Past the bound a further subscription is refused
    rather than served and dropped.
    """

    __slots__ = ("_bus", "_max_subscribers", "_resolves")

    def __init__(self, bus: SubscriptionBus | None = None, *, max_subscribers: int = _MAX_SUBSCRIBERS) -> None:
        self._bus = bus if bus is not None else InMemorySubscriptionBus()
        self._max_subscribers = max_subscribers
        # Set when an ``MCPServer`` adopts this notifier; until then there is no
        # resource set to check a URI against.
        self._resolves: Callable[[str], bool] | None = None

    @property
    def bus(self) -> SubscriptionBus:
        """The fan-out seam this notifier publishes to."""
        return self._bus

    @property
    def max_subscribers(self) -> int:
        """The per-era bound on concurrent subscribers."""
        return self._max_subscribers

    def _adopt(self, resolves: Callable[[str], bool]) -> None:
        """Bind to the resource set of the server now carrying this notifier."""
        if self._resolves is not None:
            raise ValueError(
                "This ResourceNotifier is already attached to an MCPServer; construct one per server "
                "(they may share a bus: ResourceNotifier(bus=other.bus))."
            )
        self._resolves = resolves

    async def notify_resource_updated(self, uri: str) -> None:
        """Tell subscribers the resource at ``uri`` is stale and worth re-reading.

        Raises :class:`~ag2.mcp.errors.MCPResourceNotFoundError` if ``uri``
        matches no served resource or template. A client subscribing to a URI
        nobody serves is honored in silence — the protocol says so, and the
        client may know something the server does not — but a *server*
        announcing one is a typo, and the alternative to raising is a
        notification that silently never arrives.
        """
        if self._resolves is not None and not self._resolves(uri):
            raise MCPResourceNotFoundError(uri)
        await self._bus.publish(ResourceUpdated(uri=uri))


def connection_key(ctx: "ServerRequestContext[Any, Any]") -> str:
    """The identity of the connection ``ctx`` arrived on.

    A subscription outlives the request that created it, so it has to be filed
    under something that outlives the request too. ``ctx.session`` does not:
    ``mcp`` 2.0 builds a fresh :class:`~mcp.server.session.ServerSession` proxy
    for every inbound message, so two requests on one connection carry two
    unequal session objects.

    What does outlive it is the transport's connection: the ``mcp-session-id``
    over HTTP — the same key :mod:`ag2.mcp.sessions` files conversation history
    under — and, where the transport carries a single client per process
    (stdio, in-memory streams), a constant.
    """
    request = ctx.request
    if request is None:
        return STDIO_SESSION
    session_id: str | None = request.headers.get("mcp-session-id")
    if not session_id:
        # Without connection identity, one client's subscription would deliver to
        # another's. A session-bearing transport is what the handshake era needs
        # anyway, which is why `stateless=True` turns this path off entirely.
        raise MCPError(INVALID_REQUEST, "resources/subscribe requires a session; this transport issues none")
    return session_id


class _HandshakeDelivery:
    """``resources/subscribe`` bookkeeping and delivery for handshake-era clients.

    Modern-era subscribers are served by ``ListenHandler``, whose stream *is* the
    request's response. Handshake-era subscribers have no such stream: the server
    pushes into a connection that some earlier request opened, so it has to
    remember which connection asked for which URIs.

    Each entry keeps the ``ServerSession`` last seen on that connection. The
    proxy is per-request, but a notification sent with no ``related_request_id``
    goes to the *connection's* standalone channel, so a proxy from an earlier
    request still reaches the right client.

    The bus hands events to a *synchronous* listener while sending is
    asynchronous, so the listener only buffers and a pump task does the sending.
    That is the same shape ``ListenHandler`` uses, and for the same reason: a
    slow client must not block the publisher.
    """

    __slots__ = ("_bus", "_max_subscribers", "_subscriptions", "_send", "_recv")

    def __init__(
        self,
        bus: SubscriptionBus,
        *,
        max_subscribers: int = _MAX_SUBSCRIBERS,
        max_buffered_events: int = _MAX_BUFFERED_EVENTS,
    ) -> None:
        self._bus = bus
        self._max_subscribers = max_subscribers
        self._subscriptions: dict[str, tuple[ServerSession, set[str]]] = {}
        self._send, self._recv = anyio.create_memory_object_stream[ResourceUpdated](max_buffered_events)

    def subscribe(self, key: str, session: "ServerSession", uri: str) -> None:
        """Record that the connection ``key`` wants updates for ``uri``.

        An unserved URI is accepted: the spec honors a subscription to a
        resource that does not exist, and it simply never fires.
        """
        existing = self._subscriptions.get(key)
        if existing is None:
            if len(self._subscriptions) >= self._max_subscribers:
                # A dead connection is only discovered when a send to it fails, so
                # an idle server would otherwise accumulate them without bound.
                raise MCPError(INTERNAL_ERROR, "Subscription limit reached")
            self._subscriptions[key] = (session, {uri})
        else:
            # Refresh the proxy: the oldest one works, but the newest is the one
            # least likely to be holding a channel the transport has since retired.
            _, uris = existing
            uris.add(uri)
            self._subscriptions[key] = (session, uris)

    def unsubscribe(self, key: str, uri: str) -> None:
        """Drop ``key``'s interest in ``uri``, forgetting the connection when it holds none."""
        existing = self._subscriptions.get(key)
        if existing is None:
            return
        _, uris = existing
        uris.discard(uri)
        if not uris:
            del self._subscriptions[key]

    @asynccontextmanager
    async def running(self) -> AsyncGenerator[None]:
        """Subscribe to the bus and run the pump for the duration of the block."""
        unsubscribe = self._bus.subscribe(self._enqueue)
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(self._pump)
                try:
                    yield
                finally:
                    tg.cancel_scope.cancel()
        finally:
            unsubscribe()

    def _enqueue(self, event: ServerEvent) -> None:
        """Buffer ``event`` for the pump. Synchronous, and must not raise."""
        if not isinstance(event, ResourceUpdated):
            # Handshake-era subscriptions cover individual resources only; the
            # list-changed events have no subscriber on this path.
            return
        try:
            self._send.send_nowait(event)
        except anyio.WouldBlock:
            logger.warning("handshake subscription backlog full; dropping update for %r", event.uri)
        except anyio.ClosedResourceError:  # pragma: no cover - only after the pump is torn down
            pass

    async def _pump(self) -> None:
        """Deliver buffered events to the connections that subscribed to them."""
        async for event in self._recv:
            for key, (session, uris) in list(self._subscriptions.items()):
                if event.uri not in uris:
                    continue
                try:
                    await session.send_resource_updated(event.uri)
                except Exception:
                    # A closed connection is the ordinary end of a subscription and
                    # the only signal we get: no hook reports transport teardown.
                    logger.debug("dropping subscriptions of a connection that could not be reached", exc_info=True)
                    self._subscriptions.pop(key, None)


__all__ = ("ResourceNotifier",)
