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

**An announcement is pushed, never derived.** Nothing here watches
:class:`~ag2.mcp.resources.Resource` — those stay frozen dataclasses with no
notion of mutation, and the server never learns *what* changed. The protocol
defines the notification as a URI-only invalidation signal, and whether the view
behind a URI has gone stale is known only to the code owning the underlying
data: a served tool that mutated it, a file watcher, a webhook. So the server
author calls :meth:`ResourceNotifier.notify_resource_updated` from wherever the
change happens. A mutable resource kind that publishes on its own writes remains
possible later as sugar over this same seam.
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

# ``ListenHandler``'s own defaults, copied rather than imported because the SDK
# exports them only as parameter defaults. Both are reachable as
# ``ResourceNotifier`` keywords, and ``test_subscriptions`` pins them against the
# SDK through that seam, so a drift there fails a test instead of silently
# leaving the handshake registry and the modern listen streams to fall over at
# two scales.
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

    ``max_subscribers`` bounds how many concurrent subscribers the server
    admits, in each era separately — modern-era listen streams and handshake-era
    connections are counted apart, since they cost different things — so the
    total a server admits is twice the number passed. Past the bound a further
    subscriber is refused rather than admitted and dropped.

    The same number *additionally* bounds how many URIs one handshake-era
    connection may hold, because subscribing to a URI nobody serves is
    deliberately accepted and counting connections alone would leave one of them
    free to invent URIs without end. The modern era has no equivalent dimension:
    a listen stream's filter arrives whole in one request, and the number of
    streams is already bounded.

    ``max_buffered_events`` bounds how many announcements a single handshake-era
    subscriber may fall behind by before it is dropped: the protocol offers no
    replay, so past its backlog there is nothing left worth keeping for it. The
    modern era's equivalent is ``ListenHandler``'s own per-stream buffer, which
    the SDK sizes from the same default.
    """

    __slots__ = ("_bus", "_max_buffered_events", "_max_subscribers", "_resolves")

    def __init__(
        self,
        bus: SubscriptionBus | None = None,
        *,
        max_subscribers: int = _MAX_SUBSCRIBERS,
        max_buffered_events: int = _MAX_BUFFERED_EVENTS,
    ) -> None:
        if max_subscribers < 1:
            raise ValueError(f"max_subscribers must be >= 1, got {max_subscribers}.")
        if max_buffered_events < 1:
            raise ValueError(f"max_buffered_events must be >= 1, got {max_buffered_events}.")
        self._bus = bus if bus is not None else InMemorySubscriptionBus()
        self._max_subscribers = max_subscribers
        self._max_buffered_events = max_buffered_events
        # Set when an ``MCPServer`` adopts this notifier; until then there is no
        # resource set to check a URI against.
        self._resolves: Callable[[str], bool] | None = None

    @property
    def bus(self) -> SubscriptionBus:
        """The fan-out seam this notifier publishes to."""
        return self._bus

    @property
    def max_subscribers(self) -> int:
        """The per-era bound on concurrent subscribers, and on one connection's URIs."""
        return self._max_subscribers

    @property
    def max_buffered_events(self) -> int:
        """How far one handshake-era subscriber may fall behind before it is dropped."""
        return self._max_buffered_events

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

        Raises :class:`ValueError` before any :class:`~ag2.mcp.MCPServer` has
        adopted this notifier, for the same reason: there is no resource set to
        check the URI against and nobody subscribed to hear it, so publishing
        would be the silent no-op the strictness above exists to rule out.
        """
        if self._resolves is None:
            raise ValueError(
                "This ResourceNotifier is not attached to an MCPServer yet; pass it as "
                "`MCPServer(..., subscriptions=notifier)` before announcing."
            )
        if not self._resolves(uri):
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
    if not session_id:  # pragma: no cover - streamable HTTP rejects such a request before dispatch
        # Without connection identity, one client's subscription would deliver
        # to another's. Unreachable through the streamable-HTTP transport, which
        # answers a non-``initialize`` request carrying no session id with
        # ``400 Bad Request: Missing session ID`` before a handler sees it — but
        # the handlers are transport-agnostic, so a session-bearing transport
        # that issues none must fail rather than file the subscription under a
        # key every client shares.
        raise MCPError(INVALID_REQUEST, "resource subscriptions require a session; this transport issues none")
    return session_id


class _Subscriber:
    """One handshake-era connection's subscribed URIs, and its delivery buffer.

    Each subscriber buffers and sends on its own, mirroring
    :class:`~mcp.server.subscriptions.ListenHandler`'s per-stream shape. A
    single shared queue would make one wedged client's transport write stall
    delivery to every other subscriber, and would make a full backlog a reason
    to drop the *event* for everyone rather than the subscriber that caused it.
    """

    __slots__ = ("session", "uris", "_send", "_recv")

    def __init__(self, session: "ServerSession", *, max_buffered_events: int) -> None:
        self.session = session
        self.uris: set[str] = set()
        self._send, self._recv = anyio.create_memory_object_stream[str](max_buffered_events)

    def offer(self, uri: str) -> bool:
        """Buffer ``uri`` for this subscriber's delivery task, without blocking.

        ``False`` means this subscriber must be dropped: either its backlog is
        full (it has stopped keeping up, and the protocol offers no replay, so
        there is nothing to keep it for) or it has already been closed.
        """
        try:
            self._send.send_nowait(uri)
        except (anyio.WouldBlock, anyio.ClosedResourceError):
            return False
        return True

    def close(self) -> None:
        """End this subscriber's buffer, which ends :meth:`deliver`."""
        self._send.close()

    async def deliver(self) -> None:
        """Send buffered announcements until the buffer closes or a send fails."""
        async for uri in self._recv:
            await self.session.send_resource_updated(uri)


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
    asynchronous, so the listener only buffers and one task per subscriber does
    the sending. That is the same shape ``ListenHandler`` uses, and for the same
    reasons: a slow client must block neither the publisher nor its fellow
    subscribers.

    The registry is per *serving*, not per server: :meth:`running` clears it on
    the way out, so a server served twice does not hand a fresh client the
    subscriptions of a departed one filed under the same connection key.
    """

    __slots__ = ("_bus", "_max_buffered_events", "_max_subscribers", "_subscribers", "_tasks")

    def __init__(
        self,
        bus: SubscriptionBus,
        *,
        max_subscribers: int = _MAX_SUBSCRIBERS,
        max_buffered_events: int = _MAX_BUFFERED_EVENTS,
    ) -> None:
        self._bus = bus
        self._max_subscribers = max_subscribers
        self._max_buffered_events = max_buffered_events
        self._subscribers: dict[str, _Subscriber] = {}
        # The task group running the per-subscriber delivery tasks; only set
        # while ``running()`` is held, which is whenever the server serves.
        self._tasks: anyio.abc.TaskGroup | None = None

    def subscribe(self, key: str, session: "ServerSession", uri: str) -> None:
        """Record that the connection ``key`` wants updates for ``uri``.

        An unserved URI is accepted: the spec honors a subscription to a
        resource that does not exist, and it simply never fires. That leniency
        is why ``max_subscribers`` bounds this registry along both of its
        dimensions — how many connections it holds, and how many URIs each of
        them holds. Bounding connections alone would leave one of them free to
        invent URIs nobody serves and grow the registry without limit, never
        spending more than its single slot. The URI bound is per connection
        rather than a total shared across them, so a client subscribing to many
        URIs exhausts its own quota instead of locking every other client out of
        subscribing at all.

        Overflow refuses in both dimensions; it never evicts. The
        conversation-session registry in :mod:`ag2.mcp.sessions`, keyed by this
        same connection identity, deliberately does the opposite and evicts LRU:
        losing history there costs a caller some context it can rebuild, whereas
        an evicted subscription costs it the truth — the client goes on believing
        it holds a working subscription it will never hear from again. That is
        the silent failure the stateless-mode warning exists to prevent, and
        manufacturing it here would be an odd way to honour that.
        """
        tasks = self._tasks
        if tasks is None:  # pragma: no cover - the handler is reachable only while serving
            raise MCPError(INTERNAL_ERROR, "This server is not serving subscriptions")
        subscriber = self._subscribers.get(key)
        if subscriber is None:
            if len(self._subscribers) >= self._max_subscribers:
                raise MCPError(INTERNAL_ERROR, "Subscription limit reached")
            subscriber = _Subscriber(session, max_buffered_events=self._max_buffered_events)
            subscriber.uris.add(uri)
            self._subscribers[key] = subscriber
            tasks.start_soon(self._deliver, key, subscriber)
            return
        # Refusing before the mutation is what leaves the URIs this connection
        # already holds working; re-subscribing to one of them is not a new URI
        # and so cannot be what tips it over.
        if uri not in subscriber.uris and len(subscriber.uris) >= self._max_subscribers:
            raise MCPError(INTERNAL_ERROR, "Subscription limit reached for this connection")
        subscriber.uris.add(uri)
        # Refresh the proxy: the oldest one works, but the newest is the one
        # least likely to be holding a channel the transport has since retired.
        subscriber.session = session

    def unsubscribe(self, key: str, uri: str) -> None:
        """Drop ``key``'s interest in ``uri``, forgetting the connection when it holds none."""
        subscriber = self._subscribers.get(key)
        if subscriber is None:
            return
        subscriber.uris.discard(uri)
        if not subscriber.uris:
            self._drop(key, subscriber)

    @asynccontextmanager
    async def running(self) -> AsyncGenerator[None]:
        """Subscribe to the bus and run per-subscriber delivery for the duration of the block."""
        async with anyio.create_task_group() as tg:
            self._tasks = tg
            unsubscribe = self._bus.subscribe(self._enqueue)
            try:
                yield
            finally:
                unsubscribe()
                self._tasks = None
                for key, subscriber in list(self._subscribers.items()):
                    self._drop(key, subscriber)
                # Closing each buffer ends its task; cancelling covers a task
                # left waiting on a transport write that will never complete.
                tg.cancel_scope.cancel()

    def _drop(self, key: str, subscriber: _Subscriber) -> None:
        """Forget ``subscriber`` and end its delivery task. Safe to repeat."""
        if self._subscribers.get(key) is subscriber:
            del self._subscribers[key]
        subscriber.close()

    def _enqueue(self, event: ServerEvent) -> None:
        """Buffer ``event`` for every interested subscriber. Synchronous, and must not raise."""
        if not isinstance(event, ResourceUpdated):
            # Handshake-era subscriptions cover individual resources only; the
            # list-changed events have no subscriber on this path.
            return
        for key, subscriber in list(self._subscribers.items()):
            if event.uri not in subscriber.uris:
                continue
            if not subscriber.offer(event.uri):
                logger.warning("handshake subscription backlog full; dropping subscriber %r", key)
                self._drop(key, subscriber)

    async def _deliver(self, key: str, subscriber: _Subscriber) -> None:
        """Run one subscriber's delivery until its buffer closes or its connection fails."""
        try:
            await subscriber.deliver()
        except Exception:
            # A closed connection is the ordinary end of a subscription, and on
            # a duplex transport it is the only signal we get: nothing reports
            # transport teardown.
            #
            # Streamable HTTP does not raise here. Its router drops a
            # notification whose standalone stream is gone and logs at debug, so
            # a departed HTTP client's entry is reclaimed only when its MCP
            # session is reused or the process restarts. Combined with the
            # refuse-never-evict policy above, a server can therefore sit at
            # ``max_subscribers`` holding entries for clients that are long
            # gone. Detecting that needs a teardown hook the SDK does not expose
            # to a lowlevel handler (``Connection.exit_stack`` is not reachable
            # from ``ServerRequestContext``); until it does, size
            # ``max_subscribers`` for the churn a deployment sees, not for its
            # concurrent client count.
            logger.debug("dropping subscriptions of a connection that could not be reached", exc_info=True)
        finally:
            self._drop(key, subscriber)


__all__ = ("ResourceNotifier",)
