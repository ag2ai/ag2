# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""WebSocket transport — :class:`LinkClient` / :class:`LinkEndpoint`
implementations that carry the same Frame vocabulary as
:class:`LocalLink` over a real WebSocket.

The client side is :class:`WsLinkClient`; the server side is
:class:`WsLinkEndpoint`, produced by :func:`serve_ws` for each
incoming connection. Each WS message is a JSON object — the result of
:func:`encode_frame` — and inbound messages flow through
:func:`decode_frame` before being yielded by :meth:`frames`.

Heartbeat is delegated to the underlying ``websockets`` library via
``ping_interval`` / ``ping_timeout``; the framework does not wire its
own ``PingFrame`` / ``PongFrame`` over WebSocket (the library's
control frames already handle that). Reconnect logic is the tenant's
responsibility — :meth:`WsLinkClient.open` does one connect; callers
that need backoff wrap it.
"""

import asyncio
import contextlib
import functools
import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from websockets.asyncio.client import connect as _ws_connect
from websockets.asyncio.server import serve as _ws_serve
from websockets.exceptions import ConnectionClosed

from ..ids import make_id
from .frames import Frame, decode_frame, encode_frame

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection
    from websockets.asyncio.server import Server as _WsServer
    from websockets.asyncio.server import ServerConnection

    from ..hub import Hub

__all__ = ("WsLink", "WsLinkClient", "WsLinkEndpoint", "serve_ws")


logger = logging.getLogger(__name__)


class WsLinkClient:
    """Tenant-side WebSocket link to a hub.

    Construct with the hub's URL, then ``await client.open()`` to
    perform the WebSocket handshake. Frames flow over a single bidirectional
    connection: outbound via :meth:`send_frame`, inbound via
    :meth:`frames`. ``endpoint_id`` is populated by the
    :class:`WelcomeFrame` the hub sends in response to a
    :class:`HelloFrame`; before that it is the empty string.

    Closing the client closes the underlying WebSocket. Reconnect is
    not automatic — callers that want it construct a fresh
    ``WsLinkClient`` and ``open()`` again.
    """

    def __init__(
        self,
        url: str,
        *,
        ssl_context: Any = None,
        ping_interval: float | None = 20.0,
        ping_timeout: float | None = 20.0,
        open_timeout: float | None = 10.0,
    ) -> None:
        # __init__ stores params; side effects deferred to open().
        self._url = url
        self._ssl = ssl_context
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._open_timeout = open_timeout
        self._ws: ClientConnection | None = None
        self._closed = False
        self._endpoint_id = ""

    @property
    def endpoint_id(self) -> str:
        return self._endpoint_id

    async def open(self) -> None:
        """Connect to the hub over WebSocket.

        Does not send a :class:`HelloFrame` — callers send the
        handshake explicitly via :meth:`send_frame` so they can carry
        their own auth claim and optional ``since_envelope_id``. The
        hub responds with :class:`WelcomeFrame`, which this client
        observes in :meth:`frames` and uses to populate
        :attr:`endpoint_id`.
        """
        if self._ws is not None:
            return
        self._ws = await _ws_connect(
            self._url,
            ssl=self._ssl,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
            open_timeout=self._open_timeout,
        )

    async def send_frame(self, frame: Frame) -> None:
        if self._closed or self._ws is None:
            return
        try:
            await self._ws.send(json.dumps(encode_frame(frame)))
        except ConnectionClosed:
            return

    def frames(self) -> AsyncIterator[Frame]:
        return self._frames_impl()

    async def _frames_impl(self) -> AsyncIterator[Frame]:
        if self._ws is None:
            return
        try:
            async for message in self._ws:
                # ``websockets`` yields ``str | bytes`` depending on the
                # message frame. We send only JSON text frames; if the
                # peer ever sends bytes we coerce defensively.
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                try:
                    frame = decode_frame(json.loads(message))
                except Exception:
                    logger.warning("ws frame decode failed", exc_info=True)
                    continue
                if hasattr(frame, "endpoint_id") and getattr(frame, "kind", "") == "welcome":
                    # Welcome carries the server-assigned endpoint id;
                    # cache it so HubClient (or test code) can address
                    # this connection without re-reading the WAL.
                    self._endpoint_id = getattr(frame, "endpoint_id", "")
                yield frame
        except ConnectionClosed:
            return

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()


class WsLinkEndpoint:
    """Hub-side handle for one connected WebSocket client.

    Constructed by :func:`serve_ws` on every incoming connection and
    passed to :meth:`Hub.attach_endpoint`. The hub's frame-processor
    task drives :meth:`frames`; outbound dispatch goes through
    :meth:`send_frame`. ``agent_id`` is populated by
    :meth:`Hub.bind_endpoint` after the client's :class:`HelloFrame`
    is validated.
    """

    def __init__(self, endpoint_id: str, ws: "ServerConnection") -> None:
        # __init__ stores params; no side effects.
        self.endpoint_id = endpoint_id
        self.agent_id: str | None = None
        self._ws = ws
        self._closed = False

    async def send_frame(self, frame: Frame) -> None:
        if self._closed:
            return
        try:
            await self._ws.send(json.dumps(encode_frame(frame)))
        except ConnectionClosed:
            return

    def frames(self) -> AsyncIterator[Frame]:
        return self._frames_impl()

    async def _frames_impl(self) -> AsyncIterator[Frame]:
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                try:
                    yield decode_frame(json.loads(message))
                except Exception:
                    logger.warning("ws frame decode failed", exc_info=True)
                    continue
        except ConnectionClosed:
            return

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            await self._ws.close()


class WsLink:
    """Factory for :class:`WsLinkClient` against a single hub URL.

    Mirrors the :class:`LocalLink` shape so callers that hold a link
    handle and request clients on demand work identically over
    WebSocket. Each :meth:`client` call returns a fresh, unopened
    :class:`WsLinkClient`; the caller is responsible for
    ``await client.open()``.
    """

    def __init__(
        self,
        url: str,
        *,
        ssl_context: Any = None,
        ping_interval: float | None = 20.0,
        ping_timeout: float | None = 20.0,
        open_timeout: float | None = 10.0,
    ) -> None:
        # __init__ stores params; no side effects.
        self._url = url
        self._ssl = ssl_context
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._open_timeout = open_timeout

    @property
    def url(self) -> str:
        return self._url

    def client(self) -> WsLinkClient:
        return WsLinkClient(
            self._url,
            ssl_context=self._ssl,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
            open_timeout=self._open_timeout,
        )


async def _serve_connection(hub: "Hub", ws: "ServerConnection") -> None:
    """Handle one inbound WebSocket connection for :func:`serve_ws`.

    Wraps the socket in a :class:`WsLinkEndpoint`, attaches it to the
    hub, and holds the connection open until either side closes.
    """
    endpoint = WsLinkEndpoint(endpoint_id=make_id(), ws=ws)
    hub.attach_endpoint(endpoint)
    try:
        # ``websockets`` returns from the handler when the client closes;
        # we additionally await ``wait_closed`` to handle a
        # server-initiated close cleanly.
        await ws.wait_closed()
    finally:
        with contextlib.suppress(Exception):
            await endpoint.close()


@contextlib.asynccontextmanager
async def serve_ws(
    hub: "Hub",
    host: str,
    port: int,
    *,
    ssl_context: Any = None,
    ping_interval: float | None = 20.0,
    ping_timeout: float | None = 20.0,
) -> AsyncIterator["_WsServer"]:
    """Run a WebSocket server bound to a hub.

    Each incoming connection becomes a :class:`WsLinkEndpoint`
    attached via :meth:`Hub.attach_endpoint`; the hub's existing
    frame-processor handles routing. Yields the underlying
    ``websockets`` ``Server`` so callers can inspect the bound
    address (``server.sockets[0].getsockname()`` is the usual way to
    read the actual port when ``port=0`` was passed).

    The context manager closes the server on exit and waits for all
    handler tasks to finish so the hub's endpoint registry is clean.
    """
    server = await _ws_serve(
        functools.partial(_serve_connection, hub),
        host,
        port,
        ssl=ssl_context,
        ping_interval=ping_interval,
        ping_timeout=ping_timeout,
    )
    try:
        yield server
    finally:
        server.close()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await server.wait_closed()
