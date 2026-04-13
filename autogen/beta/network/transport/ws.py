# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""WsLink — WebSocket transport for the V3 Link protocol.

Phase 3a's cross-process story. The frame vocabulary is the same as
:class:`LocalLink`; the encoder is the existing :func:`encode_frame` /
:func:`decode_frame` JSON-line format, so every Phase 2 frame type
rides over WebSocket unchanged.

Two roles:

* :class:`WsLinkServer` — hub-side listener. Wraps
  :func:`websockets.serve`, dispatches each accepted connection to
  a handler supplied via :meth:`on_connection` (typically
  :meth:`Hub.connection_handler`), and surfaces the bound URL via
  :attr:`url` for clients to connect to.
* :class:`WsLinkClient` — actor-side connection factory. Satisfies
  the :class:`Link` protocol so a :class:`HubClient` can treat it
  the same way as a :class:`LocalLink`. Each :meth:`client` call
  returns a :class:`_WsClientSide` that lazily opens a new
  WebSocket to the hub the first time a frame is sent.

The split mirrors the real deployment topology: hubs run a single
:class:`WsLinkServer` accepting N actor connections, while each
actor process instantiates one :class:`WsLinkClient` pointing at
the hub URL and registers any number of identities through that
one client.

Lazy connect: ``Link.client()`` is a sync method in the protocol
(LocalLink needs to stay sync), so the WebSocket dial happens on
first ``send_frame`` or first ``frames()`` iteration. This is what
lets ``HubClient.register`` call ``link.client()`` synchronously
and then ``await client.send_frame(HelloFrame(...))`` to trigger
the dial and handshake in one go.

Binding: server-side ``_WsEndpointSide.binding`` is ``"ws"`` so
:meth:`Hub._write_runtime` stamps the actor's ``runtime.json`` with
the ``ws`` binding + the hub's public URL.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from ..errors import LinkClosedError, TransportError
from ..ids import new_id
from .frames import Frame, decode_frame, encode_frame

__all__ = (
    "WsLinkClient",
    "WsLinkServer",
)

log = logging.getLogger("autogen.beta.network.transport.ws")


def _require_websockets() -> tuple[Any, Any]:
    """Lazy-import ``websockets`` so the network package stays install-optional.

    Returns ``(connect, serve)`` from the modern asyncio API. Raises a
    :class:`TransportError` with a clear install hint if the library
    is not available.
    """

    try:
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve
    except ImportError as exc:
        raise TransportError(
            "WsLink requires the 'websockets' library. "
            "Install with: pip install 'ag2[websockets]'"
        ) from exc
    return connect, serve


# ---------------------------------------------------------------------------
# Client side
# ---------------------------------------------------------------------------


class _WsClientSide:
    """Client handle returned by :meth:`WsLinkClient.client`.

    Lazily opens a WebSocket to the hub on first :meth:`send_frame`
    (or first :meth:`frames` iteration). A single instance is tied
    to one underlying WebSocket; disconnects raise
    :class:`LinkClosedError` on subsequent sends, and the caller
    must construct a new instance (via ``WsLinkClient.client()``)
    to reconnect — which is exactly what
    :meth:`ActorClient.reconnect` does.
    """

    def __init__(self, url: str, *, open_timeout: float = 5.0) -> None:
        self._url = url
        self._open_timeout = open_timeout
        self._ws: Any = None
        self._closed = False
        self._connect_lock = asyncio.Lock()

    async def _ensure_connected(self) -> Any:
        if self._ws is not None:
            return self._ws
        async with self._connect_lock:
            if self._ws is not None:
                return self._ws
            if self._closed:
                raise LinkClosedError("client is closed")
            connect, _ = _require_websockets()
            try:
                self._ws = await asyncio.wait_for(
                    connect(self._url),
                    timeout=self._open_timeout,
                )
            except Exception as exc:
                raise TransportError(
                    f"failed to connect to {self._url}: {exc}"
                ) from exc
        return self._ws

    async def send_frame(self, frame: Frame) -> None:
        if self._closed:
            raise LinkClosedError("client is closed")
        ws = await self._ensure_connected()
        try:
            await ws.send(encode_frame(frame))
        except Exception as exc:
            raise LinkClosedError(f"send failed: {exc}") from exc

    def frames(self) -> AsyncIterator[Frame]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Frame]:
        ws = await self._ensure_connected()
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                yield decode_frame(message)
        except Exception:
            # Connection closed cleanly or with an error — either way
            # the iterator ends and the caller sees it as EOF on the
            # frame stream.
            return

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        ws = self._ws
        self._ws = None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()

    @property
    def closed(self) -> bool:
        return self._closed


class WsLinkClient:
    """Actor-side :class:`Link` factory pointed at a hub WebSocket URL.

    Each :meth:`client` call produces a fresh :class:`_WsClientSide`
    that opens a new WebSocket lazily on first use. A single
    ``WsLinkClient`` can serve any number of registered identities;
    each registration drives its own underlying connection.
    """

    def __init__(self, url: str, *, open_timeout: float = 5.0) -> None:
        self._url = url
        self._open_timeout = open_timeout
        self._closed = False

    @property
    def url(self) -> str:
        return self._url

    def client(self) -> _WsClientSide:
        if self._closed:
            raise LinkClosedError("link is closed")
        return _WsClientSide(self._url, open_timeout=self._open_timeout)

    async def close(self) -> None:
        # Phase 3a: WsLinkClient itself holds no live connections —
        # every _WsClientSide owns its own. The HubClient is in
        # charge of closing them via its own close/shutdown paths.
        self._closed = True


# ---------------------------------------------------------------------------
# Server side
# ---------------------------------------------------------------------------


class _WsEndpointSide:
    """Hub-side peer handle for a single WebSocket connection.

    Wraps a ``websockets`` server-side connection and exposes the
    same surface as :class:`~.local._EndpointSide` (the LocalLink
    peer): ``endpoint_id``, ``actor_id``, ``send_frame``, ``frames``,
    ``close``, ``closed``. The extra ``binding="ws"`` plus
    ``ws_url`` attributes let :meth:`Hub._handle_hello` stamp the
    actor's ``runtime.json`` with the right transport flavor.
    """

    binding: str = "ws"

    def __init__(self, connection: Any, *, ws_url: str | None = None) -> None:
        self.endpoint_id = new_id()
        self.actor_id: str | None = None
        self.ws_url = ws_url
        self.http_url: str | None = None
        self._ws = connection
        self._closed = False

    async def send_frame(self, frame: Frame) -> None:
        if self._closed:
            raise LinkClosedError("endpoint is closed")
        try:
            await self._ws.send(encode_frame(frame))
        except Exception as exc:
            raise LinkClosedError(f"send failed: {exc}") from exc

    def frames(self) -> AsyncIterator[Frame]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Frame]:
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                yield decode_frame(message)
        except Exception:
            return

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            await self._ws.close()

    @property
    def closed(self) -> bool:
        return self._closed


ConnectionCallback = Callable[[_WsEndpointSide], Awaitable[None]]


class WsLinkServer:
    """Hub-side WebSocket listener.

    Usage::

        server = WsLinkServer(host="127.0.0.1", port=0)
        server.on_connection(hub.connection_handler)
        await server.start()
        print(server.url)  # ws://127.0.0.1:<port>/
        # ... run ...
        await server.close()

    ``port=0`` asks the OS for a free port and :attr:`url` surfaces
    the actual bound address once :meth:`start` has returned. This
    is critical for tests: they can spawn a server on a random
    port and hand the URL to a ``WsLinkClient`` without hardcoding.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._host = host
        self._port = port
        self._handler: ConnectionCallback | None = None
        self._server: Any = None
        self._url: str | None = None
        self._closed = False
        self._pending: set[asyncio.Task[None]] = set()

    def on_connection(self, handler: ConnectionCallback) -> None:
        self._handler = handler

    async def start(self) -> None:
        if self._server is not None:
            return
        if self._handler is None:
            raise LinkClosedError("no connection handler registered on WsLinkServer")

        _, serve = _require_websockets()

        async def _on_ws(connection: Any) -> None:
            endpoint = _WsEndpointSide(connection, ws_url=self._url)
            task = asyncio.current_task()
            assert task is not None
            self._pending.add(task)
            try:
                assert self._handler is not None
                await self._handler(endpoint)
            finally:
                self._pending.discard(task)
                with contextlib.suppress(Exception):
                    await endpoint.close()

        self._server = await serve(_on_ws, self._host, self._port)
        # Pick the first bound socket address and construct the URL.
        sockets = list(self._server.sockets or [])
        if not sockets:
            raise TransportError("WsLinkServer failed to bind any sockets")
        host, port = sockets[0].getsockname()[:2]
        # IPv6 addresses need square brackets in URLs.
        if ":" in host:
            host = f"[{host}]"
        self._url = f"ws://{host}:{port}/"

    @property
    def url(self) -> str:
        if self._url is None:
            raise LinkClosedError("WsLinkServer is not started")
        return self._url

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._server is not None:
            try:
                self._server.close()
                await self._server.wait_closed()
            except Exception:  # pragma: no cover
                log.warning("WsLinkServer.close failed", exc_info=True)
            self._server = None
        for task in list(self._pending):
            if not task.done():
                task.cancel()
        for task in list(self._pending):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
