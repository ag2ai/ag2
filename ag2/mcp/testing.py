# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from types import TracebackType

import anyio
import httpx
from mcp import ClientSession
from mcp.client import Client
from mcp.server.lowlevel import Server
from mcp.shared.memory import MessageStream, create_client_server_memory_streams
from mcp_types.version import LATEST_MODERN_VERSION

from .server import MCPServer


@asynccontextmanager
async def connect(
    mcp_server: MCPServer,
    *,
    raise_exceptions: bool = True,
    **session_kwargs: object,
) -> AsyncGenerator[ClientSession]:
    """Yield an in-process, initialized MCP ``ClientSession`` talking to ``mcp_server``.

    Dispatches directly into the wrapped low-level server over in-memory streams
    (no sockets, no subprocess). Extra keyword arguments (e.g.
    ``logging_callback`` / ``message_handler``) are forwarded to the underlying
    client session, which is how tests observe progress and log notifications.
    """
    async with (
        _served_streams(mcp_server.server, raise_exceptions) as streams,
        ClientSession(*streams, **session_kwargs) as session,  # type: ignore[arg-type]
    ):
        await session.initialize()
        yield session


@asynccontextmanager
async def connect_modern(
    mcp_server: MCPServer,
    *,
    raise_exceptions: bool = True,
    **client_kwargs: object,
) -> AsyncGenerator[ClientSession]:
    """Yield an in-process ``ClientSession`` talking to ``mcp_server`` at revision 2026-07-28.

    The modern-era counterpart of :func:`connect`: same shape and same yielded
    contract, with the connection pinned to the modern revision instead of
    negotiating the newest handshake one, over the same in-memory streams.
    """
    async with Client(
        _MemoryTransport(mcp_server.server, raise_exceptions=raise_exceptions),
        mode=LATEST_MODERN_VERSION,
        raise_exceptions=raise_exceptions,
        **client_kwargs,  # type: ignore[arg-type]
    ) as client:
        yield client.session


class _MemoryTransport:
    """An ``mcp.client.Transport`` serving a low-level server over memory streams.

    The SDK's own in-memory transport is private, and this keeps :func:`connect`
    and :func:`connect_modern` on one wire shape.
    """

    __slots__ = ("_server", "_raise_exceptions", "_stack")

    def __init__(self, server: Server, *, raise_exceptions: bool) -> None:
        self._server = server
        self._raise_exceptions = raise_exceptions
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> MessageStream:
        return await self._stack.enter_async_context(_served_streams(self._server, self._raise_exceptions))

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._stack.__aexit__(exc_type, exc, tb)


@asynccontextmanager
async def _served_streams(server: Server, raise_exceptions: bool) -> AsyncGenerator[MessageStream]:
    """Yield the client half of a memory stream pair whose server half is being served."""
    async with (
        create_client_server_memory_streams() as (client_streams, server_streams),
        anyio.create_task_group() as tg,
    ):
        tg.start_soon(_run_server, server, server_streams, raise_exceptions)
        yield client_streams
        # As in ``connect``: the server task runs until cancelled, and the client
        # is done, so end it here rather than leaving the task group waiting.
        tg.cancel_scope.cancel()


async def _run_server(server: Server, streams: MessageStream, raise_exceptions: bool) -> None:
    """Serve the low-level server over ``streams`` until the task group is cancelled."""
    await server.run(*streams, server.create_initialization_options(), raise_exceptions=raise_exceptions)


@asynccontextmanager
async def serve(server: MCPServer, *, base_url: str = "http://test") -> AsyncGenerator[httpx.AsyncClient]:
    """Yield an ``httpx.AsyncClient`` bound to ``server`` over the in-memory ASGI transport.

    Drives the ASGI ``lifespan`` protocol so the streamable-HTTP session manager
    is running, the way ``uvicorn`` would (``httpx.ASGITransport`` does not).
    Use it to exercise the HTTP transport without sockets.
    """
    receive_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    send_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()

    async def receive() -> dict[str, object]:
        return await receive_queue.get()

    async def send(message: dict[str, object]) -> None:
        await send_queue.put(message)

    scope = {"type": "lifespan", "asgi": {"spec_version": "2.0", "version": "3.0"}}
    lifespan_task = asyncio.ensure_future(server(scope, receive, send))

    await receive_queue.put({"type": "lifespan.startup"})
    started = await send_queue.get()
    if started["type"] == "lifespan.startup.failed":
        await lifespan_task
        raise RuntimeError(str(started.get("message", "ASGI lifespan startup failed")))

    try:
        transport = httpx.ASGITransport(app=server)
        async with httpx.AsyncClient(transport=transport, base_url=base_url, follow_redirects=True) as client:
            yield client
    finally:
        await receive_queue.put({"type": "lifespan.shutdown"})
        await send_queue.get()
        await lifespan_task
