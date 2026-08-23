# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import anyio
import httpx
from mcp import ClientSession
from mcp.server.lowlevel import Server
from mcp.shared.memory import MessageStream, create_client_server_memory_streams

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
    (no sockets, no subprocess) — the MCP analog of the A2A ``ASGITransport``
    test factory. Extra keyword arguments (e.g. ``logging_callback`` /
    ``message_handler``) are forwarded to the underlying client session, which is
    how tests observe progress / log notifications.

    Built on the memory-stream primitive rather than on ``mcp``'s own
    connected-server helper, which 2.0 removed in favour of a differently-shaped
    client object. A testing helper exists to absorb that kind of churn, so the
    contract here — an initialized ``ClientSession`` — is held steady across it.
    """
    async with (
        create_client_server_memory_streams() as (client_streams, server_streams),
        # The same background state the ASGI lifespan and ``run_stdio`` enter:
        # subscription delivery lives there, so a session without it would be
        # served by a server that is only half-running.
        mcp_server._serving(),
        anyio.create_task_group() as tg,
    ):
        tg.start_soon(_run_server, mcp_server.server, server_streams, raise_exceptions)
        async with ClientSession(*client_streams, **session_kwargs) as session:  # type: ignore[arg-type]
            await session.initialize()
            yield session
        # The server task runs until cancelled; the client is done, so end it here
        # rather than leaving the task group waiting on it.
        tg.cancel_scope.cancel()


async def _run_server(server: Server, streams: MessageStream, raise_exceptions: bool) -> None:
    """Serve the low-level server over ``streams`` until the task group is cancelled."""
    await server.run(*streams, server.create_initialization_options(), raise_exceptions=raise_exceptions)


@asynccontextmanager
async def serve(server: MCPServer, *, base_url: str = "http://test") -> AsyncGenerator[httpx.AsyncClient]:
    """Yield an ``httpx.AsyncClient`` bound to ``server`` over the in-memory ASGI transport.

    Drives the ASGI ``lifespan`` protocol so the streamable-HTTP session manager
    is running (``httpx.ASGITransport`` does not manage lifespan itself), the way
    ``uvicorn`` would. Use it to exercise the HTTP transport — POST to ``path``,
    GET the protected-resource metadata, assert status codes — without sockets.
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
