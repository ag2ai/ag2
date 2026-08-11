# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from mcp import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session

from .server import MCPServer


@asynccontextmanager
async def connect(
    mcp_server: MCPServer,
    *,
    raise_exceptions: bool = True,
    **session_kwargs: object,
) -> AsyncIterator[ClientSession]:
    """Yield an in-process, initialized MCP ``ClientSession`` talking to ``mcp_server``.

    Dispatches directly into the wrapped low-level server over in-memory streams
    (no sockets, no subprocess) — the MCP analog of the A2A ``ASGITransport``
    test factory. Extra keyword arguments (e.g. ``logging_callback`` /
    ``message_handler``) are forwarded to the underlying client session, which is
    how tests observe progress / log notifications.
    """
    async with create_connected_server_and_client_session(
        mcp_server.server,
        raise_exceptions=raise_exceptions,
        **session_kwargs,  # type: ignore[arg-type]
    ) as session:
        yield session


@asynccontextmanager
async def serve(server: MCPServer, *, base_url: str = "http://test") -> AsyncIterator[httpx.AsyncClient]:
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
