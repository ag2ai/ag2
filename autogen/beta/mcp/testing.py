# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

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
