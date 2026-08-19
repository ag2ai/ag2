# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``MCPToolkit`` against a live HTTP MCP server, through its real session construction.

The rest of ``test/tools/test_mcp.py`` substitutes ``_mcp_session`` for a fake,
which is right for tool proxying and content mapping but leaves the transport
itself — the client, the handshake, the wire — untouched. This module is the one
place that exercises it, so a change to how the streamable-HTTP session is built
cannot ship unexercised.

Lives apart from ``test_mcp.py`` because serving on a real socket needs
``uvicorn``, which ships with ``ag2[acp]`` and not ``ag2[mcp]``; the skip guard
below would otherwise take the fake-session tests down with it.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest

pytest.importorskip("mcp")
pytest.importorskip("uvicorn")

import uvicorn

from ag2 import Agent, Context
from ag2.events import TextInput, ToolCallEvent, ToolResultEvent
from ag2.mcp import MCPServer, mcp_tool
from ag2.testing import TestConfig
from ag2.tools import MCPServerConfig, MCPToolkit


@mcp_tool
def echo(message: str) -> str:
    """Echo the message back."""
    return f"echo: {message}"


@asynccontextmanager
async def _live_mcp_server(
    headers_seen: list[dict[str, str]] | None = None,
) -> AsyncGenerator[str]:
    """Serve an AG2 ``MCPServer`` on a loopback port, yielding the MCP endpoint URL.

    The URL carries the canonical trailing slash. ``test_a_slashless_url_still_reaches
    _the_server`` strips it to cover the redirect a Starlette ``Mount`` issues.

    The served side is AG2's own public serving API rather than a hand-built
    ``mcp`` server, so this test owns no handler registration of its own and
    stays valid across changes to how handlers are registered.

    When ``headers_seen`` is supplied, every request's headers are appended to it,
    which is how a test observes what the toolkit's HTTP client actually sent.
    """
    served = MCPServer(Agent("live", config=TestConfig("unused")), tools=[echo], path="/mcp")
    app = served if headers_seen is None else _recording(served, headers_seen)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    # Bound here rather than inside `serve()`: the socket is already listening, so
    # the port is known and a connection can be made without waiting for start-up.
    sock = config.bind_socket()
    uv = uvicorn.Server(config)
    serving = asyncio.create_task(uv.serve(sockets=[sock]))
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}/mcp/"
    finally:
        uv.should_exit = True
        await serving
        sock.close()


def _recording(app: Any, headers_seen: list[dict[str, str]]) -> Any:
    """Wrap an ASGI app, recording each HTTP request's headers."""

    async def recording(scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        if scope["type"] == "http":
            headers_seen.append({k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]})
        await app(scope, receive, send)

    return recording


@pytest.mark.asyncio
async def test_tools_are_discovered_over_the_real_transport(context: Context) -> None:
    async with _live_mcp_server() as url:
        schemas = list(await MCPToolkit(url).schemas(context))

    # ``ask`` is the served agent's own conversational tool; ``echo`` is the
    # custom tool. Both arriving means the handshake and ``tools/list`` completed.
    assert sorted(s.function.name for s in schemas) == ["ask", "echo"]


@pytest.mark.asyncio
async def test_a_tool_call_round_trips_over_the_real_transport(context: Context) -> None:
    async with _live_mcp_server() as url:
        toolkit = MCPToolkit(url)
        await toolkit.schemas(context)
        proxy = next(t for t in toolkit.tools if t.name == "echo")

        result = await proxy(ToolCallEvent(name="echo", arguments='{"message": "hi"}'), context)

    assert isinstance(result, ToolResultEvent)
    assert result.result.parts == [TextInput(content="echo: hi")]


@pytest.mark.asyncio
async def test_configured_headers_reach_the_server(context: Context) -> None:
    """A bearer-token MCP server is reached this way, so no request may skip them."""
    headers_seen: list[dict[str, str]] = []

    async with _live_mcp_server(headers_seen) as url:
        toolkit = MCPToolkit(MCPServerConfig(server_url=url, headers={"X-Tenant": "acme"}, authorization_token="t0ken"))
        await toolkit.schemas(context)

    assert headers_seen, "no HTTP request reached the server"
    assert all(h.get("x-tenant") == "acme" for h in headers_seen)
    assert all(h.get("authorization") == "Bearer t0ken" for h in headers_seen)


@pytest.mark.asyncio
async def test_a_slashless_url_still_reaches_the_server(context: Context) -> None:
    """A Starlette-mounted endpoint 307s the slashless form, and that form is what
    a caller naturally writes, so the toolkit's client has to follow the redirect.
    """
    async with _live_mcp_server() as url:
        schemas = list(await MCPToolkit(url.rstrip("/")).schemas(context))

    assert sorted(s.function.name for s in schemas) == ["ask", "echo"]
