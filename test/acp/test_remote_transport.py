# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""The remote transports against a real ACP agent, served on loopback.

Everywhere else the fake agent replaces the connection hook, which is what keeps
those tests fast and transport-agnostic — but it also means no test there opens a
socket. This file is the other half: an ordinary ``Agent.ask`` reaches an agent
over real HTTP and a real WebSocket, so what a request actually carries (the
configured headers, in particular) is asserted from the receiving end.
"""

import pytest

pytest.importorskip("websockets")
pytest.importorskip("h2")

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Callable
from typing import Any

import acp
import uvicorn
from acp import schema
from acp.http.asgi import create_asgi_app

from ag2 import Agent
from ag2.acp import ACPRemoteConfig

REPLY = "hi from the wire"


class _EchoAgent:
    """A minimal ACP Agent: enough of one to carry a single turn."""

    def __init__(self, conn: acp.Client) -> None:
        self.conn = conn

    async def initialize(self, **kwargs: Any) -> schema.InitializeResponse:
        return schema.InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_info=schema.Implementation(name="echo", version="test"),
        )

    async def new_session(self, **kwargs: Any) -> schema.NewSessionResponse:
        return schema.NewSessionResponse(session_id="remote-session-1")

    async def prompt(self, *, session_id: str, **kwargs: Any) -> schema.PromptResponse:
        await self.conn.session_update(session_id=session_id, update=acp.update_agent_message_text(REPLY))
        return schema.PromptResponse(stop_reason="end_turn")


@contextlib.asynccontextmanager
async def _served_agent() -> AsyncGenerator[tuple[int, list[dict[str, str]]], None]:
    """Serve ``_EchoAgent`` on a loopback port, recording every request's headers.

    The ASGI app handles both the HTTP and the WebSocket transport, so one server
    answers whichever URL scheme a test points at it.
    """
    requests: list[dict[str, str]] = []
    app = create_asgi_app(_EchoAgent)  # type: ignore[arg-type]

    async def recording(scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] in ("http", "websocket"):
            requests.append({k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]})
        await app(scope, receive, send)

    config = uvicorn.Config(recording, host="127.0.0.1", port=0, log_level="warning")
    # Bound here rather than inside `serve()`: the socket is already listening, so
    # the port is known and a connection can be made without waiting for start-up.
    sock = config.bind_socket()
    server = uvicorn.Server(config)
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        yield sock.getsockname()[1], requests
    finally:
        server.should_exit = True
        await serving
        sock.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme", ["http", "ws"], ids=["http", "websocket"])
async def test_a_turn_crosses_a_real_connection(scheme: str) -> None:
    async with _served_agent() as (port, _):
        cfg = ACPRemoteConfig(url=f"{scheme}://127.0.0.1:{port}/acp", expose_tools=False)
        try:
            reply = await Agent("acp", config=cfg).ask("hello")
        finally:
            await cfg.aclose()

    assert reply.body == REPLY


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme", ["http", "ws"], ids=["http", "websocket"])
async def test_configured_headers_are_sent_with_every_request(scheme: str) -> None:
    """This is how a bearer-token gateway is reached, so no request may skip them."""
    async with _served_agent() as (port, requests):
        cfg = ACPRemoteConfig(
            url=f"{scheme}://127.0.0.1:{port}/acp",
            headers={"Authorization": "Bearer t0ken", "X-Tenant": "acme"},
            expose_tools=False,
        )
        try:
            await Agent("acp", config=cfg).ask("hello")
        finally:
            await cfg.aclose()

    assert requests, "the agent was never reached, so nothing was carried"
    assert all(r.get("authorization") == "Bearer t0ken" for r in requests)
    assert all(r.get("x-tenant") == "acme" for r in requests)


@pytest.mark.asyncio
async def test_no_headers_configured_sends_no_authorization() -> None:
    async with _served_agent() as (port, requests):
        cfg = ACPRemoteConfig(url=f"http://127.0.0.1:{port}/acp", expose_tools=False)
        try:
            await Agent("acp", config=cfg).ask("hello")
        finally:
            await cfg.aclose()

    assert requests
    assert not any("authorization" in r for r in requests)
