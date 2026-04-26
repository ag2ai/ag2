# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator, Callable

import httpx
import pytest_asyncio

from autogen.beta import Agent
from autogen.beta.a2a import A2AConfig, A2AServer

AgentFactory = Callable[[Agent], "AgentEnv"]


class AgentEnv:
    """Holds the in-process server + a configured A2AConfig pointing at it."""

    __slots__ = ("config", "http", "server")

    def __init__(self, server: A2AServer, http: httpx.AsyncClient, config: A2AConfig) -> None:
        self.server = server
        self.http = http
        self.config = config


@pytest_asyncio.fixture
async def serve() -> AsyncIterator[AgentFactory]:
    """Mount an `Agent` behind an in-process A2AServer and return its A2AConfig.

    Usage:

        env = serve(my_agent)
        client = env.config.create()
        await client(...)

    The HTTP transport is in-memory (no sockets) — HITL works because the
    server-side state lives in `task.history` (stateless replay) and any
    follow-up message is routed back to the same in-process server.
    """
    open_clients: list[httpx.AsyncClient] = []

    def _factory(agent: Agent) -> AgentEnv:
        server = A2AServer(agent, url="http://test")
        asgi = server.build_asgi()
        transport = httpx.ASGITransport(app=asgi)
        http = httpx.AsyncClient(transport=transport, base_url="http://test")
        open_clients.append(http)
        config = A2AConfig("http://test", client_factory=lambda: http)
        return AgentEnv(server=server, http=http, config=config)

    try:
        yield _factory
    finally:
        for c in open_clients:
            await c.aclose()
