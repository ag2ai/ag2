# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator, Callable

import httpx
import pytest_asyncio

from autogen.beta import Agent
from autogen.beta.a2a import A2AConfig, A2AServer


@pytest_asyncio.fixture
async def serve() -> AsyncIterator[Callable[[Agent], A2AConfig]]:
    open_clients: list[httpx.AsyncClient] = []

    def _factory(agent: Agent) -> A2AConfig:
        server = A2AServer(agent, url="http://test")
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.build_asgi()), base_url="http://test")
        open_clients.append(http)
        return A2AConfig("http://test", client_factory=lambda: http)

    try:
        yield _factory
    finally:
        for client in open_clients:
            await client.aclose()
