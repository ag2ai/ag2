# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("anthropic")

import httpx
import uvicorn
from dirty_equals import IsPartialDict

from examples.a2a import server_basic


@pytest.mark.asyncio
async def test_basic_server_example_serves_its_agent_card(monkeypatch: pytest.MonkeyPatch) -> None:
    cards = []

    async def serve(server: uvicorn.Server) -> None:
        assert (server.config.host, server.config.port) == ("127.0.0.1", 8000)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.config.app), base_url="http://127.0.0.1:8000"
        ) as client:
            response = await client.get("/.well-known/agent-card.json")
        assert response.status_code == 200
        cards.append(response.json())

    monkeypatch.setattr(uvicorn.Server, "serve", serve)

    await server_basic.main()

    assert cards == [
        IsPartialDict({
            "name": "claude",
            "supportedInterfaces": [IsPartialDict({"url": "http://127.0.0.1:8000"})],
        }),
    ]
