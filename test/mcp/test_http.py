# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Agent
from ag2.mcp import MCPServer
from ag2.mcp.testing import serve
from ag2.testing import TestConfig

_INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
}
_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


@pytest.mark.asyncio
class TestHttpTransport:
    async def test_serves_initialize_as_asgi_app(self) -> None:
        app = MCPServer(Agent("greeter", config=TestConfig("hi")), json_response=True)

        async with serve(app) as client:
            resp = await client.post("/mcp", headers=_HEADERS, json=_INIT)

        assert resp.status_code == 200
        assert resp.json()["result"]["serverInfo"]["name"] == "greeter"

    async def test_custom_path(self) -> None:
        app = MCPServer(Agent("greeter", config=TestConfig("hi")), path="/agent", json_response=True)

        async with serve(app) as client:
            on_custom = await client.post("/agent", headers=_HEADERS, json=_INIT)
            on_default = await client.post("/mcp", headers=_HEADERS, json=_INIT)

        assert on_custom.status_code == 200
        assert on_default.status_code == 404
