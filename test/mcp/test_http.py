# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.mcp import MCPServer
from ag2.mcp.testing import serve

from ._helpers import JSON_HEADERS, greeter, initialize_request

# Pinned rather than tracking the newest handshake revision: ``mcp`` 2.0 serves
# both eras, and which one a connection lands in is settled by this request.
_INIT = initialize_request(version="2025-06-18")


@pytest.mark.asyncio
class TestHttpTransport:
    async def test_serves_initialize_as_asgi_app(self) -> None:
        app = MCPServer(greeter(), json_response=True)

        async with serve(app) as client:
            resp = await client.post("/mcp", headers=JSON_HEADERS, json=_INIT)

        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result["serverInfo"]["name"] == "greeter"
        # A silent era switch — which changes capability semantics — cannot pass.
        assert result["protocolVersion"] == "2025-06-18"

    async def test_custom_path(self) -> None:
        app = MCPServer(greeter(), path="/agent", json_response=True)

        async with serve(app) as client:
            on_custom = await client.post("/agent", headers=JSON_HEADERS, json=_INIT)
            on_default = await client.post("/mcp", headers=JSON_HEADERS, json=_INIT)

        assert on_custom.status_code == 200
        assert on_default.status_code == 404
