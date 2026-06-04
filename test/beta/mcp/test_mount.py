# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from autogen.beta import Agent
from autogen.beta.mcp import MCPServer
from autogen.beta.mcp.security import AccessToken, oauth2_scheme, require
from autogen.beta.testing import TestConfig

_INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
}
_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


class _StaticVerifier:
    def __init__(self, token: str, scopes: list[str]) -> None:
        self._token = token
        self._scopes = scopes

    async def verify_token(self, token: str) -> AccessToken | None:
        if token != self._token:
            return None
        return AccessToken(token=token, client_id="c", scopes=self._scopes)


async def _health(_request: httpx.Request) -> JSONResponse:
    return JSONResponse({"ok": True})


@pytest.mark.asyncio
class TestMountInto:
    async def test_serves_and_preserves_parent(self) -> None:
        started: dict[str, bool] = {}

        @asynccontextmanager
        async def parent_lifespan(_app: Starlette):
            started["yes"] = True
            yield

        parent = Starlette(routes=[Route("/health", _health)], lifespan=parent_lifespan)
        MCPServer(Agent("greeter", config=TestConfig("hi"))).mount_into(parent, path="/mcp", json_response=True)

        async with parent.router.lifespan_context(parent):
            transport = httpx.ASGITransport(app=parent)
            async with httpx.AsyncClient(transport=transport, base_url="http://test", follow_redirects=True) as client:
                health = await client.get("/health")
                init = await client.post("/mcp", headers=_HEADERS, json=_INIT)

        assert started.get("yes") is True  # parent lifespan still ran
        assert health.status_code == 200  # parent routes still work
        assert init.status_code == 200  # MCP works (lifespan wired)
        assert init.json()["result"]["serverInfo"]["name"] == "greeter"

    async def test_auth_scoped_and_well_known_at_host_root(self) -> None:
        security = require(
            oauth2_scheme(url="https://auth.example.com"),
            resource_url="http://test/mcp",
            verifier=_StaticVerifier("good", ["mcp.use"]),
            required_scopes=["mcp.use"],
        )
        parent = Starlette(routes=[Route("/health", _health)])
        MCPServer(Agent("greeter", config=TestConfig("hi"))).mount_into(parent, security=security, json_response=True)

        async with parent.router.lifespan_context(parent):
            transport = httpx.ASGITransport(app=parent)
            async with httpx.AsyncClient(transport=transport, base_url="http://test", follow_redirects=True) as client:
                well_known = await client.get("/.well-known/oauth-protected-resource/mcp")
                no_token = await client.post("/mcp", headers=_HEADERS, json=_INIT)
                authed = await client.post("/mcp", headers={**_HEADERS, "Authorization": "Bearer good"}, json=_INIT)
                health = await client.get("/health")

        assert well_known.status_code == 200  # RFC 9728 metadata served at host root
        assert no_token.status_code == 401  # MCP route is protected
        assert authed.status_code == 200  # valid token reaches MCP
        assert health.status_code == 200  # auth did NOT leak onto the parent's routes
