# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest

from ag2 import Agent
from ag2.mcp import MCPServer
from ag2.mcp.security import AccessToken, Requirement, oauth2_scheme, require
from ag2.mcp.testing import serve
from ag2.testing import TestConfig

_INIT_BODY = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1"},
    },
}


class _StaticVerifier:
    """Bring-your-own TokenVerifier accepting one token with fixed scopes."""

    def __init__(self, token: str, scopes: list[str]) -> None:
        self._token = token
        self._scopes = scopes

    async def verify_token(self, token: str) -> AccessToken | None:
        if token != self._token:
            return None
        return AccessToken(token=token, client_id="demo-client", scopes=self._scopes)


def _security(*, required_scopes: list[str] | None = None) -> Requirement:
    return require(
        oauth2_scheme(url="https://auth.example.com"),
        resource_url="http://test/mcp",
        verifier=_StaticVerifier("good-token", ["mcp.read"]),
        required_scopes=required_scopes or [],
        resource_name="AG2 demo",
    )


def _app(*, required_scopes: list[str] | None = None, json_response: bool = False) -> MCPServer:
    agent = Agent("greeter", config=TestConfig("hi"))
    return MCPServer(agent, security=_security(required_scopes=required_scopes), json_response=json_response)


class TestSecurityBuilders:
    def test_to_metadata(self) -> None:
        metadata = _security(required_scopes=["mcp.read"]).to_metadata()

        assert str(metadata.resource).rstrip("/") == "http://test/mcp"
        assert [str(u) for u in metadata.authorization_servers] == ["https://auth.example.com/"]
        assert metadata.scopes_supported == ["mcp.read"]
        assert metadata.bearer_methods_supported == ["header"]
        assert metadata.resource_name == "AG2 demo"

    def test_no_scopes_omits_scopes_supported(self) -> None:
        metadata = _security().to_metadata()

        assert metadata.scopes_supported is None

    def test_multiple_authorization_servers(self) -> None:
        sec = require(
            oauth2_scheme(url="https://auth1.example.com"),
            oauth2_scheme(url="https://auth2.example.com"),
            resource_url="http://test/mcp",
            verifier=_StaticVerifier("t", []),
        )

        assert [str(u) for u in sec.to_metadata().authorization_servers] == [
            "https://auth1.example.com/",
            "https://auth2.example.com/",
        ]

    def test_path_mismatch_raises(self) -> None:
        agent = Agent("greeter", config=TestConfig("hi"))
        with pytest.raises(ValueError, match="must match the MCP endpoint path"):
            MCPServer(agent, path="/other", security=_security())

    def test_oauth2_scheme_rejects_schemeless_url(self) -> None:
        # An OIDC issuer string (e.g. Stytch's) is not a usable AS URL — fail
        # early with a clear message, not a cryptic AnyHttpUrl error later.
        with pytest.raises(ValueError, match="absolute http"):
            oauth2_scheme(url="stytch.com/project-test")


@pytest.mark.asyncio
class TestProtectedResourceMetadata:
    async def test_well_known_endpoint(self) -> None:
        app = _app(required_scopes=["mcp.read"])
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/.well-known/oauth-protected-resource/mcp")

        assert resp.status_code == 200
        body = resp.json()
        assert body["resource"].rstrip("/") == "http://test/mcp"
        assert body["authorization_servers"] == ["https://auth.example.com/"]
        assert body["scopes_supported"] == ["mcp.read"]


@pytest.mark.asyncio
class TestEnforcement:
    async def test_missing_token_is_401_with_metadata_hint(self) -> None:
        app = _app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/mcp", json=_INIT_BODY)

        assert resp.status_code == 401
        assert "resource_metadata=" in resp.headers.get("www-authenticate", "")

    async def test_bad_token_is_401(self) -> None:
        app = _app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/mcp", headers={"Authorization": "Bearer nope"}, json=_INIT_BODY)

        assert resp.status_code == 401

    async def test_insufficient_scope_is_403(self) -> None:
        app = _app(required_scopes=["mcp.admin"])  # verifier only grants mcp.read
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/mcp", headers={"Authorization": "Bearer good-token"}, json=_INIT_BODY)

        assert resp.status_code == 403

    async def test_valid_token_reaches_mcp(self) -> None:
        app = _app(required_scopes=["mcp.read"], json_response=True)
        headers = {
            "Authorization": "Bearer good-token",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        async with serve(app) as client:
            resp = await client.post("/mcp", headers=headers, json=_INIT_BODY)

        # Auth passed (not 401/403) and the MCP layer handled the initialize handshake.
        assert resp.status_code == 200
        assert resp.json()["result"]["serverInfo"]["name"] == "greeter"
