# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest

from autogen.beta import Agent
from autogen.beta.mcp import MCPServer, authserver
from autogen.beta.mcp.authserver import _overlay
from autogen.beta.mcp.security import (
    AuthorizationServerMetadata,
    oauth2_scheme,
    proxy_authorization_server,
    require,
)
from autogen.beta.testing import TestConfig

_UPSTREAM = {
    "issuer": "stytch.com/project-test",  # the IdP's own (scheme-less) issuer
    "authorization_endpoint": "http://app/oauth/authorize",
    "token_endpoint": "https://idp/oauth2/token",
    "jwks_uri": "https://idp/.well-known/jwks.json",
    "scopes_supported": ["openid", "mcp.read"],
    # note: no registration_endpoint — the exact gap the facade fills
}


class _Verifier:
    async def verify_token(self, token: str):  # pragma: no cover - not exercised
        return None


def _facade_security() -> object:
    return require(
        oauth2_scheme(url="http://srv"),  # PRM advertises THIS server as the AS
        resource_url="http://srv/mcp",
        verifier=_Verifier(),
        authorization_server=proxy_authorization_server(
            issuer="http://srv",
            upstream_oidc_url="https://idp/.well-known/openid-configuration",
            registration_endpoint="https://idp/oauth2/register",
        ),
    )


class TestOverlay:
    def test_injects_issuer_and_registration(self) -> None:
        meta = AuthorizationServerMetadata(
            issuer="http://srv",
            upstream_oidc_url="https://idp/.well-known/openid-configuration",
            registration_endpoint="https://idp/oauth2/register",
        )
        out = _overlay(_UPSTREAM, meta)

        assert out["issuer"] == "http://srv"  # overlaid, not the upstream issuer
        assert out["registration_endpoint"] == "https://idp/oauth2/register"  # injected
        assert out["token_endpoint"] == "https://idp/oauth2/token"  # passed through

    def test_overrides_win(self) -> None:
        meta = AuthorizationServerMetadata(
            issuer="http://srv",
            upstream_oidc_url="x",
            overrides={"authorization_endpoint": "http://srv/authorize"},
        )
        out = _overlay(_UPSTREAM, meta)

        assert out["authorization_endpoint"] == "http://srv/authorize"


@pytest.mark.asyncio
class TestFacadeServed:
    async def test_serves_as_metadata_with_injected_dcr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_fetch(url: str) -> dict:
            assert url == "https://idp/.well-known/openid-configuration"
            return dict(_UPSTREAM)

        monkeypatch.setattr(authserver, "_fetch_oidc", fake_fetch)

        agent = Agent("greeter", config=TestConfig("hi"))
        app = MCPServer(agent).build_streamable_http(security=_facade_security())

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://srv") as client:
            resp = await client.get("/.well-known/oauth-authorization-server")

        assert resp.status_code == 200
        body = resp.json()
        assert body["issuer"] == "http://srv"
        assert body["registration_endpoint"] == "https://idp/oauth2/register"
        assert body["token_endpoint"] == "https://idp/oauth2/token"

    async def test_upstream_is_fetched_once_then_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        async def counting_fetch(url: str) -> dict:
            calls["n"] += 1
            return dict(_UPSTREAM)

        monkeypatch.setattr(authserver, "_fetch_oidc", counting_fetch)

        agent = Agent("greeter", config=TestConfig("hi"))
        app = MCPServer(agent).build_streamable_http(security=_facade_security())

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://srv") as client:
            await client.get("/.well-known/oauth-authorization-server")
            await client.get("/.well-known/openid-configuration")  # same handler/cache

        assert calls["n"] == 1  # fetched once, second request served from cache

    async def test_upstream_failure_returns_503(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def boom(url: str) -> dict:
            raise RuntimeError("idp unreachable")

        monkeypatch.setattr(authserver, "_fetch_oidc", boom)

        agent = Agent("greeter", config=TestConfig("hi"))
        app = MCPServer(agent).build_streamable_http(security=_facade_security())

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://srv") as client:
            resp = await client.get("/.well-known/oauth-authorization-server")

        assert resp.status_code == 503

    async def test_no_facade_means_no_as_route(self) -> None:
        agent = Agent("greeter", config=TestConfig("hi"))
        sec = require(
            oauth2_scheme(url="https://auth.example.com"), resource_url="http://srv/mcp", verifier=_Verifier()
        )
        app = MCPServer(agent).build_streamable_http(security=sec)

        paths = {getattr(r, "path", None) for r in app.routes}
        assert "/.well-known/oauth-authorization-server" not in paths
