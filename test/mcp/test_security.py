# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest

from ag2.mcp import MCPServer
from ag2.mcp.security import AccessToken, Requirement, oauth2_scheme, require
from ag2.mcp.testing import serve

from ._helpers import JSON_HEADERS, greeter, initialize_request

_INIT = initialize_request()


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
    return MCPServer(greeter(), security=_security(required_scopes=required_scopes), json_response=json_response)


def _client(app: MCPServer) -> httpx.AsyncClient:
    """An HTTP client bound to ``app`` without driving its lifespan.

    The auth layer answers before the session manager is reached, so unlike
    :func:`~ag2.mcp.testing.serve` the app need not be started.
    """
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


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
        with pytest.raises(ValueError, match="must match the MCP endpoint path"):
            MCPServer(greeter(), path="/other", security=_security())

    def test_oauth2_scheme_rejects_schemeless_url(self) -> None:
        # An OIDC issuer string (e.g. Stytch's) is not a usable AS URL — fail
        # early with a clear message, not a cryptic AnyHttpUrl error later.
        with pytest.raises(ValueError, match="absolute http"):
            oauth2_scheme(url="stytch.com/project-test")


@pytest.mark.asyncio
async def test_the_protected_resource_metadata_endpoint_describes_this_server() -> None:
    async with _client(_app(required_scopes=["mcp.read"])) as client:
        resp = await client.get("/.well-known/oauth-protected-resource/mcp")

    assert resp.status_code == 200
    body = resp.json()
    assert body["resource"].rstrip("/") == "http://test/mcp"
    assert body["authorization_servers"] == ["https://auth.example.com/"]
    assert body["scopes_supported"] == ["mcp.read"]


@pytest.mark.asyncio
class TestEnforcement:
    async def test_missing_token_is_401_with_metadata_hint(self) -> None:
        async with _client(_app()) as client:
            resp = await client.post("/mcp", json=_INIT)

        assert resp.status_code == 401
        assert "resource_metadata=" in resp.headers.get("www-authenticate", "")

    async def test_bad_token_is_401(self) -> None:
        async with _client(_app()) as client:
            resp = await client.post("/mcp", headers={"Authorization": "Bearer nope"}, json=_INIT)

        assert resp.status_code == 401

    async def test_insufficient_scope_is_403(self) -> None:
        async with _client(_app(required_scopes=["mcp.admin"])) as client:  # verifier only grants mcp.read
            resp = await client.post("/mcp", headers={"Authorization": "Bearer good-token"}, json=_INIT)

        assert resp.status_code == 403

    async def test_valid_token_reaches_mcp(self) -> None:
        app = _app(required_scopes=["mcp.read"], json_response=True)

        async with serve(app) as client:
            resp = await client.post("/mcp", headers={**JSON_HEADERS, "Authorization": "Bearer good-token"}, json=_INIT)

        # Auth passed (not 401/403) and the MCP layer handled the initialize handshake.
        assert resp.status_code == 200
        assert resp.json()["result"]["serverInfo"]["name"] == "greeter"


class _ResourceVerifier:
    """A verifier whose one token carries ``resource`` — or deliberately does not."""

    def __init__(self, resource: str | None) -> None:
        self._resource = resource

    async def verify_token(self, token: str) -> AccessToken | None:
        if token != "good-token":
            return None
        return AccessToken(token=token, client_id="demo-client", scopes=["mcp.read"], resource=self._resource)


def _resource_app(*, token_resource: str | None, validate: bool) -> MCPServer:
    return MCPServer(
        greeter(),
        json_response=True,
        security=require(
            oauth2_scheme(url="https://auth.example.com"),
            resource_url="http://test/mcp",
            verifier=_ResourceVerifier(token_resource),
            required_scopes=["mcp.read"],
            validate_token_resource=validate,
        ),
    )


@pytest.mark.asyncio
class TestResourceIndicator:
    """RFC 8707: a token minted for another service must not be replayable here."""

    async def test_a_token_issued_for_this_server_is_accepted(self) -> None:
        async with serve(_resource_app(token_resource="http://test/mcp", validate=True)) as client:
            resp = await client.post("/mcp", headers={**JSON_HEADERS, "Authorization": "Bearer good-token"}, json=_INIT)

        assert resp.status_code == 200
        assert resp.json()["result"]["serverInfo"]["name"] == "greeter"

    async def test_a_token_issued_for_another_service_is_refused(self) -> None:
        async with serve(_resource_app(token_resource="https://other.example.com/mcp", validate=True)) as client:
            resp = await client.post("/mcp", headers={**JSON_HEADERS, "Authorization": "Bearer good-token"}, json=_INIT)

        assert resp.status_code == 401

    async def test_a_token_carrying_no_resource_indicator_is_refused(self) -> None:
        async with serve(_resource_app(token_resource=None, validate=True)) as client:
            resp = await client.post("/mcp", headers={**JSON_HEADERS, "Authorization": "Bearer good-token"}, json=_INIT)

        assert resp.status_code == 401

    async def test_off_by_default_a_token_for_another_service_still_works(self) -> None:
        """The promise to deployments already in the field, whose verifiers set no resource."""
        async with serve(_resource_app(token_resource="https://other.example.com/mcp", validate=False)) as client:
            named_elsewhere = await client.post(
                "/mcp", headers={**JSON_HEADERS, "Authorization": "Bearer good-token"}, json=_INIT
            )
        async with serve(_resource_app(token_resource=None, validate=False)) as client:
            unnamed = await client.post(
                "/mcp", headers={**JSON_HEADERS, "Authorization": "Bearer good-token"}, json=_INIT
            )

        assert named_elsewhere.status_code == 200
        assert unnamed.status_code == 200
