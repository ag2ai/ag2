# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Serve OAuth 2.0 Authorization Server metadata FROM the MCP server itself.

For the case where the MCP resource server must front the AS metadata because
the real IdP's discovery is incomplete for MCP — it omits ``registration_endpoint``
(no Dynamic Client Registration) or doesn't expose an RFC 8414 document. The
server proxies the upstream OpenID config and overlays ``issuer`` (+ an injected
``registration_endpoint`` / arbitrary ``overrides``). See
:class:`~autogen.beta.mcp.security.AuthorizationServerMetadata`.
"""

import logging
from typing import Any

from starlette.responses import JSONResponse
from starlette.routing import BaseRoute, Route

from .security import AuthorizationServerMetadata

_LOGGER_NAME = "ag2.mcp"


def _overlay(upstream: dict[str, Any], meta: AuthorizationServerMetadata) -> dict[str, Any]:
    """Build the served AS metadata: the upstream OIDC doc with ``issuer`` (and
    optional ``registration_endpoint`` / ``overrides``) overlaid. Pure helper."""
    doc = dict(upstream)
    doc["issuer"] = meta.issuer
    if meta.registration_endpoint:
        doc["registration_endpoint"] = meta.registration_endpoint
    if meta.overrides:
        doc.update(meta.overrides)
    return doc


async def _fetch_oidc(url: str) -> dict[str, Any]:
    """Fetch an upstream OpenID-configuration document (module-level so tests can
    monkeypatch it without hitting the network)."""
    import httpx

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


def authorization_server_routes(meta: AuthorizationServerMetadata) -> list[BaseRoute]:
    """Routes serving RFC 8414 + OIDC authorization-server metadata for the AS
    facade, by proxying ``meta.upstream_oidc_url`` (fetched once and cached) and
    overlaying it via :func:`_overlay`. Served at the host-root well-known paths
    so they match an issuer equal to the server's own base."""
    logger = logging.getLogger(_LOGGER_NAME)
    cache: dict[str, Any] = {}

    async def _upstream() -> dict[str, Any]:
        if not cache.get("doc"):
            cache["doc"] = await _fetch_oidc(meta.upstream_oidc_url)
        return cache["doc"]

    async def handler(_request: Any) -> JSONResponse:
        try:
            upstream = await _upstream()
        except Exception as exc:  # upstream IdP unreachable — surface, stay up
            logger.warning("MCP authorization-server facade: upstream fetch failed: %s", exc)
            return JSONResponse({"error": "upstream_unavailable"}, status_code=503)
        return JSONResponse(_overlay(upstream, meta))

    return [
        Route("/.well-known/oauth-authorization-server", handler, methods=["GET"]),
        Route("/.well-known/openid-configuration", handler, methods=["GET"]),
    ]
