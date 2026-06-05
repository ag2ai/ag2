# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.shared.auth import ProtectedResourceMetadata
from pydantic import AnyHttpUrl


@dataclass(frozen=True, slots=True)
class Scheme:
    """A named OAuth 2.0 authorization server that may issue tokens for this MCP
    resource server.

    MCP authorization is bearer-only — RFC 9728 Protected Resource Metadata
    advertises a list of authorization servers — so this is the single scheme
    kind (cf. A2A's ``bearer_scheme`` / ``api_key_scheme`` / ``oauth2_scheme``
    variants). Build one with :func:`oauth2_scheme`; pass ``Scheme`` objects to
    :func:`require` to build a :class:`Requirement`."""

    url: str


@dataclass(frozen=True, slots=True)
class AuthorizationServerMetadata:
    """Make the MCP server FRONT the OAuth authorization-server metadata.

    Use when the real IdP's discovery is incomplete for MCP — e.g. it omits
    ``registration_endpoint`` (so clients can't do Dynamic Client Registration),
    or it doesn't expose RFC 8414 metadata at all. When set on a
    :class:`Requirement`, the server serves ``/.well-known/oauth-authorization-server``
    (and ``/.well-known/openid-configuration``) by proxying ``upstream_oidc_url``
    and overlaying ``issuer`` (+ ``registration_endpoint`` + any ``overrides``).

    In this mode the PRM must advertise THIS server as the authorization server
    (set the scheme url to the server's own base, equal to ``issuer``), and the
    metadata is served at the host-root well-known paths so it matches that
    issuer. Build via :func:`proxy_authorization_server`."""

    issuer: str
    upstream_oidc_url: str
    registration_endpoint: str | None = None
    overrides: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Requirement:
    """The OAuth 2.0 Resource Server security requirement for an MCP server.

    Mirrors A2A's ``Requirement``: it declares the auth a remote client must
    satisfy. Unlike A2A (which only advertises), an MCP server also *enforces*,
    so this carries the bring-your-own ``verifier`` and the ``required_scopes``
    enforced on the MCP endpoint. :meth:`to_metadata` renders the raw RFC 9728
    ``ProtectedResourceMetadata`` served at
    ``/.well-known/oauth-protected-resource`` (cf. A2A ``Requirement.to_proto``).

    ``authorization_server`` optionally makes this server FRONT the AS metadata
    (see :class:`AuthorizationServerMetadata`).

    Build via :func:`require`."""

    schemes: tuple[Scheme, ...]
    verifier: TokenVerifier
    resource_url: str
    required_scopes: tuple[str, ...] = ()
    resource_name: str | None = None
    resource_documentation: str | None = None
    authorization_server: AuthorizationServerMetadata | None = None

    def to_metadata(self) -> ProtectedResourceMetadata:
        """Render this requirement as RFC 9728 ``ProtectedResourceMetadata``."""
        return ProtectedResourceMetadata(
            resource=AnyHttpUrl(self.resource_url),
            authorization_servers=[AnyHttpUrl(s.url) for s in self.schemes],
            scopes_supported=list(self.required_scopes) or None,
            resource_name=self.resource_name,
            resource_documentation=(AnyHttpUrl(self.resource_documentation) if self.resource_documentation else None),
        )


def oauth2_scheme(*, url: str) -> Scheme:
    """OAuth 2.0 authorization-server declaration (the issuer ``url`` that mints
    tokens for this resource server).

    ``url`` must be an absolute ``http(s)`` URL (RFC 9728 advertises it as such).
    An OIDC issuer *string* like ``stytch.com/project-...`` is not usable here —
    pass the full URL whose ``/.well-known/...`` metadata resolves."""
    if not url.startswith(("http://", "https://")):
        raise ValueError(
            f"oauth2_scheme url must be an absolute http(s) URL, got {url!r} "
            "(an OIDC issuer string is not a usable authorization-server URL)."
        )
    return Scheme(url=url)


def proxy_authorization_server(
    *,
    issuer: str,
    upstream_oidc_url: str,
    registration_endpoint: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AuthorizationServerMetadata:
    """Build an :class:`AuthorizationServerMetadata` that proxies an upstream
    IdP's OpenID config and overlays ``issuer`` (+ optional ``registration_endpoint``
    / ``overrides``). Pass it to :func:`require` as ``authorization_server``."""
    return AuthorizationServerMetadata(
        issuer=issuer,
        upstream_oidc_url=upstream_oidc_url,
        registration_endpoint=registration_endpoint,
        overrides=overrides,
    )


def require(
    *schemes: Scheme,
    resource_url: str,
    verifier: TokenVerifier,
    required_scopes: Sequence[str] = (),
    resource_name: str | None = None,
    resource_documentation: str | None = None,
    authorization_server: AuthorizationServerMetadata | None = None,
) -> Requirement:
    """Build a :class:`Requirement` from one or more authorization-server schemes.

    ``resource_url`` is this MCP server's public endpoint (the RFC 9728 resource
    identifier); ``verifier`` validates presented bearer tokens; a token must
    carry every scope in ``required_scopes``. ``authorization_server`` optionally
    makes the server front the AS metadata (see :func:`proxy_authorization_server`).

    Example::

        from autogen.beta.mcp.security import oauth2_scheme, require

        security = require(
            oauth2_scheme(url="https://auth.example.com"),
            resource_url="https://api.example.com/mcp",
            verifier=my_verifier,
            required_scopes=["mcp.read"],
        )
        app = MCPServer(agent).build_streamable_http(security=security)
    """
    return Requirement(
        schemes=schemes,
        verifier=verifier,
        resource_url=resource_url,
        required_scopes=tuple(required_scopes),
        resource_name=resource_name,
        resource_documentation=resource_documentation,
        authorization_server=authorization_server,
    )


__all__ = (
    "AccessToken",
    "AuthorizationServerMetadata",
    "Requirement",
    "Scheme",
    "TokenVerifier",
    "oauth2_scheme",
    "proxy_authorization_server",
    "require",
)
