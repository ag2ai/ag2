# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from dataclasses import dataclass

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
class Requirement:
    """The OAuth 2.0 Resource Server security requirement for an MCP server.

    Mirrors A2A's ``Requirement``: it declares the auth a remote client must
    satisfy. Unlike A2A (which only advertises), an MCP server also *enforces*,
    so this carries the bring-your-own ``verifier`` and the ``required_scopes``
    enforced on the MCP endpoint. :meth:`to_metadata` renders the raw RFC 9728
    ``ProtectedResourceMetadata`` served at
    ``/.well-known/oauth-protected-resource`` (cf. A2A ``Requirement.to_proto``).

    The MCP server is purely an OAuth 2.1 Resource Server here: it advertises the
    trusted authorization server(s) and verifies tokens. Issuing tokens and
    serving authorization-server metadata stay with the external authorization
    server (out of scope per the MCP authorization spec).

    Build via :func:`require`."""

    schemes: tuple[Scheme, ...]
    verifier: TokenVerifier
    resource_url: str
    required_scopes: tuple[str, ...] = ()
    resource_name: str | None = None
    resource_documentation: str | None = None

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


def require(
    *schemes: Scheme,
    resource_url: str,
    verifier: TokenVerifier,
    required_scopes: Sequence[str] = (),
    resource_name: str | None = None,
    resource_documentation: str | None = None,
) -> Requirement:
    """Build a :class:`Requirement` from one or more authorization-server schemes.

    ``resource_url`` is this MCP server's public endpoint (the RFC 9728 resource
    identifier); ``verifier`` validates presented bearer tokens; a token must
    carry every scope in ``required_scopes``.

    Example::

        from ag2.mcp.security import oauth2_scheme, require

        security = require(
            oauth2_scheme(url="https://auth.example.com"),
            resource_url="https://api.example.com/mcp",
            verifier=my_verifier,
            required_scopes=["mcp.read"],
        )
        app = MCPServer(agent, security=security)
    """
    return Requirement(
        schemes=schemes,
        verifier=verifier,
        resource_url=resource_url,
        required_scopes=tuple(required_scopes),
        resource_name=resource_name,
        resource_documentation=resource_documentation,
    )


__all__ = (
    "AccessToken",
    "Requirement",
    "Scheme",
    "TokenVerifier",
    "oauth2_scheme",
    "require",
)
