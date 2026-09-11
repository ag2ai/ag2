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
    """An OAuth 2.0 authorization server that may issue tokens for this MCP server.

    Build one with :func:`oauth2_scheme` and pass it to :func:`require`."""

    url: str


@dataclass(frozen=True, slots=True)
class Requirement:
    """The OAuth 2.0 Resource Server security requirement for an MCP server.

    Carries the bring-your-own ``verifier`` and the ``required_scopes`` enforced
    on the MCP endpoint; :meth:`to_metadata` renders the RFC 9728
    ``ProtectedResourceMetadata`` served at
    ``/.well-known/oauth-protected-resource``. Issuing tokens stays with the
    external authorization server. Build via :func:`require`."""

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
    """Declare an authorization server by the issuer ``url`` that mints its tokens.

    ``url`` must be an absolute ``http(s)`` URL: an OIDC issuer *string* like
    ``stytch.com/project-...`` is not usable here — pass the full URL whose
    ``/.well-known/...`` metadata resolves."""
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
