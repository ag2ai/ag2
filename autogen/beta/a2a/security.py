# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace

from a2a.types import (
    APIKeySecurityScheme,
    HTTPAuthSecurityScheme,
    MutualTlsSecurityScheme,
    OAuth2SecurityScheme,
    OAuthFlows,
    OpenIdConnectSecurityScheme,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)


@dataclass(frozen=True, slots=True)
class Scheme:
    """A named security scheme: binds a card-level identifier to a proto scheme
    and optional OAuth2/OIDC scopes. Use the ``*_scheme(name=...)`` factories
    below to construct one; pass ``Scheme`` objects to ``require(...)`` to
    build a :class:`Requirement`. ``with_scopes(...)`` returns a copy with the
    given scopes attached."""

    name: str
    scheme: SecurityScheme
    scopes: tuple[str, ...] = ()

    def with_scopes(self, *scopes: str) -> "Scheme":
        """Return a copy of this scheme with the given OAuth2/OIDC scopes."""
        return replace(self, scopes=scopes)


@dataclass(frozen=True, slots=True)
class Requirement:
    """A single ``AgentCard.security_requirements`` entry: an AND-set of named
    schemes that must all be presented together. Multiple ``Requirement``s on
    a card are OR-ed (any one suffices). Built via ``require(...)``."""

    schemes: tuple[Scheme, ...]

    def to_proto(self) -> SecurityRequirement:
        """Render this requirement as a raw a2a-sdk ``SecurityRequirement``."""
        return SecurityRequirement(
            schemes={s.name: StringList(list=list(s.scopes)) for s in self.schemes},
        )


def bearer_scheme(*, name: str, bearer_format: str = "JWT", description: str = "") -> Scheme:
    """HTTP Bearer auth declaration (``Authorization: Bearer <token>``)."""
    return Scheme(
        name=name,
        scheme=SecurityScheme(
            http_auth_security_scheme=HTTPAuthSecurityScheme(
                scheme="bearer",
                bearer_format=bearer_format,
                description=description,
            ),
        ),
    )


def http_auth_scheme(*, name: str, scheme: str, bearer_format: str = "", description: str = "") -> Scheme:
    """HTTP authentication declaration (basic, digest, bearer with custom format, ...)."""
    return Scheme(
        name=name,
        scheme=SecurityScheme(
            http_auth_security_scheme=HTTPAuthSecurityScheme(
                scheme=scheme,
                bearer_format=bearer_format,
                description=description,
            ),
        ),
    )


def api_key_scheme(*, name: str, key_name: str, location: str = "header", description: str = "") -> Scheme:
    """API key auth declaration. ``key_name`` is the header/query/cookie key
    sent by the client; ``location`` is ``"header"``, ``"query"``, or ``"cookie"``."""
    return Scheme(
        name=name,
        scheme=SecurityScheme(
            api_key_security_scheme=APIKeySecurityScheme(
                name=key_name,
                location=location,
                description=description,
            ),
        ),
    )


def oauth2_scheme(
    *,
    name: str,
    flows: OAuthFlows,
    oauth2_metadata_url: str = "",
    description: str = "",
) -> Scheme:
    """OAuth2 auth declaration wrapping a pre-built ``OAuthFlows``."""
    return Scheme(
        name=name,
        scheme=SecurityScheme(
            oauth2_security_scheme=OAuth2SecurityScheme(
                flows=flows,
                oauth2_metadata_url=oauth2_metadata_url,
                description=description,
            ),
        ),
    )


def open_id_connect_scheme(*, name: str, url: str, description: str = "") -> Scheme:
    """OpenID Connect discovery URL declaration."""
    return Scheme(
        name=name,
        scheme=SecurityScheme(
            open_id_connect_security_scheme=OpenIdConnectSecurityScheme(
                open_id_connect_url=url,
                description=description,
            ),
        ),
    )


def mtls_scheme(*, name: str, description: str = "") -> Scheme:
    """Mutual TLS declaration (client-cert auth)."""
    return Scheme(
        name=name,
        scheme=SecurityScheme(
            mtls_security_scheme=MutualTlsSecurityScheme(description=description),
        ),
    )


def require(*schemes: Scheme) -> Requirement:
    """Build a :class:`Requirement` from one or more ``Scheme`` objects.

    All schemes in a single ``require()`` call must be presented together
    (AND). Multiple ``Requirement`` entries on a card are OR-ed (any one
    suffices). Attach OAuth2/OIDC scopes via :meth:`Scheme.with_scopes`.

    Example::

        bearer = bearer_scheme(name="bearer")
        oauth = oauth2_scheme(name="oauth", flows=...)

        require(bearer)  # bearer alone
        require(bearer, oauth.with_scopes("read"))  # AND
        require(oauth.with_scopes("read", "write"))  # scoped oauth alone
    """
    return Requirement(schemes=schemes)


__all__ = (
    "Requirement",
    "Scheme",
    "api_key_scheme",
    "bearer_scheme",
    "http_auth_scheme",
    "mtls_scheme",
    "oauth2_scheme",
    "open_id_connect_scheme",
    "require",
)
