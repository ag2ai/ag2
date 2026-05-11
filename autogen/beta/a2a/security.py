# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Mapping

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


def bearer_scheme(*, bearer_format: str = "JWT", description: str = "") -> SecurityScheme:
    """HTTP Bearer auth declaration (``Authorization: Bearer <token>``)."""
    return SecurityScheme(
        http_auth_security_scheme=HTTPAuthSecurityScheme(
            scheme="bearer",
            bearer_format=bearer_format,
            description=description,
        ),
    )


def http_auth_scheme(*, scheme: str, bearer_format: str = "", description: str = "") -> SecurityScheme:
    """HTTP authentication declaration (basic, digest, bearer with custom format, ...)."""
    return SecurityScheme(
        http_auth_security_scheme=HTTPAuthSecurityScheme(
            scheme=scheme,
            bearer_format=bearer_format,
            description=description,
        ),
    )


def api_key_scheme(*, name: str, location: str = "header", description: str = "") -> SecurityScheme:
    """API key auth declaration. ``location`` is ``"header"``, ``"query"``, or ``"cookie"``."""
    return SecurityScheme(
        api_key_security_scheme=APIKeySecurityScheme(
            name=name,
            location=location,
            description=description,
        ),
    )


def oauth2_scheme(
    *,
    flows: OAuthFlows,
    oauth2_metadata_url: str = "",
    description: str = "",
) -> SecurityScheme:
    """OAuth2 auth declaration wrapping a pre-built ``OAuthFlows``."""
    return SecurityScheme(
        oauth2_security_scheme=OAuth2SecurityScheme(
            flows=flows,
            oauth2_metadata_url=oauth2_metadata_url,
            description=description,
        ),
    )


def open_id_connect_scheme(*, url: str, description: str = "") -> SecurityScheme:
    """OpenID Connect discovery URL declaration."""
    return SecurityScheme(
        open_id_connect_security_scheme=OpenIdConnectSecurityScheme(
            open_id_connect_url=url,
            description=description,
        ),
    )


def mtls_scheme(*, description: str = "") -> SecurityScheme:
    """Mutual TLS declaration (client-cert auth)."""
    return SecurityScheme(
        mtls_security_scheme=MutualTlsSecurityScheme(description=description),
    )


def require(**schemes: Iterable[str]) -> SecurityRequirement:
    """Build a ``SecurityRequirement`` mapping scheme names to required scopes.

    Each kwarg's key is a scheme name registered in ``security_schemes``;
    the value is an iterable of scopes (empty for schemes without scopes,
    e.g. bearer / api-key). All schemes in a single requirement must be
    satisfied together; multiple ``SecurityRequirement`` entries on the
    card are OR-ed (any one suffices).

    Example::

        require(bearer=[])  # bearer scheme, no scopes
        require(oauth2=["read", "write"])  # oauth2 with two scopes
        require(bearer=[], api_key=[])  # AND of bearer + api_key
    """
    return SecurityRequirement(
        schemes={name: StringList(list=list(scopes)) for name, scopes in schemes.items()},
    )


def require_scopes(name: str, scopes: Mapping[str, Iterable[str]] | Iterable[str] | None = None) -> SecurityRequirement:
    """Build a single-scheme ``SecurityRequirement`` with the given scopes.

    Convenience for cases where the scheme name isn't a valid Python
    identifier (e.g. ``"My-Scheme"``) and can't be used as a kwarg.
    """
    return SecurityRequirement(schemes={name: StringList(list=list(scopes or []))})


__all__ = (
    "api_key_scheme",
    "bearer_scheme",
    "http_auth_scheme",
    "mtls_scheme",
    "oauth2_scheme",
    "open_id_connect_scheme",
    "require",
    "require_scopes",
)
