# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

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


def require(*schemes_no_scopes: str, **schemes_with_scopes: Iterable[str]) -> SecurityRequirement:
    """Build a ``SecurityRequirement`` from named schemes.

    Positional ``str`` arguments list schemes that don't need scopes
    (Bearer, API-key, mTLS, etc.). Keyword arguments carry scopes for
    schemes that do (OAuth2 / OIDC). Scheme names that aren't valid
    Python identifiers (e.g. ``"X-My-Scheme"``) can be passed
    positionally.

    All schemes in a single ``require()`` call must be satisfied
    together (AND); multiple ``SecurityRequirement`` entries on the card
    are OR-ed (any one suffices).

    Example::

        require("bearer")  # bearer alone, no scopes
        require("bearer", "x_api_key")  # AND of bearer + x_api_key
        require(oauth2=["read", "write"])  # oauth2 with two scopes
        require("bearer", oauth2=["read"])  # AND: bearer (no scopes) + oauth2 (with scopes)
        require("X-My-Scheme")  # non-identifier name
    """
    combined: dict[str, list[str]] = {name: [] for name in schemes_no_scopes}
    combined.update({name: list(scopes) for name, scopes in schemes_with_scopes.items()})
    return SecurityRequirement(
        schemes={name: StringList(list=v) for name, v in combined.items()},
    )


__all__ = (
    "api_key_scheme",
    "bearer_scheme",
    "http_auth_scheme",
    "mtls_scheme",
    "oauth2_scheme",
    "open_id_connect_scheme",
    "require",
)
