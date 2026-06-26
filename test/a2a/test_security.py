# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from a2a.types import (
    APIKeySecurityScheme,
    AuthorizationCodeOAuthFlow,
    ClientCredentialsOAuthFlow,
    HTTPAuthSecurityScheme,
    MutualTlsSecurityScheme,
    OAuth2SecurityScheme,
    OAuthFlows,
    OpenIdConnectSecurityScheme,
    SecurityScheme,
)

from ag2.a2a.security import (
    Requirement,
    Scheme,
    api_key_scheme,
    bearer_scheme,
    http_auth_scheme,
    mtls_scheme,
    oauth2_scheme,
    open_id_connect_scheme,
    require,
)


class TestBearerScheme:
    def test_defaults_to_jwt(self) -> None:
        s = bearer_scheme(name="bearer")

        assert s == Scheme(
            name="bearer",
            scheme=SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(scheme="bearer", bearer_format="JWT"),
            ),
        )

    def test_custom_format_and_description(self) -> None:
        s = bearer_scheme(name="b", bearer_format="opaque", description="internal")

        assert s == Scheme(
            name="b",
            scheme=SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(
                    scheme="bearer",
                    bearer_format="opaque",
                    description="internal",
                ),
            ),
        )


def test_http_auth_scheme_basic() -> None:
    s = http_auth_scheme(name="basic-auth", scheme="basic", description="HTTP basic")

    assert s == Scheme(
        name="basic-auth",
        scheme=SecurityScheme(
            http_auth_security_scheme=HTTPAuthSecurityScheme(scheme="basic", description="HTTP basic"),
        ),
    )


class TestApiKeyScheme:
    def test_header_default(self) -> None:
        s = api_key_scheme(name="api-key", key_name="X-API-Key")

        assert s == Scheme(
            name="api-key",
            scheme=SecurityScheme(
                api_key_security_scheme=APIKeySecurityScheme(name="X-API-Key", location="header"),
            ),
        )

    def test_query_location(self) -> None:
        s = api_key_scheme(name="api-key", key_name="api_key", location="query")

        assert s == Scheme(
            name="api-key",
            scheme=SecurityScheme(
                api_key_security_scheme=APIKeySecurityScheme(name="api_key", location="query"),
            ),
        )


class TestOAuth2Scheme:
    def test_with_client_credentials_flow(self) -> None:
        flows = OAuthFlows(client_credentials=ClientCredentialsOAuthFlow(token_url="https://x/token"))
        s = oauth2_scheme(name="oauth", flows=flows)

        assert s == Scheme(
            name="oauth",
            scheme=SecurityScheme(oauth2_security_scheme=OAuth2SecurityScheme(flows=flows)),
        )

    def test_with_authorization_code_flow_and_metadata_url(self) -> None:
        flows = OAuthFlows(
            authorization_code=AuthorizationCodeOAuthFlow(
                authorization_url="https://x/auth",
                token_url="https://x/token",
            ),
        )
        s = oauth2_scheme(name="oauth", flows=flows, oauth2_metadata_url="https://x/.well-known/openid")

        assert s == Scheme(
            name="oauth",
            scheme=SecurityScheme(
                oauth2_security_scheme=OAuth2SecurityScheme(
                    flows=flows,
                    oauth2_metadata_url="https://x/.well-known/openid",
                ),
            ),
        )


def test_open_id_connect_scheme() -> None:
    s = open_id_connect_scheme(name="oidc", url="https://x/.well-known/openid")

    assert s == Scheme(
        name="oidc",
        scheme=SecurityScheme(
            open_id_connect_security_scheme=OpenIdConnectSecurityScheme(
                open_id_connect_url="https://x/.well-known/openid",
            ),
        ),
    )


def test_mtls_scheme() -> None:
    s = mtls_scheme(name="mtls", description="client cert required")

    assert s == Scheme(
        name="mtls",
        scheme=SecurityScheme(
            mtls_security_scheme=MutualTlsSecurityScheme(description="client cert required"),
        ),
    )


class TestWithScopes:
    def test_attaches_scopes(self) -> None:
        oauth = oauth2_scheme(name="oauth", flows=OAuthFlows())

        scoped = oauth.with_scopes("read", "write")

        assert scoped.scopes == ("read", "write")
        assert scoped.name == "oauth"
        assert scoped.scheme == oauth.scheme

    def test_returns_a_copy(self) -> None:
        oauth = oauth2_scheme(name="oauth", flows=OAuthFlows())

        oauth.with_scopes("read")

        assert oauth.scopes == ()


class TestRequire:
    def test_single_scheme(self) -> None:
        bearer = bearer_scheme(name="bearer")

        req = require(bearer)

        assert req == Requirement(schemes=(bearer,))

    def test_multiple_schemes_and(self) -> None:
        bearer = bearer_scheme(name="bearer")
        api = api_key_scheme(name="api", key_name="X-API-Key")

        req = require(bearer, api)

        assert req == Requirement(schemes=(bearer, api))

    def test_scheme_with_scopes(self) -> None:
        oauth = oauth2_scheme(name="oauth", flows=OAuthFlows())

        req = require(oauth.with_scopes("read", "write"))

        proto = req.to_proto()
        assert list(proto.schemes["oauth"].list) == ["read", "write"]

    def test_mix_scoped_and_unscoped(self) -> None:
        bearer = bearer_scheme(name="bearer")
        oauth = oauth2_scheme(name="oauth", flows=OAuthFlows())

        req = require(bearer, oauth.with_scopes("read"))

        proto = req.to_proto()
        assert set(proto.schemes.keys()) == {"bearer", "oauth"}
        assert list(proto.schemes["bearer"].list) == []
        assert list(proto.schemes["oauth"].list) == ["read"]

    def test_non_identifier_scheme_name(self) -> None:
        s = bearer_scheme(name="X-My-Scheme")

        req = require(s)

        proto = req.to_proto()
        assert list(proto.schemes["X-My-Scheme"].list) == []
