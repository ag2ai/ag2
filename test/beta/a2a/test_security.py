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

from autogen.beta.a2a.security import (
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
        assert bearer_scheme() == SecurityScheme(
            http_auth_security_scheme=HTTPAuthSecurityScheme(scheme="bearer", bearer_format="JWT"),
        )

    def test_custom_format_and_description(self) -> None:
        assert bearer_scheme(bearer_format="opaque", description="internal") == SecurityScheme(
            http_auth_security_scheme=HTTPAuthSecurityScheme(
                scheme="bearer",
                bearer_format="opaque",
                description="internal",
            ),
        )


def test_http_auth_scheme_basic() -> None:
    assert http_auth_scheme(scheme="basic", description="HTTP basic") == SecurityScheme(
        http_auth_security_scheme=HTTPAuthSecurityScheme(scheme="basic", description="HTTP basic"),
    )


class TestApiKeyScheme:
    def test_header_default(self) -> None:
        assert api_key_scheme(name="X-API-Key") == SecurityScheme(
            api_key_security_scheme=APIKeySecurityScheme(name="X-API-Key", location="header"),
        )

    def test_query_location(self) -> None:
        assert api_key_scheme(name="api_key", location="query") == SecurityScheme(
            api_key_security_scheme=APIKeySecurityScheme(name="api_key", location="query"),
        )


class TestOAuth2Scheme:
    def test_with_client_credentials_flow(self) -> None:
        flows = OAuthFlows(client_credentials=ClientCredentialsOAuthFlow(token_url="https://x/token"))

        assert oauth2_scheme(flows=flows) == SecurityScheme(
            oauth2_security_scheme=OAuth2SecurityScheme(flows=flows),
        )

    def test_with_authorization_code_flow_and_metadata_url(self) -> None:
        flows = OAuthFlows(
            authorization_code=AuthorizationCodeOAuthFlow(
                authorization_url="https://x/auth",
                token_url="https://x/token",
            ),
        )

        assert oauth2_scheme(flows=flows, oauth2_metadata_url="https://x/.well-known/openid") == SecurityScheme(
            oauth2_security_scheme=OAuth2SecurityScheme(
                flows=flows,
                oauth2_metadata_url="https://x/.well-known/openid",
            ),
        )


def test_open_id_connect_scheme() -> None:
    assert open_id_connect_scheme(url="https://x/.well-known/openid") == SecurityScheme(
        open_id_connect_security_scheme=OpenIdConnectSecurityScheme(
            open_id_connect_url="https://x/.well-known/openid",
        ),
    )


def test_mtls_scheme() -> None:
    assert mtls_scheme(description="client cert required") == SecurityScheme(
        mtls_security_scheme=MutualTlsSecurityScheme(description="client cert required"),
    )


class TestRequire:
    def test_single_scheme_positional(self) -> None:
        req = require("bearer")

        assert list(req.schemes.keys()) == ["bearer"]
        assert list(req.schemes["bearer"].list) == []

    def test_multiple_schemes_positional(self) -> None:
        req = require("bearer", "api_key")

        assert set(req.schemes.keys()) == {"bearer", "api_key"}
        assert list(req.schemes["bearer"].list) == []
        assert list(req.schemes["api_key"].list) == []

    def test_oauth_with_scopes(self) -> None:
        req = require(oauth=["read", "write"])

        assert list(req.schemes["oauth"].list) == ["read", "write"]

    def test_mix_positional_and_kwargs(self) -> None:
        req = require("bearer", oauth=["read"])

        assert set(req.schemes.keys()) == {"bearer", "oauth"}
        assert list(req.schemes["bearer"].list) == []
        assert list(req.schemes["oauth"].list) == ["read"]

    def test_non_identifier_scheme_name(self) -> None:
        req = require("X-My-Scheme")

        assert list(req.schemes["X-My-Scheme"].list) == []

    def test_kwargs_only_remains_supported(self) -> None:
        req = require(bearer=[], api_key=[])

        assert set(req.schemes.keys()) == {"bearer", "api_key"}
