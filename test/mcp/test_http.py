# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from mcp.server.streamable_http_manager import DEFAULT_MAX_SESSIONS, DEFAULT_SESSION_IDLE_TIMEOUT
from mcp.server.transport_security import DEFAULT_MAX_REQUEST_BODY_SIZE, TransportSecuritySettings

from ag2.mcp import MCPServer, TransportConfig
from ag2.mcp.testing import serve

from ._helpers import JSON_HEADERS, greeter, initialize_request

# Pinned rather than tracking the newest handshake revision: ``mcp`` 2.0 serves
# both eras, and which one a connection lands in is settled by this request.
_INIT = initialize_request(version="2025-06-18")


@pytest.mark.asyncio
class TestHttpTransport:
    async def test_serves_initialize_as_asgi_app(self) -> None:
        app = MCPServer(greeter(), json_response=True)

        async with serve(app) as client:
            resp = await client.post("/mcp", headers=JSON_HEADERS, json=_INIT)

        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result["serverInfo"]["name"] == "greeter"
        # A silent era switch — which changes capability semantics — cannot pass.
        assert result["protocolVersion"] == "2025-06-18"

    async def test_custom_path(self) -> None:
        app = MCPServer(greeter(), path="/agent", json_response=True)

        async with serve(app) as client:
            on_custom = await client.post("/agent", headers=JSON_HEADERS, json=_INIT)
            on_default = await client.post("/mcp", headers=JSON_HEADERS, json=_INIT)

        assert on_custom.status_code == 200
        assert on_default.status_code == 404


@pytest.mark.asyncio
class TestTransportBounds:
    """The transport settings an operator configures, observed through the transport."""

    async def test_a_body_over_the_limit_is_refused_before_a_session_exists(self) -> None:
        app = MCPServer(greeter(), json_response=True, transport=TransportConfig(max_request_body_size=1024))

        async with serve(app) as client:
            resp = await client.post("/mcp", headers=JSON_HEADERS, content=b'{"padding": "' + b"x" * 4096 + b'"}')

        assert resp.status_code == 413
        # The refusal is the *limit*, not a handler that read the body and gave
        # up: no session was opened for the request that carried it.
        assert "mcp-session-id" not in resp.headers

    async def test_a_default_server_refuses_a_body_over_four_mebibytes(self) -> None:
        """The documented default, exercised rather than read back off the server."""
        app = MCPServer(greeter(), json_response=True)

        async with serve(app) as client:
            oversized = await client.post(
                "/mcp", headers=JSON_HEADERS, content=b'{"padding": "' + b"x" * (4 * 1024 * 1024) + b'"}'
            )
            ordinary = await client.post("/mcp", headers=JSON_HEADERS, json=_INIT)

        assert oversized.status_code == 413
        assert ordinary.status_code == 200

    async def test_the_session_beyond_the_cap_is_refused(self) -> None:
        app = MCPServer(greeter(), json_response=True, transport=TransportConfig(max_mcp_sessions=2))

        async with serve(app) as client:
            opened = [await client.post("/mcp", headers=JSON_HEADERS, json=_INIT) for _ in range(2)]
            beyond = await client.post("/mcp", headers=JSON_HEADERS, json=_INIT)

        assert [r.status_code for r in opened] == [200, 200]
        assert len({r.headers["mcp-session-id"] for r in opened}) == 2, "the cap admitted one session twice"
        assert beyond.status_code == 503

    async def test_an_idle_mcp_session_is_reaped_and_must_initialize_again(self) -> None:
        app = MCPServer(greeter(), json_response=True, transport=TransportConfig(mcp_session_idle_timeout=0.2))

        async with serve(app) as client:
            opened = await client.post("/mcp", headers=JSON_HEADERS, json=_INIT)
            session_id = opened.headers["mcp-session-id"]
            await asyncio.sleep(1.0)
            after_idle = await client.post(
                "/mcp",
                headers={**JSON_HEADERS, "mcp-session-id": session_id},
                json=initialize_request(request_id=2, version="2025-06-18"),
            )
            reopened = await client.post("/mcp", headers=JSON_HEADERS, json=_INIT)

        assert opened.status_code == 200
        assert after_idle.status_code == 404
        assert reopened.status_code == 200
        assert reopened.headers["mcp-session-id"] != session_id

    async def test_a_disallowed_host_is_rejected_and_an_allowed_one_serves(self) -> None:
        app = MCPServer(
            greeter(),
            json_response=True,
            transport=TransportConfig(
                security_settings=TransportSecuritySettings(
                    allowed_hosts=["agent.example.com"],
                    allowed_origins=["https://agent.example.com"],
                )
            ),
        )

        async with serve(app, base_url="http://agent.example.com") as client:
            allowed = await client.post("/mcp", headers=JSON_HEADERS, json=_INIT)
            rebound = await client.post("/mcp", headers={**JSON_HEADERS, "Host": "evil.example.com"}, json=_INIT)

        assert allowed.status_code == 200
        assert rebound.status_code == 421

    async def test_default_constructed_security_settings_refuse_every_request(self) -> None:
        """The documented trap: enabling the protection with empty allow-lists."""
        app = MCPServer(
            greeter(), json_response=True, transport=TransportConfig(security_settings=TransportSecuritySettings())
        )

        async with serve(app) as client:
            resp = await client.post("/mcp", headers=JSON_HEADERS, json=_INIT)

        assert resp.status_code == 421

    async def test_omitting_the_transport_config_matches_a_default_constructed_one(self) -> None:
        """Compared where the default bites, not on a request any config would serve.

        Two small requests agree under almost any pair of configs; the boundary
        the omitted default sets is what has to agree.
        """
        omitted = MCPServer(greeter(), json_response=True)
        explicit = MCPServer(greeter(), json_response=True, transport=TransportConfig())
        under = {**_INIT, "padding": "x" * (4 * 1024 * 1024 - 8192)}
        over = {**_INIT, "padding": "x" * (4 * 1024 * 1024 + 8192)}

        async with serve(omitted) as client:
            without_under = await client.post("/mcp", headers=JSON_HEADERS, json=under)
            without_over = await client.post("/mcp", headers=JSON_HEADERS, json=over)
        async with serve(explicit) as client:
            with_default_under = await client.post("/mcp", headers=JSON_HEADERS, json=under)
            with_default_over = await client.post("/mcp", headers=JSON_HEADERS, json=over)

        assert without_under.status_code == with_default_under.status_code == 200
        assert without_over.status_code == with_default_over.status_code == 413


class TestTransportDefaults:
    """AG2's own numbers, and the upstream ones they were aligned with."""

    def test_the_documented_defaults_are_the_numbers_ag2_promises(self) -> None:
        config = TransportConfig()

        assert config.mcp_session_idle_timeout == 1800.0
        assert config.max_mcp_sessions == 10_000
        assert config.max_request_body_size == 4 * 1024 * 1024
        assert config.security_settings is None
        assert config.sse_retry_interval is None
        assert config.event_store is None

    def test_the_defaults_still_match_the_sdks(self) -> None:
        """Red when upstream retunes a default, rather than a support ticket later.

        AG2 passes its own values through explicitly, so a retune upstream cannot
        change what a deployment does — it can only make this promise stale.
        """
        config = TransportConfig()

        assert config.mcp_session_idle_timeout == DEFAULT_SESSION_IDLE_TIMEOUT
        assert config.max_mcp_sessions == DEFAULT_MAX_SESSIONS
        assert config.max_request_body_size == DEFAULT_MAX_REQUEST_BODY_SIZE


class TestTransportValidation:
    """Bounds refused at construction, which is the only seam before a request exists."""

    def test_stateless_with_a_configured_idle_timeout_names_both_settings(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            MCPServer(greeter(), stateless=True, transport=TransportConfig(mcp_session_idle_timeout=60.0))

        message = str(excinfo.value)
        assert "stateless" in message
        assert "mcp_session_idle_timeout" in message

    def test_spelling_out_the_default_idle_timeout_is_not_asking_for_reaping(self) -> None:
        """Value, not provenance: the default number asks for nothing, however it arrives.

        Telling it apart from an inherited default would need the value to carry
        where it came from, and that is not worth a sentinel — a caller who types
        the default has expressed what the default already says.
        """
        app = MCPServer(
            greeter(),
            stateless=True,
            transport=TransportConfig(mcp_session_idle_timeout=DEFAULT_SESSION_IDLE_TIMEOUT),
        )

        assert app.agent.name == "greeter"

    def test_stateless_may_ask_for_no_reaping_at_all(self) -> None:
        """``None`` asks the transport *not* to reap, which stateless cannot contradict."""
        app = MCPServer(greeter(), stateless=True, transport=TransportConfig(mcp_session_idle_timeout=None))

        assert app.agent.name == "greeter"

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({"mcp_session_idle_timeout": 0}, id="idle-timeout"),
            pytest.param({"max_mcp_sessions": 0}, id="max-mcp-sessions"),
            pytest.param({"max_request_body_size": 0}, id="request-body-size"),
            pytest.param({"sse_retry_interval": 0}, id="sse-retry-interval"),
        ],
    )
    def test_a_non_positive_bound_is_refused(self, config: dict[str, int]) -> None:
        with pytest.raises(ValueError):
            TransportConfig(**config)

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({"mcp_session_idle_timeout": None}, id="never-reap"),
            pytest.param({"max_mcp_sessions": None}, id="no-cap"),
        ],
    )
    def test_the_off_value_is_not_a_non_positive_bound(self, config: dict[str, None]) -> None:
        MCPServer(greeter(), transport=TransportConfig(**config))
