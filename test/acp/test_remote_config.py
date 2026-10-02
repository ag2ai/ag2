# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""The remote config's own surface: what it accepts, and what it refuses.

Only what a config answers on its own belongs here. Which transport a URL
resolves to is asserted where it is observable — ``ACPTransportError`` names it
in ``test_remote_e2e`` — and what a connection actually carries is asserted
against a real server in ``test_remote_transport``.
"""

import pytest

pytest.importorskip("websockets")
pytest.importorskip("h2")

import subprocess
import sys
import textwrap

from ag2.acp import ACPConfig, ACPRemoteConfig, ClaudeCodeConfig
from ag2.acp.tool_gateway import GatewayAddress
from ag2.config.client import LLMClient


class TestTransportResolution:
    """A URL AG2 cannot map is refused at construction, not at the first turn."""

    def test_unmappable_scheme_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot infer the ACP transport"):
            ACPRemoteConfig(url="ftp://box.internal/acp")

    def test_unmappable_scheme_is_fine_with_an_override(self) -> None:
        # A gateway whose scheme says nothing about the transport is still usable.
        assert ACPRemoteConfig(url="acp+tls://box.internal/acp", transport="websocket").transport == "websocket"

    def test_unknown_override_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown ACP transport"):
            ACPRemoteConfig(url="https://box.internal/acp", transport="carrier-pigeon")  # type: ignore[arg-type]


class TestConstruction:
    def test_url_is_required(self) -> None:
        with pytest.raises(TypeError, match="url"):
            ACPRemoteConfig()  # type: ignore[call-arg]

    def test_a_command_and_a_url_together_are_rejected(self) -> None:
        # Not a precedence rule: a remote config carries no launch fields at all,
        # so an ambiguous config cannot be constructed in the first place.
        with pytest.raises(TypeError, match="command"):
            ACPRemoteConfig(url="https://box.internal/acp", command=["claude-agent-acp"])  # type: ignore[call-arg]

    def test_create_returns_llmclient(self) -> None:
        assert isinstance(ACPRemoteConfig(url="https://box.internal/acp").create(), LLMClient)

    def test_driving_fields_carry_over_from_the_base(self) -> None:
        cfg = ACPRemoteConfig(
            url="https://box.internal/acp",
            cwd="/repo",
            model="claude-sonnet-5",
            permission_policy="auto",
            elicitation_policy="decline",
            turn_timeout=30.0,
            expose_tools=False,
        )
        assert (cfg.cwd, cfg.model, cfg.permission_policy) == ("/repo", "claude-sonnet-5", "auto")
        assert (cfg.elicitation_policy, cfg.turn_timeout, cfg.expose_tools) == ("decline", 30.0, False)

    def test_copy_preserves_type_and_url(self) -> None:
        cfg = ACPRemoteConfig(url="https://box.internal/acp", cwd="/a")
        copied = cfg.copy(cwd="/b")
        assert isinstance(copied, ACPRemoteConfig)
        assert (copied.url, copied.cwd, cfg.cwd) == ("https://box.internal/acp", "/b", "/a")

    def test_identity_ignores_run_scoped_state(self) -> None:
        one = ACPRemoteConfig(url="https://box.internal/acp")
        other = ACPRemoteConfig(url="https://box.internal/acp")
        assert one == other
        assert one != ACPRemoteConfig(url="wss://box.internal/acp")

    def test_a_configured_token_stays_out_of_the_repr(self) -> None:
        cfg = ACPRemoteConfig(url="https://box.internal/acp", headers={"Authorization": "Bearer s3cret"})
        assert "s3cret" not in repr(cfg)
        assert cfg.headers == {"Authorization": "Bearer s3cret"}  # still carried, just not printed


class TestGatewayAddress:
    """Parsing and validation only — where the gateway actually binds, and what a
    remote agent is handed, is asserted through a driven turn in ``test_remote_e2e``.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("10.0.0.5:9000", GatewayAddress(host="10.0.0.5", port=9000)),
            ("ag2.internal", GatewayAddress(host="ag2.internal", port=0)),
            ("[::1]:9000", GatewayAddress(host="::1", port=9000)),
            ("  10.0.0.5:9000  ", GatewayAddress(host="10.0.0.5", port=9000)),
        ],
    )
    def test_parse(self, text: str, expected: GatewayAddress) -> None:
        assert GatewayAddress.parse(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "host:port",  # non-numeric port
            "10.0.0.5:70000",  # out of range
            "[::1",  # unbalanced
            "http://10.0.0.5:8931",  # a URL, not an address
            "user@10.0.0.5",
            "10.0.0.5:1:2",  # neither host:port nor an IPv6 literal
        ],
    )
    def test_parse_rejects_nonsense(self, text: str) -> None:
        with pytest.raises(ValueError):
            GatewayAddress.parse(text)

    def test_parse_accepts_a_bare_ipv6_literal(self) -> None:
        assert GatewayAddress.parse("::1") == GatewayAddress(host="::1", port=0)

    def test_a_bad_address_is_rejected_at_construction(self) -> None:
        # Not at the first tool-bearing turn, which is a long way from the typo.
        with pytest.raises(ValueError, match="not a URL"):
            ACPRemoteConfig(url="https://box.internal/acp", gateway_address="http://10.0.0.5:8931")

    def test_ipv6_host_is_bracketed_in_a_url(self) -> None:
        assert GatewayAddress(host="::1").authority == "[::1]"
        assert GatewayAddress(host="10.0.0.5").authority == "10.0.0.5"

    def test_an_address_is_never_inferred_from_the_url(self) -> None:
        # The URL names where the *agent* is; it says nothing about whether the
        # agent can dial back, so it must not be read as consent to open a port.
        assert ACPRemoteConfig(url="https://box.internal/acp").gateway_address is None


class TestLaunchFields:
    """A remote config drives an agent the same way, and launches nothing."""

    @pytest.mark.parametrize("launch_field", ["command", "env"])
    def test_a_launch_field_is_refused(self, launch_field: str) -> None:
        # Refused rather than ignored: a caller who passes a command is telling
        # AG2 to launch something, and this config cannot.
        with pytest.raises(TypeError, match=launch_field):
            ACPRemoteConfig(url="https://box.internal/acp", **{launch_field: ["agent"]})

    def test_a_launch_field_is_not_in_the_repr(self) -> None:
        assert "command" not in repr(ACPRemoteConfig(url="https://box.internal/acp"))

    def test_the_driving_fields_are_shared_with_the_launch_config(self) -> None:
        remote = ACPRemoteConfig(url="https://box.internal/acp", cwd="/w", permission_policy="auto")
        assert (remote.cwd, remote.permission_policy) == (ACPConfig(cwd="/w", permission_policy="auto").cwd, "auto")
        assert (remote.expose_tools, remote.elicitation_policy) == (True, "ask")

    def test_every_config_is_an_acp_config(self) -> None:
        assert isinstance(ACPRemoteConfig(url="https://box.internal/acp"), ACPConfig)
        assert isinstance(ClaudeCodeConfig(), ACPConfig)


def test_import_without_the_extra_names_what_to_install() -> None:
    """A caller who forgot ``agent-client-protocol[http]`` gets told, not a bare ImportError."""
    script = textwrap.dedent("""
        import sys

        class Blocker:
            \"\"\"Stand in for an install without the remote transports' own deps.\"\"\"

            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] == "websockets":
                    raise ImportError(f"No module named {name!r}")
                return None

        sys.meta_path.insert(0, Blocker())

        import ag2.acp

        assert ag2.acp.ACPConfig(command=["agent"]).command == ["agent"]  # the rest still works

        try:
            ag2.acp.ACPRemoteConfig(url="https://box.internal/acp")
        except ImportError as e:
            print(e)
        else:
            print("NO ERROR RAISED")
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert 'pip install "agent-client-protocol[http]"' in result.stdout
