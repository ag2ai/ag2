# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.acp import ACPConfig, ClaudeCodeConfig, CodexConfig, KiloCodeConfig, OpenCodeConfig
from ag2.config.client import LLMClient


class TestClaudeCodeConfig:
    def test_defaults(self) -> None:
        cfg = ClaudeCodeConfig()
        assert cfg.command  # non-empty launch command
        assert cfg.permission_policy == "ask"
        assert cfg.cwd == "."
        assert cfg.allow_terminal is True

    def test_copy_preserves_subclass(self) -> None:
        cfg = ClaudeCodeConfig(cwd="/a")
        cfg2 = cfg.copy(cwd="/b")
        assert cfg.cwd == "/a"
        assert cfg2.cwd == "/b"
        assert isinstance(cfg2, ClaudeCodeConfig)

    def test_create_returns_llmclient(self) -> None:
        client = ClaudeCodeConfig().create()
        assert isinstance(client, LLMClient)


class TestCodexConfig:
    def test_defaults(self) -> None:
        cfg = CodexConfig()
        assert cfg.command == ["codex-acp"]
        assert cfg.permission_policy == "ask"
        assert cfg.cwd == "."
        assert cfg.allow_terminal is True

    def test_copy_preserves_subclass(self) -> None:
        cfg = CodexConfig(cwd="/a")
        cfg2 = cfg.copy(cwd="/b")
        assert cfg.cwd == "/a"
        assert cfg2.cwd == "/b"
        assert isinstance(cfg2, CodexConfig)

    def test_create_returns_llmclient(self) -> None:
        client = CodexConfig().create()
        assert isinstance(client, LLMClient)


class TestOpenCodeConfig:
    def test_defaults(self) -> None:
        cfg = OpenCodeConfig()
        assert cfg.command == ["opencode", "acp"]
        assert cfg.permission_policy == "ask"
        assert cfg.cwd == "."
        assert cfg.allow_terminal is True

    def test_copy_preserves_subclass(self) -> None:
        cfg = OpenCodeConfig(cwd="/a")
        cfg2 = cfg.copy(cwd="/b")
        assert cfg.cwd == "/a"
        assert cfg2.cwd == "/b"
        assert isinstance(cfg2, OpenCodeConfig)

    def test_create_returns_llmclient(self) -> None:
        client = OpenCodeConfig().create()
        assert isinstance(client, LLMClient)

    def test_model_does_not_alter_command(self) -> None:
        # `model` is applied via ACP session/set_config_option after the handshake,
        # never via CLI flags — the launch command must stay untouched.
        cfg = OpenCodeConfig(model="anthropic/claude-sonnet-5")
        assert cfg.command == ["opencode", "acp"]


class TestKiloCodeConfig:
    def test_defaults(self) -> None:
        cfg = KiloCodeConfig()
        assert cfg.command == ["kilo", "acp"]
        assert cfg.permission_policy == "ask"
        assert cfg.cwd == "."
        assert cfg.allow_terminal is True

    def test_copy_preserves_subclass(self) -> None:
        cfg = KiloCodeConfig(cwd="/a")
        cfg2 = cfg.copy(cwd="/b")
        assert cfg.cwd == "/a"
        assert cfg2.cwd == "/b"
        assert isinstance(cfg2, KiloCodeConfig)

    def test_create_returns_llmclient(self) -> None:
        client = KiloCodeConfig().create()
        assert isinstance(client, LLMClient)

    def test_model_does_not_alter_command(self) -> None:
        # `model` is applied via ACP session/set_config_option after the handshake,
        # never via CLI flags — the launch command must stay untouched.
        cfg = KiloCodeConfig(model="kilo/anthropic/claude-haiku-4.5")
        assert cfg.command == ["kilo", "acp"]


def test_acp_config_is_usable_directly() -> None:
    cfg = ACPConfig(command=["my-agent", "--acp"], permission_policy="auto")
    assert cfg.command == ["my-agent", "--acp"]
    assert cfg.permission_policy == "auto"


def test_the_published_parameter_order_still_holds() -> None:
    # The parameter order is API: a caller who passed these positionally wrote
    # against it, so growing the config — a shared base for the remote one, a new
    # policy — must leave every existing position meaning what it meant. A new
    # field therefore goes after ``expose_tools``, never between.
    cfg = ACPConfig(["my-agent", "--acp"], "/work", {"KEY": "v"}, "sonnet", "auto", "/root", False, ["/extra"])

    assert cfg == ACPConfig(
        command=["my-agent", "--acp"],
        cwd="/work",
        env={"KEY": "v"},
        model="sonnet",
        permission_policy="auto",
        fs_root="/root",
        allow_terminal=False,
        additional_directories=["/extra"],
    )


def test_presets_take_a_command_positionally_too() -> None:
    # A preset is an ACPConfig whose command has a default, and overriding that
    # default positionally is how the docs' `npx -y ...` variants are written.
    assert ClaudeCodeConfig(["npx", "-y", "@agentclientprotocol/claude-agent-acp"]).command[0] == "npx"
    assert CodexConfig(["npx", "-y", "@agentclientprotocol/codex-acp"], "/work").cwd == "/work"
    assert OpenCodeConfig(["opencode", "acp"]).cwd == "."
    assert KiloCodeConfig(["npx", "-y", "@kilocode/cli", "acp"]).command[-1] == "acp"


def test_elicitation_policy_defaults_to_ask_and_survives_copy() -> None:
    # Same default as the permission policy: asking is what a caller gets for free.
    assert ACPConfig().elicitation_policy == "ask"
    assert ClaudeCodeConfig().elicitation_policy == "ask"

    unattended = ACPConfig().copy(elicitation_policy="decline")
    assert unattended.elicitation_policy == "decline"
    assert ACPConfig().elicitation_policy == "ask"  # original untouched


def test_expose_tools_defaults_on_and_survives_copy() -> None:
    cfg = ACPConfig()
    assert cfg.expose_tools is True

    off = cfg.copy(expose_tools=False)
    assert off.expose_tools is False
    assert cfg.expose_tools is True  # original untouched
