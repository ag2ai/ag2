# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.acp import ACPConfig, ClaudeCodeConfig, CodexConfig, OpenCodeConfig
from ag2.config.client import LLMClient


def test_claude_defaults() -> None:
    cfg = ClaudeCodeConfig()
    assert cfg.command  # non-empty launch command
    assert cfg.permission_policy == "ask"
    assert cfg.cwd == "."
    assert cfg.allow_terminal is True


def test_codex_defaults() -> None:
    cfg = CodexConfig()
    assert cfg.command == ["codex-acp"]
    assert cfg.permission_policy == "ask"
    assert cfg.cwd == "."
    assert cfg.allow_terminal is True


def test_codex_copy_preserves_subclass() -> None:
    cfg = CodexConfig(cwd="/a")
    cfg2 = cfg.copy(cwd="/b")
    assert cfg.cwd == "/a"
    assert cfg2.cwd == "/b"
    assert isinstance(cfg2, CodexConfig)


def test_codex_create_returns_llmclient() -> None:
    client = CodexConfig().create()
    assert isinstance(client, LLMClient)


def test_opencode_defaults() -> None:
    cfg = OpenCodeConfig()
    assert cfg.command == ["opencode", "acp"]
    assert cfg.permission_policy == "ask"
    assert cfg.cwd == "."
    assert cfg.allow_terminal is True


def test_opencode_copy_preserves_subclass() -> None:
    cfg = OpenCodeConfig(cwd="/a")
    cfg2 = cfg.copy(cwd="/b")
    assert cfg.cwd == "/a"
    assert cfg2.cwd == "/b"
    assert isinstance(cfg2, OpenCodeConfig)


def test_opencode_create_returns_llmclient() -> None:
    client = OpenCodeConfig().create()
    assert isinstance(client, LLMClient)


def test_opencode_model_is_metadata_only() -> None:
    # The `acp` subcommand takes no `--model` flag; model selection lives in
    # OpenCode's own config, so `model` must not alter the launch command.
    cfg = OpenCodeConfig(model="anthropic/claude-sonnet-4")
    assert cfg.command == ["opencode", "acp"]


def test_copy_overrides_without_mutating() -> None:
    cfg = ClaudeCodeConfig(cwd="/a")
    cfg2 = cfg.copy(cwd="/b")
    assert cfg.cwd == "/a"
    assert cfg2.cwd == "/b"
    assert isinstance(cfg2, ClaudeCodeConfig)


def test_create_returns_llmclient() -> None:
    client = ClaudeCodeConfig().create()
    assert isinstance(client, LLMClient)


def test_acp_config_is_usable_directly() -> None:
    cfg = ACPConfig(command=["my-agent", "--acp"], permission_policy="auto")
    assert cfg.command == ["my-agent", "--acp"]
    assert cfg.permission_policy == "auto"
