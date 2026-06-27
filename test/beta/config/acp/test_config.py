# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.config.acp import ACPConfig, ClaudeCodeConfig
from autogen.beta.config.client import LLMClient


def test_claude_defaults() -> None:
    cfg = ClaudeCodeConfig()
    assert cfg.command  # non-empty launch command
    assert cfg.permission_policy == "ask"
    assert cfg.cwd == "."
    assert cfg.allow_terminal is True


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
