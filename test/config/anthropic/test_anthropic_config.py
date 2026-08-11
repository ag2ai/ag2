# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.config import AnthropicConfig
from ag2.config.anthropic import AnthropicClient


def test_copy_without_overrides_returns_new_equal_instance() -> None:
    config = AnthropicConfig(model="claude-sonnet-4-6", temperature=0.2, streaming=True)

    copied = config.copy()

    assert copied == config
    assert copied is not config


def test_copy_applies_overrides_without_mutating_original() -> None:
    config = AnthropicConfig(model="claude-sonnet-4-6", api_key="key", temperature=0.2, streaming=False)

    copied = config.copy(model="claude-haiku-4-5-20251001", temperature=0.8, streaming=True, api_key=None)

    assert copied.model == "claude-haiku-4-5-20251001"
    assert copied.temperature == 0.8
    assert copied.streaming is True
    assert copied.api_key is None

    assert config.model == "claude-sonnet-4-6"
    assert config.temperature == 0.2
    assert config.streaming is False
    assert config.api_key == "key"  # pragma: allowlist secret


def test_create_returns_anthropic_client() -> None:
    config = AnthropicConfig(model="claude-sonnet-4-6", api_key="test-key")
    client = config.create()

    assert isinstance(client, AnthropicClient)


def test_max_tokens_defaults_to_4096() -> None:
    config = AnthropicConfig(model="claude-sonnet-4-6")
    assert config.max_tokens == 4096


def test_max_tokens_can_be_overridden() -> None:
    config = AnthropicConfig(model="claude-sonnet-4-6", max_tokens=8192)
    assert config.max_tokens == 8192


def test_extra_body_defaults_to_none() -> None:
    config = AnthropicConfig(model="claude-sonnet-4-6")
    assert config.extra_body is None


def test_extra_body_can_be_set() -> None:
    extra = {"tool_choice": {"type": "auto", "disable_parallel_tool_use": True}}
    config = AnthropicConfig(model="claude-sonnet-4-6", extra_body=extra)

    assert config.extra_body == extra


def test_extra_body_passed_to_client() -> None:
    extra = {"tool_choice": {"type": "auto", "disable_parallel_tool_use": True}}
    config = AnthropicConfig(model="claude-sonnet-4-6", api_key="test-key", extra_body=extra)

    client = config.create()

    assert client._extra_body == extra


def test_copy_with_extra_body() -> None:
    base = AnthropicConfig(model="claude-sonnet-4-6")
    extra = {"thinking": {"type": "enabled", "budget_tokens": 5000}}

    copied = base.copy(extra_body=extra)

    assert copied.extra_body == extra
    assert base.extra_body is None
