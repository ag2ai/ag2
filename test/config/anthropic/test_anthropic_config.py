# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any, Literal

import pytest

from ag2.config import AnthropicConfig
from ag2.config.anthropic import AnthropicClient, _compat


def test_copy_without_overrides_returns_new_equal_instance() -> None:
    config = AnthropicConfig(model="claude-sonnet-5", max_tokens=2048, streaming=True)

    copied = config.copy()

    assert copied == config
    assert copied is not config


def test_copy_applies_overrides_without_mutating_original() -> None:
    config = AnthropicConfig(model="claude-sonnet-5", api_key="key", max_tokens=2048, streaming=False)

    copied = config.copy(model="claude-haiku-4-5-20251001", max_tokens=4096, streaming=True, api_key=None)

    assert copied.model == "claude-haiku-4-5-20251001"
    assert copied.max_tokens == 4096
    assert copied.streaming is True
    assert copied.api_key is None

    assert config.model == "claude-sonnet-5"
    assert config.max_tokens == 2048
    assert config.streaming is False
    assert config.api_key == "key"  # pragma: allowlist secret


def test_create_returns_anthropic_client() -> None:
    config = AnthropicConfig(model="claude-sonnet-5", api_key="test-key")
    client = config.create()

    assert isinstance(client, AnthropicClient)


def test_max_tokens_defaults_to_4096() -> None:
    config = AnthropicConfig(model="claude-sonnet-5")
    assert config.max_tokens == 4096


@pytest.mark.parametrize("sampling", [{"temperature": 0.2}, {"top_p": 0.9}, {"top_k": 5}])
def test_sampling_parameters_are_deprecated_and_sent_only_by_sdk_1x(sampling: dict[str, Any]) -> None:
    # `anthropic` 2.x sends an API version that answers 400 to temperature, top_p and top_k.
    # The `test-anthropic-1x` CI job runs this against the 1.x SDK.
    sent = not _compat.SDK_2X
    config = AnthropicConfig(model="claude-sonnet-5", **sampling)

    with pytest.warns(DeprecationWarning, match="removed in 1.4.0"):
        client = config.create()

    assert not set(sampling) & client._create_options.keys()
    assert client._extra_body == (sampling if sent else None)


def test_no_warning_without_sampling_parameters() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        AnthropicConfig(model="claude-sonnet-5").create()


def test_sampling_parameters_survive_copy() -> None:
    config = AnthropicConfig(model="claude-sonnet-5", temperature=0.2)

    assert config.copy(max_tokens=10).temperature == 0.2


def test_max_tokens_can_be_overridden() -> None:
    config = AnthropicConfig(model="claude-sonnet-5", max_tokens=8192)
    assert config.max_tokens == 8192


def test_speed_defaults_to_none() -> None:
    config = AnthropicConfig(model="claude-opus-5-5")
    assert config.speed is None


@pytest.mark.parametrize("speed", ["standard", "fast", None])
def test_copy_with_speed(speed: Literal["standard", "fast"] | None) -> None:
    config = AnthropicConfig(model="claude-opus-5-5", speed="fast")

    copied = config.copy(speed=speed)

    assert copied == AnthropicConfig(model="claude-opus-5-5", speed=speed)
    assert config.speed == "fast"
    assert copied is not config


def test_extra_body_defaults_to_none() -> None:
    config = AnthropicConfig(model="claude-sonnet-5")
    assert config.extra_body is None


def test_extra_body_can_be_set() -> None:
    extra = {"tool_choice": {"type": "auto", "disable_parallel_tool_use": True}}
    config = AnthropicConfig(model="claude-sonnet-5", extra_body=extra)

    assert config.extra_body == extra


def test_extra_body_passed_to_client() -> None:
    extra = {"tool_choice": {"type": "auto", "disable_parallel_tool_use": True}}
    config = AnthropicConfig(model="claude-sonnet-5", api_key="test-key", extra_body=extra)

    client = config.create()

    assert client._extra_body == extra


def test_copy_with_extra_body() -> None:
    base = AnthropicConfig(model="claude-sonnet-5")
    extra = {"thinking": {"type": "enabled", "budget_tokens": 5000}}

    copied = base.copy(extra_body=extra)

    assert copied.extra_body == extra
    assert base.extra_body is None


def test_drop_sampling_keeps_extra_body() -> None:
    with pytest.warns(DeprecationWarning, match="removed in 1.4.0"):
        assert _compat._drop_sampling({"top_k": 5}, {"a": 1}) == {"a": 1}


def test_send_sampling_merges_and_extra_body_wins() -> None:
    with pytest.warns(DeprecationWarning, match="removed in 1.4.0"):
        result = _compat._send_sampling({"temperature": 0.2, "top_k": 5}, {"top_k": 9})

    assert result == {"temperature": 0.2, "top_k": 9}


def test_open_telemetry_defaults_to_none() -> None:
    config = AnthropicConfig(model="claude-sonnet-5")

    assert config.open_telemetry is None


@pytest.mark.parametrize("open_telemetry", [None, False, {"traces": {"content_mode": "content"}}])
def test_open_telemetry_passed_to_sdk_clients(open_telemetry: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from ag2.config.anthropic import anthropic_client, files

    calls: list[dict[str, Any]] = []

    def fake_async_anthropic(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(anthropic_client, "AsyncAnthropic", fake_async_anthropic)
    monkeypatch.setattr(files, "AsyncAnthropic", fake_async_anthropic)
    config = AnthropicConfig(model="claude-sonnet-5", api_key="test-key", open_telemetry=open_telemetry)

    config.create()
    config.create_files_client()

    assert [call["open_telemetry"] for call in calls] == [open_telemetry, open_telemetry]


def test_copy_with_open_telemetry() -> None:
    base = AnthropicConfig(model="claude-sonnet-5")

    copied = base.copy(open_telemetry=False)

    assert copied.open_telemetry is False
    assert base.open_telemetry is None
