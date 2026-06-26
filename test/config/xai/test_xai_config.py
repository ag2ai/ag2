# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from ag2.config import XAIConfig
from ag2.config.xai import XAIClient, XAIFilesClient


def test_copy_without_overrides_returns_new_equal_instance() -> None:
    config = XAIConfig(model="grok-4-fast", temperature=0.2, streaming=True)

    copied = config.copy()

    assert copied == config
    assert copied is not config


def test_copy_applies_overrides_without_mutating_original() -> None:
    config = XAIConfig(model="grok-4-fast", api_key="key", temperature=0.2, streaming=False)

    copied = config.copy(model="grok-4.20", temperature=0.8, streaming=True, api_key=None)

    assert copied.model == "grok-4.20"
    assert copied.temperature == 0.8
    assert copied.streaming is True
    assert copied.api_key is None

    assert config.model == "grok-4-fast"
    assert config.temperature == 0.2
    assert config.streaming is False
    assert config.api_key == "key"  # pragma: allowlist secret


def test_create_returns_xai_client() -> None:
    config = XAIConfig(model="grok-4-fast", api_key="test-key")

    with patch("ag2.config.xai.xai_client.AsyncClient"):
        client = config.create()

    assert isinstance(client, XAIClient)


def test_defaults() -> None:
    config = XAIConfig(model="grok-4-fast")

    assert config.api_host == "api.x.ai"
    assert config.timeout is None
    assert config.api_key is None
    assert config.streaming is False
    assert config.temperature is None
    assert config.max_tokens is None
    assert config.reasoning_effort is None
    assert config.include is None


def test_api_host_can_be_overridden() -> None:
    config = XAIConfig(model="grok-4-fast", api_host="custom.x.ai")

    assert config.api_host == "custom.x.ai"


def test_reasoning_effort_can_be_set() -> None:
    config = XAIConfig(model="grok-4-fast", reasoning_effort="high")

    assert config.reasoning_effort == "high"


def test_include_can_be_set() -> None:
    config = XAIConfig(model="grok-4-fast", include=["verbose_streaming", "code_execution_call_output"])

    assert config.include == ["verbose_streaming", "code_execution_call_output"]


def test_create_files_client_returns_xai_files_client() -> None:
    config = XAIConfig(model="grok-4-fast")

    with patch("ag2.config.xai.files.AsyncClient"):
        client = config.create_files_client()

    assert isinstance(client, XAIFilesClient)
