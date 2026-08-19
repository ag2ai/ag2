# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.config import ModelProvider, OrcaRouterConfig
from ag2.config.orcarouter import OrcaRouterClient
from ag2.config.orcarouter.config import ORCAROUTER_DEFAULT_BASE_URL, ORCAROUTER_DEFAULT_MODEL


def test_defaults() -> None:
    config = OrcaRouterConfig()

    assert config.model == ORCAROUTER_DEFAULT_MODEL
    assert config.api_key is None
    assert config.base_url is None
    assert config.streaming is False


def test_provider() -> None:
    config = OrcaRouterConfig()

    assert config.provider == ModelProvider.ORCAROUTER


def test_copy_returns_equal_new_instance() -> None:
    config = OrcaRouterConfig(model="orcarouter/auto", temperature=0.2)

    copied = config.copy()

    assert copied == config
    assert copied is not config


def test_copy_applies_overrides_without_mutating_original() -> None:
    config = OrcaRouterConfig(model="orcarouter/auto", temperature=0.2)

    copied = config.copy(model="anthropic/claude-3.5-sonnet", temperature=0.7)

    assert copied == OrcaRouterConfig(model="anthropic/claude-3.5-sonnet", temperature=0.7)
    assert config == OrcaRouterConfig(model="orcarouter/auto", temperature=0.2)


def test_create_returns_client() -> None:
    assert isinstance(OrcaRouterConfig(api_key="sk-orca-test").create(), OrcaRouterClient)


def test_create_applies_env_key_and_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test")
    config = OrcaRouterConfig()

    client = config.create()

    assert client._client.api_key == "sk-orca-test"
    assert str(client._client.base_url).rstrip("/") == ORCAROUTER_DEFAULT_BASE_URL


def test_create_uses_explicit_api_key_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-env")
    config = OrcaRouterConfig(api_key="sk-orca-explicit", base_url="https://example.test/v1")

    client = config.create()

    assert client._client.api_key == "sk-orca-explicit"
    assert str(client._client.base_url).rstrip("/") == "https://example.test/v1"


def test_streaming_defaults_to_false() -> None:
    config = OrcaRouterConfig()

    assert config.streaming is False
