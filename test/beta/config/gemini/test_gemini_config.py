# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.config import GeminiConfig
from autogen.beta.config.gemini import GeminiClient


def test_copy_without_overrides_returns_new_equal_instance() -> None:
    config = GeminiConfig(model="gemini-2.0-flash", temperature=0.2, streaming=True)

    copied = config.copy()

    assert copied == config
    assert copied is not config


def test_copy_applies_overrides_without_mutating_original() -> None:
    config = GeminiConfig(model="gemini-2.0-flash", api_key="key", temperature=0.2, streaming=False)

    copied = config.copy(model="gemini-2.5-flash", temperature=0.8, streaming=True, api_key=None)

    assert copied.model == "gemini-2.5-flash"
    assert copied.temperature == 0.8
    assert copied.streaming is True
    assert copied.api_key is None

    assert config.model == "gemini-2.0-flash"
    assert config.temperature == 0.2
    assert config.streaming is False
    assert config.api_key == "key"


def test_create_returns_gemini_client() -> None:
    config = GeminiConfig(model="gemini-2.0-flash", api_key="test-key")
    client = config.create()

    assert isinstance(client, GeminiClient)


def test_defaults() -> None:
    config = GeminiConfig(model="gemini-2.0-flash")
    assert config.streaming is False
    assert config.temperature is None
    assert config.max_output_tokens is None
    assert config.api_key is None


def test_max_output_tokens_can_be_set() -> None:
    config = GeminiConfig(model="gemini-2.0-flash", max_output_tokens=8192)
    assert config.max_output_tokens == 8192
