# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.config import OllamaConfig
from autogen.beta.config.ollama import OllamaClient


def test_copy_without_overrides_returns_new_equal_instance() -> None:
    config = OllamaConfig(model="qwen3.5:latest", temperature=0.2, streaming=True)

    copied = config.copy()

    assert copied == config
    assert copied is not config


def test_copy_applies_overrides_without_mutating_original() -> None:
    config = OllamaConfig(model="qwen3.5:latest", temperature=0.2, streaming=False)

    copied = config.copy(model="llama3:latest", temperature=0.8, streaming=True)

    assert copied.model == "llama3:latest"
    assert copied.temperature == 0.8
    assert copied.streaming is True

    assert config.model == "qwen3.5:latest"
    assert config.temperature == 0.2
    assert config.streaming is False


def test_create_returns_ollama_client() -> None:
    config = OllamaConfig(model="qwen3.5:latest")
    client = config.create()

    assert isinstance(client, OllamaClient)


def test_defaults() -> None:
    config = OllamaConfig(model="qwen3.5:latest")
    assert config.host == "http://localhost:11434"
    assert config.streaming is False
    assert config.temperature is None
    assert config.max_tokens is None


def test_host_can_be_overridden() -> None:
    config = OllamaConfig(model="qwen3.5:latest", host="http://remote:11434")
    assert config.host == "http://remote:11434"
