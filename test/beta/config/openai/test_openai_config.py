# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.config import OpenAIConfig


def test_copy_without_overrides_returns_new_equal_instance() -> None:
    config = OpenAIConfig(model="gpt-5", temperature=0.2, streaming=True)

    copied = config.copy()

    assert copied == config
    assert copied is not config


def test_copy_applies_overrides_without_mutating_original() -> None:
    config = OpenAIConfig(model="gpt-5", api_key="key", temperature=0.2, streaming=False)

    copied = config.copy(model="gpt-5-mini", temperature=0.8, streaming=True, api_key=None)

    assert copied.model == "gpt-5-mini"
    assert copied.temperature == 0.8
    assert copied.streaming is True
    assert copied.api_key is None

    assert config.model == "gpt-5"
    assert config.temperature == 0.2
    assert config.streaming is False
    assert config.api_key == "key"


def test_extra_body_defaults_to_none() -> None:
    config = OpenAIConfig(model="gpt-5")

    assert config.extra_body is None


def test_extra_body_passed_to_create_options() -> None:
    extra = {"chat_template_kwargs": {"enable_thinking": True}}
    config = OpenAIConfig(model="test-model", base_url="http://localhost:8000/v1", api_key="EMPTY", extra_body=extra)

    client = config.create()

    assert client._create_options.get("extra_body") == extra


def test_copy_with_extra_body() -> None:
    config = OpenAIConfig(model="gpt-5")
    extra = {"reasoning_split": True}

    copied = config.copy(extra_body=extra)

    assert copied.extra_body == extra
    assert config.extra_body is None
