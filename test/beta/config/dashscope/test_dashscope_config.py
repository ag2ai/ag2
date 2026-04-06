# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.config import DashScopeConfig
from autogen.beta.config.dashscope import DashScopeClient


def test_copy_without_overrides_returns_new_equal_instance() -> None:
    config = DashScopeConfig(model="qwen-plus", temperature=0.2, streaming=True)

    copied = config.copy()

    assert copied == config
    assert copied is not config


def test_copy_applies_overrides_without_mutating_original() -> None:
    config = DashScopeConfig(model="qwen-plus", temperature=0.2, streaming=False)

    copied = config.copy(model="qwen-max", temperature=0.8, streaming=True)

    assert copied.model == "qwen-max"
    assert copied.temperature == 0.8
    assert copied.streaming is True

    assert config.model == "qwen-plus"
    assert config.temperature == 0.2
    assert config.streaming is False


def test_create_returns_dashscope_client() -> None:
    config = DashScopeConfig(model="qwen-plus", api_key="test-key")
    client = config.create()

    assert isinstance(client, DashScopeClient)


def test_defaults() -> None:
    config = DashScopeConfig(model="qwen-plus")
    assert config.base_url == "https://dashscope-intl.aliyuncs.com/api/v1"
    assert config.api_key is None
    assert config.streaming is False
    assert config.temperature is None
    assert config.max_tokens is None


def test_base_url_can_be_overridden() -> None:
    config = DashScopeConfig(model="qwen-plus", base_url="https://dashscope.aliyuncs.com/api/v1")
    assert config.base_url == "https://dashscope.aliyuncs.com/api/v1"
