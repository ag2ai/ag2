# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

pytest.importorskip("openai")

from openai import AsyncOpenAI

from ag2.live import (
    OpenAIRealTimeConfig,
    OpenAITTSConfig,
    OpenAITranscriber,
    OpenAITranslationTranscriber,
)

CONFIGS_WITH_CLIENT = [OpenAIRealTimeConfig, OpenAITranscriber, OpenAITranslationTranscriber]


@pytest.mark.parametrize("config_class", CONFIGS_WITH_CLIENT)
class TestClientAttribute:
    def test_explicit_key_reaches_client(self, config_class: Any) -> None:
        config = config_class("test-model", api_key="test-explicit-key")

        assert config.client.api_key == "test-explicit-key"

    def test_injected_client_is_reused(self, config_class: Any) -> None:
        client = AsyncOpenAI(api_key="test-injected-key")

        config = config_class("test-model", client=client)

        assert config.client is client


@pytest.mark.parametrize("config_class", [*CONFIGS_WITH_CLIENT, OpenAITTSConfig])
@pytest.mark.parametrize("key", ["test-other-key", "test-injected-key", ""], ids=["different", "same", "empty"])
def test_client_and_key_conflict(config_class: Any, key: str) -> None:
    client = AsyncOpenAI(api_key="test-injected-key")

    with pytest.raises(ValueError, match="client.*api_key"):
        config_class("test-model", client=client, api_key=key)
