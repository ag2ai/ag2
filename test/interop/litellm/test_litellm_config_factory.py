# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from typing import Any

import pytest

from autogen.interop import LiteLLmConfigFactory


class TestLiteLLmConfigFactory:
    def test_number_of_factories(self) -> None:
        assert len(LiteLLmConfigFactory._factories) == 3

    @pytest.mark.parametrize(
        ("config_list", "expected"),
        [
            (
                [{"api_type": "openai", "model": "gpt-4o-mini", "api_key": ""}],
                {"api_token": "", "provider": "openai/gpt-4o-mini"},
            ),
            (
                [
                    {"api_type": "deepseek", "model": "deepseek-model", "api_key": "", "base_url": "test-url"},
                ],
                {"base_url": "test-url", "api_token": "", "provider": "deepseek/deepseek-model"},
            ),
            (
                [
                    {
                        "api_type": "azure",
                        "model": "gpt-4o-mini",
                        "api_key": "",
                        "base_url": "test",
                        "api_version": "test",
                    },
                ],
                {"base_url": "test", "api_version": "test", "api_token": "", "provider": "azure/gpt-4o-mini"},
            ),
            (
                [
                    {"api_type": "google", "model": "gemini", "api_key": ""},
                ],
                {"api_token": "", "provider": "gemini/gemini"},
            ),
            (
                [
                    {"api_type": "anthropic", "model": "sonnet", "api_key": ""},
                ],
                {"api_token": "", "provider": "anthropic/sonnet"},
            ),
            (
                [{"api_type": "ollama", "model": "mistral:7b"}],
                {"provider": "ollama/mistral:7b"},
            ),
            (
                [{"api_type": "ollama", "model": "mistral:7b", "client_host": "http://127.0.0.1:11434"}],
                {"api_base": "http://127.0.0.1:11434", "provider": "ollama/mistral:7b"},
            ),
        ],
    )
    def test_get_provider_and_api_key(self, config_list: list[dict[str, Any]], expected: dict[str, Any]) -> None:
        lite_llm_config = LiteLLmConfigFactory.create_lite_llm_config({"config_list": config_list})
        assert lite_llm_config == expected
