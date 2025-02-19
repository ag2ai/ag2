# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

import pytest

from autogen.import_utils import optional_import_block, skip_on_missing_imports
from autogen.tools.experimental.browser_use.langchain_factory import ChatOpenAIFactory, LangchainFactory

with optional_import_block():
    from langchain_openai import ChatOpenAI


@skip_on_missing_imports(
    ["langchain_anthropic", "langchain_google_genai", "langchain_ollama", "langchain_openai", "langchain_core"],
    "browser-use",
)
class TestLangchainFactory:
    test_api_key = "test"  # pragma: allowlist secret

    def test_number_of_factories(self) -> None:
        assert len(LangchainFactory._factories) == 6

    @pytest.mark.parametrize(
        ("llm_config", "expected"),
        [
            (
                {"model": "gpt-4o-mini", "api_key": test_api_key},
                {"model": "gpt-4o-mini", "api_key": test_api_key},
            ),
            (
                {"config_list": [{"model": "gpt-4o-mini", "api_key": test_api_key}]},
                {"model": "gpt-4o-mini", "api_key": test_api_key},
            ),
            (
                {
                    "config_list": [
                        {"model": "gpt-4o-mini", "api_key": test_api_key},
                        {"model": "gpt-4o", "api_key": test_api_key},
                    ]
                },
                {"model": "gpt-4o-mini", "api_key": test_api_key},
            ),
        ],
    )
    def test_get_first_llm_config(self, llm_config: dict[str, Any], expected: dict[str, Any]) -> None:
        assert LangchainFactory.get_first_llm_config(llm_config) == expected

    @pytest.mark.parametrize(
        ("llm_config", "error_message"),
        [
            ({}, "llm_config must be a valid config dictionary."),
            ({"config_list": []}, "Config list must contain at least one config."),
        ],
    )
    def test_get_first_llm_config_incorrect_config(self, llm_config: dict[str, Any], error_message: str) -> None:
        with pytest.raises(ValueError, match=error_message):
            LangchainFactory.get_first_llm_config(llm_config)

    @pytest.mark.parametrize(
        ("config_list", "llm_class_name", "base_url"),
        [
            (
                [
                    {"api_type": "openai", "model": "gpt-4o-mini", "api_key": test_api_key},
                ],
                "ChatOpenAI",
                None,
            ),
            (
                [
                    {
                        "api_type": "deepseek",
                        "model": "deepseek-model",
                        "api_key": test_api_key,
                        "base_url": "test-url",
                    },
                ],
                "ChatOpenAI",
                "test-url",
            ),
            (
                [
                    {
                        "api_type": "azure",
                        "model": "gpt-4o-mini",
                        "api_key": test_api_key,
                        "base_url": "test-url",
                        "api_version": "test",
                    },
                ],
                "AzureChatOpenAI",
                "test-url",
            ),
            (
                [
                    {"api_type": "google", "model": "gemini", "api_key": test_api_key},
                ],
                "ChatGoogleGenerativeAI",
                None,
            ),
            (
                [
                    {"api_type": "anthropic", "model": "sonnet", "api_key": test_api_key},
                ],
                "ChatAnthropic",
                None,
            ),
            (
                [{"api_type": "ollama", "model": "mistral:7b-instruct-v0.3-q6_K"}],
                "ChatOllama",
                None,
            ),
            (
                [{"api_type": "ollama", "model": "mistral:7b-instruct-v0.3-q6_K", "client_host": "test-url"}],
                "ChatOllama",
                "test-url",
            ),
        ],
    )
    def test_create_base_chat_model(  # type: ignore[no-any-unimported]
        self,
        config_list: list[dict[str, str]],
        llm_class_name: str,
        base_url: Optional[str],
    ) -> None:
        llm = LangchainFactory.create_base_chat_model(llm_config={"config_list": config_list})
        assert llm.__class__.__name__ == llm_class_name
        if llm_class_name == "AzureChatOpenAI":
            assert llm.azure_endpoint == base_url
        elif llm_class_name == "ChatOpenAI" and base_url:
            assert llm.openai_api_base == base_url
        elif base_url:
            assert llm.base_url == base_url

    @pytest.mark.parametrize(
        ("config_list", "error_msg"),
        [
            (
                [
                    {"api_type": "deepseek", "model": "gpt-4o-mini", "api_key": test_api_key},
                ],
                "base_url is required for deepseek api type.",
            ),
            (
                [
                    {"api_type": "azure", "model": "gpt-4o-mini", "api_key": test_api_key, "base_url": "test"},
                ],
                "api_version is required for azure api type.",
            ),
        ],
    )
    def test_create_base_chat_model_raises_if_mandatory_key_missing(
        self, config_list: list[dict[str, str]], error_msg: str
    ) -> None:
        with pytest.raises(ValueError, match=error_msg):
            LangchainFactory.create_base_chat_model(llm_config={"config_list": config_list})


@skip_on_missing_imports(
    ["langchain_anthropic", "langchain_google_genai", "langchain_ollama", "langchain_openai", "langchain_core"],
    "browser-use",
)
class TestChatOpenAIFactory:
    test_api_key = "test"  # pragma: allowlist secret

    @pytest.mark.parametrize(
        "llm_config",
        [
            {"model": "gpt-4o-mini", "api_key": test_api_key},
            {"config_list": [{"model": "gpt-4o-mini", "api_key": test_api_key}]},
        ],
    )
    def test_create(self, llm_config: dict[str, Any]) -> None:
        assert isinstance(ChatOpenAIFactory.create(llm_config), ChatOpenAI)
