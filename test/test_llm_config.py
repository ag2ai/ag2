# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.llm_config import OpenAILLMConfigEntry
from test.conftest import Credentials

# def test_llm_provider():
#     assert LLMProvider.openai == "openai"
#     assert LLMProvider.bedrock == "bedrock"
#     assert LLMProvider.anthropic == "anthropic"
#     assert LLMProvider.cerebras == "cerebras"
#     assert LLMProvider.cohere == "cohere"
#     assert LLMProvider.deepseek == "deepseek"
#     assert LLMProvider.google == "google"
#     assert LLMProvider.groq == "groq"
#     assert LLMProvider.mistral == "mistral"
#     assert LLMProvider.ollama == "ollama"
#     assert LLMProvider.together == "together"


# def test_individual_llm_config():
#     individual_llm_config = LLMConfigEntry(api_type=LLMProvider.openai, model="model", api_key="api_key")
#     assert individual_llm_config.api_type == LLMProvider.openai
#     assert individual_llm_config.model == "model"
#     assert individual_llm_config.api_key == "api_key"


# def test_llm_config():
#     llm_config = LLMConfig(
#         config_list=[LLMConfigEntry(api_type=LLMProvider.openai, model="model", api_key="api_key")]
#     )
#     assert llm_config.config_list[0].api_type == LLMProvider.openai
#     assert llm_config.config_list[0].model == "model"
#     assert llm_config.config_list[0].api_key == "api_key"
#     assert llm_config.config_list[0].base_url is None
#     assert llm_config.config_list[0].tags is None


# def test_current_llm_config():
#     llm_config = LLMConfig(
#         config_list=[LLMConfigEntry(api_type=LLMProvider.openai, model="model", api_key="api_key")]
#     )
#     with llm_config:
#         print(current_llm_config.get())
#         assert current_llm_config.get() == llm_config.model_dump_json()
#     with pytest.raises(LookupError):
#         current_llm_config.get()


class TestLLMConfig:
    @pytest.fixture
    def openai_llm_config_entry(self, mock_credentials: Credentials) -> OpenAILLMConfigEntry:
        return OpenAILLMConfigEntry(model="gpt-4o-mini", api_key=mock_credentials.api_key)

    def test_serialization(self, openai_llm_config_entry: OpenAILLMConfigEntry):
        actual = openai_llm_config_entry.model_dump()
        expected = {
            "api_type": "openai",
            "model": "gpt-4o-mini",
            "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
            "tags": [],
        }
        assert actual == expected
