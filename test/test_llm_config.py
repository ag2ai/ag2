# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.llm_config import IndividualLLMConfig, LLMConfig, LLMProvider, current_llm_config


def test_llm_provider():
    assert LLMProvider.openai == "openai"
    assert LLMProvider.bedrock == "bedrock"
    assert LLMProvider.anthropic == "anthropic"
    assert LLMProvider.cerebras == "cerebras"
    assert LLMProvider.cohere == "cohere"
    assert LLMProvider.deepseek == "deepseek"
    assert LLMProvider.google == "google"
    assert LLMProvider.groq == "groq"
    assert LLMProvider.mistral == "mistral"
    assert LLMProvider.ollama == "ollama"
    assert LLMProvider.together == "together"


def test_individual_llm_config():
    individual_llm_config = IndividualLLMConfig(api_type=LLMProvider.openai, model="model", api_key="api_key")
    assert individual_llm_config.api_type == LLMProvider.openai
    assert individual_llm_config.model == "model"
    assert individual_llm_config.api_key == "api_key"


def test_llm_config():
    llm_config = LLMConfig(
        config_list=[IndividualLLMConfig(api_type=LLMProvider.openai, model="model", api_key="api_key")]
    )
    assert llm_config.config_list[0].api_type == LLMProvider.openai
    assert llm_config.config_list[0].model == "model"
    assert llm_config.config_list[0].api_key == "api_key"
    assert llm_config.config_list[0].base_url is None
    assert llm_config.config_list[0].tags is None


def test_current_llm_config():
    llm_config = LLMConfig(
        config_list=[IndividualLLMConfig(api_type=LLMProvider.openai, model="model", api_key="api_key")]
    )
    with llm_config:
        print(current_llm_config.get())
        assert current_llm_config.get() == llm_config.model_dump_json()
    with pytest.raises(LookupError):
        current_llm_config.get()
