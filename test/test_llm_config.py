# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.llm_config import LLMConfig, OpenAILLMConfigEntry

# def test_current_llm_config():
#     llm_config = LLMConfig(
#         config_list=[LLMConfigEntry(api_type=LLMProvider.openai, model="model", api_key="api_key")]
#     )
#     with llm_config:
#         print(current_llm_config.get())
#         assert current_llm_config.get() == llm_config.model_dump_json()
#     with pytest.raises(LookupError):
#         current_llm_config.get()


@pytest.fixture
def openai_llm_config_entry() -> OpenAILLMConfigEntry:
    return OpenAILLMConfigEntry(model="gpt-4o-mini", api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly")


class TestLLMConfigEntry:
    def test_serialization(self, openai_llm_config_entry: OpenAILLMConfigEntry) -> None:
        actual = openai_llm_config_entry.model_dump()
        expected = {
            "api_type": "openai",
            "model": "gpt-4o-mini",
            "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
            "tags": [],
        }
        assert actual == expected

    def test_deserialization(self, openai_llm_config_entry: OpenAILLMConfigEntry) -> None:
        actual = OpenAILLMConfigEntry(**openai_llm_config_entry.model_dump())
        assert actual == openai_llm_config_entry


class TestLLMConfig:
    @pytest.fixture
    def openai_llm_config(self, openai_llm_config_entry: OpenAILLMConfigEntry) -> LLMConfig:
        return LLMConfig(config_list=[openai_llm_config_entry], temperature=0.5, check_every_ms=1000, cache_seed=42)

    def test_serialization(self, openai_llm_config: LLMConfig) -> None:
        actual = openai_llm_config.model_dump()
        expected = {
            "config_list": [
                {
                    "api_type": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    "tags": [],
                }
            ],
            "temperature": 0.5,
            "check_every_ms": 1000,
            "cache_seed": 42,
        }
        assert actual == expected

    def test_deserialization(self, openai_llm_config: LLMConfig) -> None:
        actual = LLMConfig(**openai_llm_config.model_dump())
        print(actual)
        print(actual.model_dump())
        assert actual.model_dump() == openai_llm_config.model_dump()
        assert type(actual._model) == type(openai_llm_config._model)
        assert actual._model == openai_llm_config._model
        assert actual == openai_llm_config
        # assert isinstance(actual.config_list[0], OpenAILLMConfigEntry)
