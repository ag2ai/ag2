# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from _collections_abc import dict_items, dict_keys, dict_values
from copy import copy, deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from autogen.llm_config import LLMConfig
from autogen.oai import (
    AnthropicLLMConfigEntry,
    AzureOpenAILLMConfigEntry,
    BedrockLLMConfigEntry,
    CerebrasLLMConfigEntry,
    CohereLLMConfigEntry,
    DeepSeekLLMConfigEntry,
    GeminiLLMConfigEntry,
    GroqLLMConfigEntry,
    MistralLLMConfigEntry,
    OllamaLLMConfigEntry,
    OpenAILLMConfigEntry,
    OpenAIResponsesLLMConfigEntry,
    TogetherLLMConfigEntry,
)

JSON_SAMPLE = """
[
    {
        "model": "gpt-3.5-turbo",
        "api_type": "openai",
        "tags": ["gpt35"]
    },
    {
        "model": "gpt-4",
        "api_type": "openai",
        "tags": ["gpt4"]
    },
    {
        "model": "gpt-35-turbo-v0301",
        "tags": ["gpt-3.5-turbo", "gpt35_turbo"],
        "api_key": "Your Azure OAI API Key",
        "base_url": "https://deployment_name.openai.azure.com/",
        "api_type": "azure",
        "api_version": "2024-02-01"
    },
    {
        "model": "gpt",
        "api_key": "not-needed",
        "base_url": "http://localhost:1234/v1",
        "tags": []
    }
]
"""

JSON_SAMPLE_DICT = json.loads(JSON_SAMPLE)


@pytest.fixture
def openai_llm_config_entry() -> OpenAILLMConfigEntry:
    return OpenAILLMConfigEntry(model="gpt-4o-mini", api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly")


class TestLLMConfigEntry:
    def test_extra_fields(self) -> None:
        assert (
            OpenAILLMConfigEntry(  # type: ignore[attr-defined]
                model="gpt-4o-mini",
                api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                extra="extra",
            ).extra
            == "extra"
        )

    def test_serialization(self, openai_llm_config_entry: OpenAILLMConfigEntry) -> None:
        actual = openai_llm_config_entry.model_dump()
        expected = {
            "api_type": "openai",
            "model": "gpt-4o-mini",
            "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
            "tags": [],
            "stream": False,
        }
        assert actual == expected

    def test_deserialization(self, openai_llm_config_entry: OpenAILLMConfigEntry) -> None:
        actual = OpenAILLMConfigEntry(**openai_llm_config_entry.model_dump())
        assert actual == openai_llm_config_entry

    def test_get(self, openai_llm_config_entry: OpenAILLMConfigEntry) -> None:
        assert openai_llm_config_entry.get("api_type") == "openai"
        assert openai_llm_config_entry.get("model") == "gpt-4o-mini"
        assert openai_llm_config_entry.get("api_key") == "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly"
        assert openai_llm_config_entry.get("doesnt_exists") is None
        assert openai_llm_config_entry.get("doesnt_exists", "default") == "default"

    def test_get_item_and_set_item(self, openai_llm_config_entry: OpenAILLMConfigEntry) -> None:
        # Test __getitem__
        assert openai_llm_config_entry["api_type"] == "openai"
        assert openai_llm_config_entry["model"] == "gpt-4o-mini"
        assert openai_llm_config_entry["api_key"] == "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly"
        assert openai_llm_config_entry["tags"] == []
        with pytest.raises(KeyError) as e:
            openai_llm_config_entry["wrong_key"]
        assert str(e.value) == "\"Key 'wrong_key' not found in OpenAILLMConfigEntry\""

        # Test __setitem__
        assert openai_llm_config_entry["base_url"] is None
        openai_llm_config_entry["base_url"] = "https://api.openai.com"
        assert openai_llm_config_entry["base_url"] == "https://api.openai.com"
        openai_llm_config_entry["base_url"] = None
        assert openai_llm_config_entry["base_url"] is None


class TestLLMConfig:
    @pytest.fixture
    def openai_llm_config(self, openai_llm_config_entry: OpenAILLMConfigEntry) -> LLMConfig:
        return LLMConfig(
            openai_llm_config_entry,
            temperature=0.5,
            check_every_ms=1000,
            cache_seed=42,
        )

    def test_init_with_extras(self) -> None:
        assert LLMConfig(
            {
                "model": "gpt-4o-mini",
                "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                "api_type": "openai",
                "extra": "extra",
            },
            temperature=0.5,
            max_tokens=1024,
            check_every_ms=1000,
            cache_seed=42,
        ) == LLMConfig(
            OpenAILLMConfigEntry(
                model="gpt-4o-mini",
                api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                temperature=0.5,
                max_tokens=1024,
                extra="extra",
            ),
            temperature=0.5,
            max_tokens=1024,
            check_every_ms=1000,
            cache_seed=42,
        )

    @pytest.mark.parametrize(
        (
            "llm_config",
            "expected",
        ),
        [
            pytest.param(
                {
                    "model": "o3",
                    "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    "max_completion_tokens": 1024,
                    "reasoning_effort": "low",
                },
                LLMConfig(
                    OpenAILLMConfigEntry(
                        model="o3",
                        api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                        max_completion_tokens=1024,
                        reasoning_effort="low",
                    )
                ),
                id="openai o3",
            ),
            pytest.param(
                {
                    "model": "o3",
                    "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    "api_type": "azure",
                    "base_url": "https://api.openai.com",
                    "max_completion_tokens": 1024,
                    "reasoning_effort": "low",
                },
                LLMConfig(
                    AzureOpenAILLMConfigEntry(
                        model="o3",
                        api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                        base_url="https://api.openai.com",
                        max_completion_tokens=1024,
                        reasoning_effort="low",
                    )
                ),
                id="azure o3",
            ),
            pytest.param(
                {
                    "api_type": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "api_key": "dummy_api_key",
                    "stream": False,
                    "temperature": 1.0,
                    "max_tokens": 100,
                },
                LLMConfig(
                    AnthropicLLMConfigEntry(
                        model="claude-sonnet-4-5",
                        api_key="dummy_api_key",
                        stream=False,
                        temperature=1.0,
                        max_tokens=100,
                    )
                ),
                id="anthropic claude-sonnet-4-5",
            ),
            pytest.param(
                {
                    "api_type": "bedrock",
                    "model": "anthropic.claude-4-5-sonnet-20250929-v1:0",
                    "aws_region": "us-east-1",
                    "aws_access_key": "test_access_key_id",
                    "aws_secret_key": "test_secret_access_key",
                    "aws_session_token": "test_session_token",
                    "temperature": 0.8,
                    "supports_system_prompts": True,
                },
                LLMConfig(
                    BedrockLLMConfigEntry(
                        model="anthropic.claude-4-5-sonnet-20250929-v1:0",
                        aws_region="us-east-1",
                        aws_access_key="test_access_key_id",
                        aws_secret_key="test_secret_access_key",
                        aws_session_token="test_session_token",
                        temperature=0.8,
                    )
                ),
                id="bedrock claude-4-5-sonnet",
            ),
            pytest.param(
                {
                    "api_type": "cerebras",
                    "api_key": "fake_api_key",
                    "model": "llama3.1-8b",
                    "max_tokens": 1000,
                    "seed": 42,
                    "stream": False,
                    "temperature": 1.0,
                },
                LLMConfig(
                    CerebrasLLMConfigEntry(
                        api_key="fake_api_key",
                        model="llama3.1-8b",
                        max_tokens=1000,
                        seed=42,
                        stream=False,
                        temperature=1.0,
                    )
                ),
                id="cerebras llama3.1-8b",
            ),
            pytest.param(
                {
                    "api_type": "cohere",
                    "model": "command-r-plus",
                    "api_key": "dummy_api_key",
                    "frequency_penalty": 0,
                    "k": 0,
                    "top_p": 0.75,
                    "presence_penalty": 0,
                    "strict_tools": False,
                    "tags": [],
                },
                LLMConfig(
                    CohereLLMConfigEntry(
                        model="command-r-plus",
                        api_key="dummy_api_key",
                        top_p=0.75,
                    )
                ),
                id="cohere command-r-plus",
            ),
            pytest.param(
                {
                    "api_type": "deepseek",
                    "api_key": "fake_api_key",
                    "model": "deepseek-chat",
                    "base_url": "https://api.deepseek.com/v1",
                    "max_tokens": 8192,
                    "temperature": 0.5,
                },
                LLMConfig(
                    DeepSeekLLMConfigEntry(
                        api_key="fake_api_key",
                        model="deepseek-chat",
                        temperature=0.5,
                    )
                ),
                id="deepseek deepseek-chat",
            ),
            pytest.param(
                {
                    "api_type": "google",
                    "model": "gemini-2.0-flash-lite",
                    "api_key": "dummy_api_key",
                    "project_id": "fake-project-id",
                    "location": "us-west1",
                    "stream": False,
                },
                LLMConfig(
                    GeminiLLMConfigEntry(
                        model="gemini-2.0-flash-lite",
                        api_key="dummy_api_key",
                        project_id="fake-project-id",
                        location="us-west1",
                    )
                ),
                id="google gemini-2.0-flash-lite",
            ),
            pytest.param(
                {
                    "api_type": "groq",
                    "model": "llama3-8b-8192",
                    "api_key": "fake_api_key",
                    "temperature": 1.0,
                },
                LLMConfig(
                    GroqLLMConfigEntry(
                        api_key="fake_api_key",
                        model="llama3-8b-8192",
                        temperature=1.0,
                    ),
                ),
                id="groq llama3-8b-8192",
            ),
            pytest.param(
                {
                    "api_type": "mistral",
                    "model": "mistral-small-latest",
                    "api_key": "fake_api_key",
                    "temperature": 0.7,
                },
                LLMConfig(
                    MistralLLMConfigEntry(
                        model="mistral-small-latest",
                        api_key="fake_api_key",
                        temperature=0.7,
                    )
                ),
                id="mistral mistral-small-latest",
            ),
            pytest.param(
                {
                    "api_type": "ollama",
                    "model": "llama3.1:8b",
                    "num_ctx": 2048,
                    "num_predict": -1,
                    "repeat_penalty": 1.1,
                    "seed": 0,
                    "stream": False,
                    "tags": [],
                    "temperature": 0.8,
                    "top_k": 40,
                    "native_tool_calls": False,
                },
                LLMConfig(
                    OllamaLLMConfigEntry(model="llama3.1:8b", temperature=0.8),
                ),
                id="ollama llama3.1:8b",
            ),
            pytest.param(
                {
                    "api_type": "together",
                    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "api_key": "fake_api_key",
                    "safety_model": "Meta-Llama/Llama-Guard-7b",
                    "tags": [],
                    "max_tokens": 512,
                    "stream": False,
                },
                LLMConfig(
                    TogetherLLMConfigEntry(
                        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        api_key="fake_api_key",
                        safety_model="Meta-Llama/Llama-Guard-7b",
                    )
                ),
                id="together mistralai/Mixtral-8x7B-Instruct-v0.1",
            ),
            pytest.param(
                {
                    "api_type": "responses",
                    "model": "o3",
                    "api_key": "fake_api_key",
                    "max_tokens": 512,
                    "stream": False,
                },
                LLMConfig(
                    OpenAIResponsesLLMConfigEntry(
                        model="o3",
                        api_key="fake_api_key",
                        max_tokens=512,
                    )
                ),
                id="openaoi responses o3",
            ),
        ],
    )
    def test_init_with_entities(self, llm_config: dict[str, Any], expected: LLMConfig) -> None:
        assert LLMConfig(llm_config) == expected

    @pytest.mark.parametrize(
        (
            "llm_config",
            "expected",
        ),
        [
            pytest.param(
                {
                    "model": "o3",
                    "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                },
                LLMConfig(
                    OpenAILLMConfigEntry(
                        model="o3",
                        api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    )
                ),
                id="config from dict",
            ),
            pytest.param(
                OpenAILLMConfigEntry(
                    model="o3",
                    api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                ),
                LLMConfig(
                    OpenAILLMConfigEntry(
                        model="o3",
                        api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    )
                ),
                id="config from entry",
            ),
            pytest.param(
                OpenAILLMConfigEntry(
                    model="o3",
                    api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                ),
                LLMConfig(
                    OpenAILLMConfigEntry(
                        model="o3",
                        api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    )
                ),
                id="config from config",
            ),
            pytest.param(
                {
                    "config_list": {
                        "model": "o3",
                        "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    }
                },
                LLMConfig(
                    OpenAILLMConfigEntry(
                        model="o3",
                        api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    )
                ),
                id="config from config_list",
                marks=pytest.mark.filterwarnings("ignore::DeprecationWarning"),
            ),
            pytest.param(
                [
                    {
                        "model": "o3",
                        "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    },
                    OpenAILLMConfigEntry(
                        model="gpt-4",
                        api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    ),
                ],
                LLMConfig(
                    OpenAILLMConfigEntry(
                        model="o3",
                        api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    ),
                    OpenAILLMConfigEntry(
                        model="gpt-4",
                        api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    ),
                ),
                id="config from list",
            ),
        ],
    )
    def test_ensure_config(self, llm_config: Any, expected: LLMConfig) -> None:
        assert LLMConfig.ensure_config(llm_config) == expected

    def test_serialization(self, openai_llm_config: LLMConfig) -> None:
        actual = openai_llm_config.model_dump()
        expected = {
            "config_list": [
                {
                    "api_type": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    "tags": [],
                    "temperature": 0.5,
                    "stream": False,
                }
            ],
            "temperature": 0.5,
            "check_every_ms": 1000,
            "cache_seed": 42,
        }
        assert actual == expected

    def test_get(self, openai_llm_config: LLMConfig) -> None:
        assert openai_llm_config.get("temperature") == 0.5
        assert openai_llm_config.get("check_every_ms") == 1000
        assert openai_llm_config.get("cache_seed") == 42
        assert openai_llm_config.get("doesnt_exists") is None
        assert openai_llm_config.get("doesnt_exists", "default") == "default"

    def test_getattr(self, openai_llm_config: LLMConfig) -> None:
        assert openai_llm_config.temperature == 0.5
        assert openai_llm_config.check_every_ms == 1000
        assert openai_llm_config.cache_seed == 42
        assert openai_llm_config.config_list == [openai_llm_config.config_list[0]]
        with pytest.raises(AttributeError) as e:
            openai_llm_config.wrong_key
        assert str(e.value) == "'LLMConfig' object has no attribute 'wrong_key'"

    def test_setattr(self, openai_llm_config: LLMConfig) -> None:
        assert openai_llm_config.temperature == 0.5
        openai_llm_config.temperature = 0.8
        assert openai_llm_config.temperature == 0.8

    def test_get_item_and_set_item(self, openai_llm_config: LLMConfig) -> None:
        # Test __getitem__
        assert openai_llm_config["temperature"] == 0.5
        assert openai_llm_config["check_every_ms"] == 1000
        assert openai_llm_config["cache_seed"] == 42
        assert openai_llm_config["config_list"] == [openai_llm_config.config_list[0]]
        with pytest.raises(KeyError) as e:
            openai_llm_config["wrong_key"]
        assert str(e.value) == "\"Key 'wrong_key' not found in LLMConfig\""

        # Test __setitem__
        assert openai_llm_config["timeout"] is None
        openai_llm_config["timeout"] = 60
        assert openai_llm_config["timeout"] == 60
        openai_llm_config["timeout"] = None
        assert openai_llm_config["timeout"] is None

    def test_items(self, openai_llm_config: LLMConfig) -> None:
        actual = openai_llm_config.items()  # type: ignore[var-annotated]
        assert isinstance(actual, dict_items)
        expected = {
            "config_list": [
                {
                    "api_type": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    "tags": [],
                    "temperature": 0.5,
                    "stream": False,
                }
            ],
            "temperature": 0.5,
            "check_every_ms": 1000,
            "cache_seed": 42,
        }
        assert dict(actual) == expected, dict(actual)

    def test_keys(self, openai_llm_config: LLMConfig) -> None:
        actual = openai_llm_config.keys()  # type: ignore[var-annotated]
        assert isinstance(actual, dict_keys)
        expected = ["temperature", "check_every_ms", "cache_seed", "config_list"]
        assert list(actual) == expected, list(actual)

    def test_values(self, openai_llm_config: LLMConfig) -> None:
        actual = openai_llm_config.values()  # type: ignore[var-annotated]
        assert isinstance(actual, dict_values)
        expected = [
            0.5,
            1000,
            42,
            # Remove ["**"] - allowed_paths is None and excluded
            [
                {
                    "api_type": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    "tags": [],
                    "stream": False,
                    "temperature": 0.5,
                }
            ],
        ]
        assert list(actual) == expected, list(actual)

    def test_unpack(self, openai_llm_config: LLMConfig, openai_llm_config_entry: OpenAILLMConfigEntry) -> None:
        openai_llm_config_entry.base_url = "localhost:8080"  # type: ignore[assignment]
        openai_llm_config.config_list = [  # type: ignore[attr-defined]
            openai_llm_config_entry,
        ]
        expected = {
            "config_list": [
                {
                    "api_type": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    "base_url": "localhost:8080",
                    "stream": False,
                    "tags": [],
                }
            ],
            "temperature": 0.5,
            "check_every_ms": 1000,
            "cache_seed": 42,
        }

        def test_unpacking(**kwargs: Any) -> None:
            for k, v in expected.items():
                assert k in kwargs
                if k == "config_list":
                    assert kwargs[k][0].model_dump() == v[0]  # type: ignore[index]
                else:
                    assert kwargs[k] == v
            # assert kwargs == expected, kwargs

        test_unpacking(**openai_llm_config)

    def test_contains(self, openai_llm_config: LLMConfig) -> None:
        assert "temperature" in openai_llm_config
        assert "check_every_ms" in openai_llm_config
        assert "cache_seed" in openai_llm_config
        assert "config_list" in openai_llm_config
        assert "doesnt_exists" not in openai_llm_config
        assert "config_list" in openai_llm_config
        assert not "config_list" not in openai_llm_config

    @pytest.mark.parametrize(
        ("filter_dict, exclude, expected"),
        [
            (
                {"tags": ["gpt35", "gpt4"]},
                False,
                JSON_SAMPLE_DICT[0:2],
            ),
            (
                {"tags": ["gpt35", "gpt4"]},
                True,
                JSON_SAMPLE_DICT[2:4],
            ),
            (
                {"api_type": "azure", "api_version": "2024-02-01"},
                False,
                [JSON_SAMPLE_DICT[2]],
            ),
            (
                {"api_type": ["azure"]},
                False,
                [JSON_SAMPLE_DICT[2]],
            ),
            (
                {},
                False,
                JSON_SAMPLE_DICT,
            ),
        ],
    )
    def test_where(self, filter_dict: dict[str, Any], exclude: bool, expected: list[dict[str, Any]]) -> None:
        openai_llm_config = LLMConfig(*JSON_SAMPLE_DICT)

        actual = openai_llm_config.where(**filter_dict, exclude=exclude)

        assert actual == LLMConfig(*expected)

    def test_where_invalid_filter(self) -> None:
        openai_llm_config = LLMConfig(*JSON_SAMPLE_DICT)

        with pytest.raises(ValueError) as e:
            openai_llm_config.where(api_type="invalid")
        assert str(e.value) == "No config found that satisfies the filter criteria: {'api_type': 'invalid'}"

    def test_repr(self, openai_llm_config_entry: OpenAILLMConfigEntry) -> None:
        # Case 1: routing_method is None (default)
        config_default_routing = LLMConfig(openai_llm_config_entry)
        actual_repr_default = repr(config_default_routing)
        assert config_default_routing.routing_method is None
        assert "routing_method" not in actual_repr_default

        # Check for key components of the config_list item's dictionary representation
        assert "config_list=[{" in actual_repr_default
        assert f"'api_type': '{openai_llm_config_entry.api_type}'" in actual_repr_default
        assert f"'model': '{openai_llm_config_entry.model}'" in actual_repr_default
        assert "'api_key': '**********'" in actual_repr_default  # Redacted
        assert f"'tags': {openai_llm_config_entry.tags!r}" in actual_repr_default
        if openai_llm_config_entry.base_url:  # Should not be present if None due to exclude_none
            assert f"'base_url': '{str(openai_llm_config_entry.base_url)}'" in actual_repr_default
        else:
            assert "'base_url'" not in actual_repr_default  # Ensure it's omitted

        # Case 2: routing_method is explicitly set
        config_custom_routing = LLMConfig(openai_llm_config_entry, routing_method="round_robin", temperature=0.77)
        actual_repr_custom = repr(config_custom_routing)
        assert config_custom_routing.routing_method == "round_robin"
        assert "routing_method='round_robin'" in actual_repr_custom
        assert "config_list=[{" in actual_repr_custom  # Basic structure check
        assert "'api_key': '**********'" in actual_repr_custom  # Redacted
        assert "temperature=0.77" in actual_repr_custom

    def test_str(self, openai_llm_config_entry: OpenAILLMConfigEntry) -> None:
        # str calls repr, so logic is similar
        # Case 1: routing_method is None (default)
        config_default_routing = LLMConfig(openai_llm_config_entry)
        actual_str_default = str(config_default_routing)
        assert config_default_routing.routing_method is None
        assert "routing_method" not in actual_str_default
        assert "config_list=[{" in actual_str_default
        assert f"'api_type': '{openai_llm_config_entry.api_type}'" in actual_str_default
        assert f"'model': '{openai_llm_config_entry.model}'" in actual_str_default
        assert "'api_key': '**********'" in actual_str_default  # Redacted
        assert f"'tags': {openai_llm_config_entry.tags!r}" in actual_str_default
        if openai_llm_config_entry.base_url:
            assert f"'base_url': '{str(openai_llm_config_entry.base_url)}'" in actual_str_default
        else:
            assert "'base_url'" not in actual_str_default

        # Case 2: routing_method is explicitly set
        config_custom_routing = LLMConfig(openai_llm_config_entry, routing_method="round_robin", temperature=0.77)
        actual_str_custom = str(config_custom_routing)
        assert config_custom_routing.routing_method == "round_robin"
        assert "routing_method='round_robin'" in actual_str_custom
        assert "config_list=[{" in actual_str_custom
        assert "'api_key': '**********'" in actual_str_custom  # Redacted
        assert "temperature=0.77" in actual_str_custom

    def test_routing_method_default(self, openai_llm_config_entry: OpenAILLMConfigEntry) -> None:
        llm_config = LLMConfig(openai_llm_config_entry)
        assert llm_config.routing_method is None

    def test_routing_method_custom(self, openai_llm_config_entry: OpenAILLMConfigEntry) -> None:
        llm_config = LLMConfig(openai_llm_config_entry, routing_method="round_robin")
        assert llm_config.routing_method == "round_robin"

    def test_routing_method_invalid(self, openai_llm_config_entry: OpenAILLMConfigEntry) -> None:
        with pytest.raises(ValidationError):  # Changed from ValueError to ValidationError
            LLMConfig(openai_llm_config_entry, routing_method="invalid_method")  # type: ignore[arg-type]

    def test_from_json_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_CONFIG", JSON_SAMPLE)
        expected = LLMConfig(*JSON_SAMPLE_DICT)
        actual = LLMConfig.from_json(env="LLM_CONFIG")
        assert isinstance(actual, LLMConfig)
        assert actual == expected, actual

    @pytest.mark.xfail(reason="Currently raises FileNotFoundError")
    def test_from_json_env_not_found(self) -> None:
        with pytest.raises(ValueError) as e:
            LLMConfig.from_json(env="INVALID_ENV")
        assert str(e.value) == "Environment variable 'INVALID_ENV' not found"

    def test_from_json_env_with_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_CONFIG", JSON_SAMPLE)
        expected = LLMConfig(*JSON_SAMPLE_DICT, temperature=0.5, check_every_ms=1000, cache_seed=42)
        actual = LLMConfig.from_json(env="LLM_CONFIG", temperature=0.5, check_every_ms=1000, cache_seed=42)
        assert isinstance(actual, LLMConfig)
        assert actual == expected, actual

    def test_from_json_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = f"{tmpdirname}/llm_config.json"
            with open(file_path, "w") as f:
                f.write(JSON_SAMPLE)

            expected = LLMConfig(*JSON_SAMPLE_DICT)
            actual = LLMConfig.from_json(path=file_path)
            assert isinstance(actual, LLMConfig)
            assert actual == expected, actual

        with pytest.raises(FileNotFoundError) as e:
            LLMConfig.from_json(path="invalid_path")
        assert "No such file or directory: 'invalid_path'" in str(e.value)

    def test_copy(self, openai_llm_config: LLMConfig) -> None:
        actual = openai_llm_config.copy()
        assert actual == openai_llm_config
        assert actual is not openai_llm_config

        actual = openai_llm_config.deepcopy()
        assert actual == openai_llm_config
        assert actual is not openai_llm_config

        actual = copy(openai_llm_config)
        assert actual == openai_llm_config
        assert actual is not openai_llm_config

        actual = deepcopy(openai_llm_config)
        assert actual == openai_llm_config
        assert actual is not openai_llm_config

    def test_llm_config_doesnt_patch_entry(self) -> None:
        entry = OpenAILLMConfigEntry(top_p=0.5, model="o3", extra="extra")

        # test entry combination
        assert LLMConfig(entry, max_tokens=1024).config_list == [
            OpenAILLMConfigEntry(top_p=0.5, model="o3", extra="extra", max_tokens=1024)
        ]

        assert entry.max_tokens is None

    def test_llm_config_doesnt_patch_entry_dict(self) -> None:
        entry = {"top_p": 0.5, "model": "o3", "extra": "extra"}

        # test entry combination
        assert LLMConfig(entry, max_tokens=1024).config_list == [
            OpenAILLMConfigEntry(top_p=0.5, model="o3", extra="extra", max_tokens=1024)
        ]

        assert entry == {"top_p": 0.5, "model": "o3", "extra": "extra"}

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_with_context(self, openai_llm_config: LLMConfig) -> None:
        # Test with dummy agent
        class DummyAgent:
            def __init__(self) -> None:
                self.llm_config = LLMConfig.get_current_llm_config()

        with openai_llm_config:
            agent = DummyAgent()
        assert agent.llm_config == openai_llm_config
        assert agent.llm_config.temperature == 0.5
        assert agent.llm_config.config_list[0]["model"] == "gpt-4o-mini"

        # Test passing LLMConfig object as parameter
        assert LLMConfig.get_current_llm_config(openai_llm_config) == openai_llm_config

        # Test accessing current_llm_config outside the context
        assert LLMConfig.get_current_llm_config() is None
        with openai_llm_config:
            actual = LLMConfig.get_current_llm_config()
            assert actual == openai_llm_config

        assert LLMConfig.get_current_llm_config() is None

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_current(self) -> None:
        llm_config = LLMConfig(*JSON_SAMPLE_DICT)

        # Test without context. Should raise an error
        expected_error = "No current LLMConfig set. Are you inside a context block?"
        with pytest.raises(ValueError) as e:
            LLMConfig.current
        assert str(e.value) == expected_error
        with pytest.raises(ValueError) as e:
            LLMConfig.default
        assert str(e.value) == expected_error

        with llm_config:
            assert LLMConfig.get_current_llm_config() == llm_config
            assert LLMConfig.current == llm_config
            assert LLMConfig.default == llm_config

            with LLMConfig.current.where(api_type="openai"):
                assert LLMConfig.get_current_llm_config() == llm_config.where(api_type="openai")
                assert LLMConfig.current == llm_config.where(api_type="openai")
                assert LLMConfig.default == llm_config.where(api_type="openai")

                with LLMConfig.default.where(model="gpt-4"):
                    assert LLMConfig.get_current_llm_config() == llm_config.where(api_type="openai", model="gpt-4")
                    assert LLMConfig.current == llm_config.where(api_type="openai", model="gpt-4")
                    assert LLMConfig.default == llm_config.where(api_type="openai", model="gpt-4")

    def test_openai_llm_config_entry_with_workspace_dir(
        self, openai_llm_config_entry: OpenAIResponsesLLMConfigEntry
    ) -> None:
        """Test OpenAIResponsesLLMConfigEntry with workspace_dir parameter."""
        entry = OpenAIResponsesLLMConfigEntry(
            model="gpt-4o-mini",
            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
            workspace_dir="./my_project",
        )
        assert entry.workspace_dir == "./my_project"
        assert entry.model_dump()["workspace_dir"] == "./my_project"

    def test_openai_llm_config_entry_with_allowed_paths(
        self, openai_llm_config_entry: OpenAIResponsesLLMConfigEntry
    ) -> None:
        """Test OpenAIResponsesLLMConfigEntry with allowed_paths parameter."""
        entry = OpenAIResponsesLLMConfigEntry(
            model="gpt-4o-mini",
            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
            allowed_paths=["src/**", "tests/**"],
        )
        assert entry.allowed_paths == ["src/**", "tests/**"]
        assert entry.model_dump()["allowed_paths"] == ["src/**", "tests/**"]

    def test_openai_llm_config_entry_workspace_dir_and_allowed_paths(self) -> None:
        """Test OpenAIResponsesLLMConfigEntry with both workspace_dir and allowed_paths."""
        entry = OpenAIResponsesLLMConfigEntry(
            model="gpt-4o-mini",
            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
            workspace_dir="./project",
            allowed_paths=["src/**", "*.py"],
        )
        assert entry.workspace_dir == "./project"
        assert entry.allowed_paths == ["src/**", "*.py"]

        # Test that they're included in model_dump
        dumped = entry.model_dump()
        assert dumped["workspace_dir"] == "./project"
        assert dumped["allowed_paths"] == ["src/**", "*.py"]

    def test_openai_llm_config_entry_defaults_workspace_dir_and_allowed_paths(self) -> None:
        """Test that workspace_dir and allowed_paths default to None."""
        entry = OpenAIResponsesLLMConfigEntry(
            model="gpt-4o-mini",
            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
        )
        assert entry.workspace_dir is None
        assert entry.allowed_paths is None

        # Test that None values are excluded from model_dump with exclude_none=True
        dumped = entry.model_dump(exclude_none=True)
        assert "workspace_dir" not in dumped
        assert "allowed_paths" not in dumped

    def test_llm_config_with_entry_workspace_dir_and_allowed_paths(self) -> None:
        """Test LLMConfig with OpenAIResponsesLLMConfigEntry containing workspace_dir and allowed_paths."""
        entry = OpenAIResponsesLLMConfigEntry(
            model="gpt-4o-mini",
            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
            workspace_dir="./project",
            allowed_paths=["src/**"],
        )
        config = LLMConfig(entry)

        # Verify they're in the config entry, not at LLMConfig level
        assert config.config_list[0].workspace_dir == "./project"  # type: ignore[union-attr]
        assert config.config_list[0].allowed_paths == ["src/**"]  # type: ignore[union-attr]

        # Verify LLMConfig doesn't have these attributes
        assert not hasattr(config, "workspace_dir")
        assert not hasattr(config, "allowed_paths")

    def test_llm_config_copy_preserves_entry_workspace_dir(self) -> None:
        """Test that copy() preserves workspace_dir in config entry."""
        entry = OpenAILLMConfigEntry(
            model="gpt-4o-mini",
            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
            workspace_dir="./project",
        )
        config = LLMConfig(entry)
        copied = config.copy()

        # Verify workspace_dir is preserved in the copied config entry
        assert copied.config_list[0].workspace_dir == "./project"  # type: ignore[union-attr]
        assert copied.config_list[0].workspace_dir == config.config_list[0].workspace_dir  # type: ignore[union-attr]

    def test_llm_config_dict_with_workspace_dir_and_allowed_paths(self) -> None:
        """Test LLMConfig with dict config containing workspace_dir and allowed_paths."""
        config = LLMConfig({
            "api_type": "openai",
            "model": "gpt-4o-mini",
            "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
            "workspace_dir": "./project",
            "allowed_paths": ["src/**", "*.py"],
        })

        # Verify they're in the config entry
        assert config.config_list[0].workspace_dir == "./project"  # type: ignore[union-attr]
        assert config.config_list[0].allowed_paths == ["src/**", "*.py"]  # type: ignore[union-attr]

        # Verify LLMConfig doesn't have these attributes
        assert not hasattr(config, "workspace_dir")
        assert not hasattr(config, "allowed_paths")
