# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from _collections_abc import dict_items, dict_keys, dict_values
from typing import Any

import pytest

from autogen.llm_config import LLMConfig
from autogen.oai.anthropic import AnthropicLLMConfigEntry
from autogen.oai.bedrock import BedrockLLMConfigEntry
from autogen.oai.cerebras import CerebrasLLMConfigEntry
from autogen.oai.client import AzureOpenAILLMConfigEntry, DeepSeekLLMConfigEntry, OpenAILLMConfigEntry
from autogen.oai.cohere import CohereLLMConfigEntry
from autogen.oai.gemini import GeminiLLMConfigEntry
from autogen.oai.groq import GroqLLMConfigEntry
from autogen.oai.mistral import MistralLLMConfigEntry
from autogen.oai.ollama import OllamaLLMConfigEntry
from autogen.oai.together import TogetherLLMConfigEntry

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
        with pytest.raises(ValueError) as e:
            # Intentionally passing extra field to raise an error
            OpenAILLMConfigEntry(  # type: ignore [call-arg]
                model="gpt-4o-mini", api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly", extra="extra"
            )
        assert "Extra inputs are not permitted [type=extra_forbidden, input_value='extra', input_type=str]" in str(
            e.value
        )

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
        return LLMConfig(config_list=[openai_llm_config_entry], temperature=0.5, check_every_ms=1000, cache_seed=42)

    @pytest.mark.parametrize(
        "llm_config, expected",
        [
            (
                # todo add more test cases
                {
                    "config_list": [
                        {"model": "gpt-4o-mini", "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly"}
                    ]
                },
                LLMConfig(
                    config_list=[
                        OpenAILLMConfigEntry(
                            model="gpt-4o-mini",
                            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                        )
                    ]
                ),
            ),
            (
                {"model": "gpt-4o-mini", "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly"},
                LLMConfig(
                    config_list=[
                        OpenAILLMConfigEntry(
                            model="gpt-4o-mini",
                            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                        )
                    ]
                ),
            ),
            (
                {
                    "model": "gpt-4o-mini",
                    "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    "cache_seed": 42,
                },
                LLMConfig(
                    config_list=[
                        OpenAILLMConfigEntry(
                            model="gpt-4o-mini",
                            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                        )
                    ],
                    cache_seed=42,
                ),
            ),
            (
                {
                    "config_list": [
                        {"model": "gpt-4o-mini", "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly"}
                    ],
                    "max_tokens": 1024,
                },
                LLMConfig(
                    config_list=[
                        OpenAILLMConfigEntry(
                            model="gpt-4o-mini",
                            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                            max_tokens=1024,
                        )
                    ]
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "model": "gpt-4o-mini",
                            "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                            "api_type": "openai",
                        }
                    ],
                    "temperature": 0.5,
                    "check_every_ms": 1000,
                    "cache_seed": 42,
                },
                LLMConfig(
                    config_list=[
                        OpenAILLMConfigEntry(
                            model="gpt-4o-mini",
                            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                        )
                    ],
                    temperature=0.5,
                    check_every_ms=1000,
                    cache_seed=42,
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "model": "gpt-4o-mini",
                            "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                            "api_type": "azure",
                            "base_url": "https://api.openai.com",
                        }
                    ],
                },
                LLMConfig(
                    config_list=[
                        AzureOpenAILLMConfigEntry(
                            model="gpt-4o-mini",
                            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                            base_url="https://api.openai.com",
                        )
                    ],
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "anthropic",
                            "model": "claude-3-5-sonnet-latest",
                            "api_key": "dummy_api_key",
                            "stream": False,
                            "temperature": 1.0,
                            "top_p": 0.8,
                            "max_tokens": 100,
                            "tags": [],
                        }
                    ]
                },
                LLMConfig(
                    config_list=[
                        AnthropicLLMConfigEntry(
                            model="claude-3-5-sonnet-latest",
                            api_key="dummy_api_key",
                            stream=False,
                            temperature=1.0,
                            top_p=0.8,
                            max_tokens=100,
                        )
                    ],
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "bedrock",
                            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                            "aws_region": "us-east-1",
                            "aws_access_key": "test_access_key_id",
                            "aws_secret_key": "test_secret_access_key",
                            "aws_session_token": "test_session_token",
                            "temperature": 0.8,
                            "topP": 0.6,
                            "stream": False,
                            "tags": [],
                            "supports_system_prompts": True,
                        }
                    ]
                },
                LLMConfig(
                    config_list=[
                        BedrockLLMConfigEntry(
                            model="anthropic.claude-3-sonnet-20240229-v1:0",
                            aws_region="us-east-1",
                            aws_access_key="test_access_key_id",
                            aws_secret_key="test_secret_access_key",
                            aws_session_token="test_session_token",
                            temperature=0.8,
                            topP=0.6,
                            stream=False,
                        )
                    ]
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "cerebras",
                            "api_key": "fake_api_key",
                            "model": "llama3.1-8b",
                            "max_tokens": 1000,
                            "seed": 42,
                            "stream": False,
                            "temperature": 1.0,
                            "tags": [],
                        }
                    ]
                },
                LLMConfig(
                    config_list=[
                        CerebrasLLMConfigEntry(
                            api_key="fake_api_key",
                            model="llama3.1-8b",
                            max_tokens=1000,
                            seed=42,
                            stream=False,
                            temperature=1,
                        )
                    ]
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "cohere",
                            "model": "command-r-plus",
                            "api_key": "dummy_api_key",
                            "frequency_penalty": 0,
                            "k": 0,
                            "p": 0.75,
                            "presence_penalty": 0,
                            "strict_tools": False,
                            "tags": [],
                            "temperature": 0.3,
                        }
                    ]
                },
                LLMConfig(
                    config_list=[
                        CohereLLMConfigEntry(
                            model="command-r-plus",
                            api_key="dummy_api_key",
                            stream=False,
                        )
                    ]
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "deepseek",
                            "api_key": "fake_api_key",
                            "model": "deepseek-chat",
                            "base_url": "https://api.deepseek.com/v1",
                            "max_tokens": 8192,
                            "temperature": 0.5,
                            "tags": [],
                        }
                    ]
                },
                LLMConfig(
                    config_list=[
                        DeepSeekLLMConfigEntry(
                            api_key="fake_api_key",
                            model="deepseek-chat",
                        )
                    ]
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "google",
                            "model": "gemini-2.0-flash-lite",
                            "api_key": "dummy_api_key",
                            "project_id": "fake-project-id",
                            "location": "us-west1",
                            "stream": False,
                            "tags": [],
                        }
                    ]
                },
                LLMConfig(
                    config_list=[
                        GeminiLLMConfigEntry(
                            model="gemini-2.0-flash-lite",
                            api_key="dummy_api_key",
                            project_id="fake-project-id",
                            location="us-west1",
                        )
                    ]
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "groq",
                            "model": "llama3-8b-8192",
                            "api_key": "fake_api_key",
                            "temperature": 1,
                            "stream": False,
                            "tags": [],
                        }
                    ]
                },
                LLMConfig(config_list=[GroqLLMConfigEntry(api_key="fake_api_key", model="llama3-8b-8192")]),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "mistral",
                            "model": "mistral-small-latest",
                            "api_key": "fake_api_key",
                            "safe_prompt": False,
                            "stream": False,
                            "temperature": 0.7,
                            "tags": [],
                        }
                    ]
                },
                LLMConfig(
                    config_list=[
                        MistralLLMConfigEntry(
                            model="mistral-small-latest",
                            api_key="fake_api_key",
                        )
                    ]
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "ollama",
                            "model": "llama3.1:8b",
                            "num_ctx": 2048,
                            "num_predict": 128,
                            "repeat_penalty": 1.1,
                            "seed": 42,
                            "stream": False,
                            "tags": [],
                            "temperature": 0.8,
                            "top_k": 40,
                            "top_p": 0.9,
                        }
                    ]
                },
                LLMConfig(config_list=[OllamaLLMConfigEntry(model="llama3.1:8b")]),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "together",
                            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                            "api_key": "fake_api_key",
                            "safety_model": "Meta-Llama/Llama-Guard-7b",
                            "tags": [],
                            "max_tokens": 512,
                            "stream": False,
                        }
                    ]
                },
                LLMConfig(
                    config_list=[
                        TogetherLLMConfigEntry(
                            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                            api_key="fake_api_key",
                            safety_model="Meta-Llama/Llama-Guard-7b",
                        )
                    ]
                ),
            ),
        ],
    )
    def test_init(self, llm_config: dict[str, Any], expected: LLMConfig) -> None:
        actual = LLMConfig(**llm_config)
        assert actual == expected, actual

    def test_extra_fields(self) -> None:
        with pytest.raises(ValueError) as e:
            LLMConfig(
                config_list=[
                    OpenAILLMConfigEntry(
                        model="gpt-4o-mini", api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly"
                    )
                ],
                extra="extra",
            )
        assert "Extra inputs are not permitted [type=extra_forbidden, input_value='extra', input_type=str]" in str(
            e.value
        )

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
        assert actual.model_dump() == openai_llm_config.model_dump()
        assert type(actual._model) == type(openai_llm_config._model)
        assert actual._model == openai_llm_config._model
        assert actual == openai_llm_config
        assert isinstance(actual.config_list[0], OpenAILLMConfigEntry)

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
            [
                {
                    "api_type": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    "tags": [],
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

        # Test accessing current_llm_config outside the context
        assert LLMConfig.get_current_llm_config() is None
        with openai_llm_config:
            actual = LLMConfig.get_current_llm_config()
            assert actual == openai_llm_config

        assert LLMConfig.get_current_llm_config() is None

    @pytest.mark.parametrize(
        "filter_dict, exclude, expected",
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
        openai_llm_config = LLMConfig(config_list=JSON_SAMPLE_DICT)

        actual = openai_llm_config.where(**filter_dict, exclude=exclude)
        assert isinstance(actual, LLMConfig)
        assert actual.config_list == LLMConfig(config_list=expected).config_list

    def test_where_invalid_filter(self) -> None:
        openai_llm_config = LLMConfig(config_list=JSON_SAMPLE_DICT)

        with pytest.raises(ValueError) as e:
            openai_llm_config.where(api_type="invalid")
        assert str(e.value) == "No config found that satisfies the filter criteria: {'api_type': 'invalid'}"

    def test_repr(self, openai_llm_config: LLMConfig) -> None:
        actual = repr(openai_llm_config)
        expected = "LLMConfig(temperature=0.5, check_every_ms=1000, cache_seed=42, config_list=[{'api_type': 'openai', 'model': 'gpt-4o-mini', 'api_key': '**********', 'tags': []}])"
        assert actual == expected, actual

    def test_str(self, openai_llm_config: LLMConfig) -> None:
        actual = str(openai_llm_config)
        expected = "LLMConfig(temperature=0.5, check_every_ms=1000, cache_seed=42, config_list=[{'api_type': 'openai', 'model': 'gpt-4o-mini', 'api_key': '**********', 'tags': []}])"
        assert actual == expected, actual

    def test_from_json_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_CONFIG", JSON_SAMPLE)
        expected = LLMConfig(config_list=JSON_SAMPLE_DICT)
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
        expected = LLMConfig(config_list=JSON_SAMPLE_DICT, temperature=0.5, check_every_ms=1000, cache_seed=42)
        actual = LLMConfig.from_json(env="LLM_CONFIG", temperature=0.5, check_every_ms=1000, cache_seed=42)
        assert isinstance(actual, LLMConfig)
        assert actual == expected, actual

    def test_from_json_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = f"{tmpdirname}/llm_config.json"
            with open(file_path, "w") as f:
                f.write(JSON_SAMPLE)

            expected = LLMConfig(config_list=JSON_SAMPLE_DICT)
            actual = LLMConfig.from_json(path=file_path)
            assert isinstance(actual, LLMConfig)
            assert actual == expected, actual

        with pytest.raises(FileNotFoundError) as e:
            LLMConfig.from_json(path="invalid_path")
        assert "No such file or directory: 'invalid_path'" in str(e.value)

    def test_current(self) -> None:
        llm_config = LLMConfig(config_list=JSON_SAMPLE_DICT)

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
