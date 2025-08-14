# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

import copy
import inspect
import os
import shutil
import time
from collections.abc import Generator
from typing import Any  # Added import for Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from autogen import OpenAIWrapper
from autogen.cache.cache import Cache
from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.llm_config import LLMConfig
from autogen.oai.client import (
    AOPENAI_FALLBACK_KWARGS,
    LEGACY_CACHE_DIR,
    LEGACY_DEFAULT_CACHE_SEED,
    OPENAI_FALLBACK_KWARGS,
    AzureOpenAILLMConfigEntry,
    DeepSeekLLMConfigEntry,
    OpenAIClient,
    OpenAILLMConfigEntry,
)
from autogen.oai.oai_models import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage

# Attempt to import APIError from openai, define as base Exception if openai is not available.
try:
    from openai import APIError
except ImportError:
    APIError = Exception


from ..conftest import Credentials


class MockModelClient:
    def __init__(self, config: dict, name: str = "mock_client"):
        self.config = config
        self.name = name  # Store the name if provided in config, else use default
        self.call_count = 0

    def create(self, params: dict[str, Any]):
        self.call_count += 1
        # Simulate a successful response or raise an exception based on config
        if self.config.get("should_fail", False):
            raise APIError(
                message="Mock API Error", request=None, body=None
            )  # Use openai.APIError or a general Exception

        client_name_to_respond = self.config.get("name", self.name)
        # Simulate a ChatCompletion response
        return ChatCompletion(
            id="chatcmpl-test",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(content=f"Response from {client_name_to_respond}", role="assistant"),
                )
            ],
            created=1677652288,
            model=params.get("model", "gpt-3.5-turbo"),
            object="chat.completion",
            usage=CompletionUsage(completion_tokens=10, prompt_tokens=10, total_tokens=20),
        )

    def message_retrieval(self, response):
        return [choice.message.content for choice in response.choices]

    def cost(self, response):
        return 0.02  # Example cost

    @staticmethod
    def get_usage(response):
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "cost": response.cost if hasattr(response, "cost") else 0,
            "model": response.model,
        }


# Fixture for OpenAIWrapper with mocked clients
@pytest.fixture
def mock_openai_wrapper_fixed_order_default():
    # Test case where routing_method is not specified in OpenAIWrapper constructor,
    # so it should default to "fixed_order".
    config_list = [
        {"model": "gpt-3.5-turbo", "api_key": "key1", "model_client_cls": "MockModelClient", "name": "client1"},
        {"model": "gpt-4", "api_key": "key2", "model_client_cls": "MockModelClient", "name": "client2"},
    ]
    wrapper = OpenAIWrapper(config_list=config_list)
    assert wrapper.routing_method == "fixed_order"

    for i in range(len(config_list)):
        wrapper._clients[i] = MockModelClient(config=wrapper._config_list[i])
    return wrapper


@pytest.fixture
def mock_openai_wrapper_fixed_order_explicit():
    # Test case where routing_method IS specified as "fixed_order" in OpenAIWrapper constructor.
    config_list = [
        {"model": "gpt-3.5-turbo", "api_key": "key1", "model_client_cls": "MockModelClient", "name": "client1"},
        {"model": "gpt-4", "api_key": "key2", "model_client_cls": "MockModelClient", "name": "client2"},
    ]
    wrapper = OpenAIWrapper(config_list=config_list, routing_method="fixed_order")
    assert wrapper.routing_method == "fixed_order"
    for i in range(len(config_list)):
        wrapper._clients[i] = MockModelClient(config=wrapper._config_list[i])
    return wrapper


@pytest.fixture
def mock_openai_wrapper_round_robin():
    config_list = [
        {"model": "gpt-3.5-turbo", "api_key": "key1", "model_client_cls": "MockModelClient", "name": "client1"},
        {"model": "gpt-4", "api_key": "key2", "model_client_cls": "MockModelClient", "name": "client2"},
        {"model": "gpt-4o", "api_key": "key3", "model_client_cls": "MockModelClient", "name": "client3"},
    ]
    wrapper = OpenAIWrapper(config_list=config_list, routing_method="round_robin")
    assert wrapper.routing_method == "round_robin"
    for i in range(len(config_list)):
        wrapper._clients[i] = MockModelClient(config=wrapper._config_list[i])
    return wrapper


@pytest.mark.parametrize(
    "fixture_name", ["mock_openai_wrapper_fixed_order_default", "mock_openai_wrapper_fixed_order_explicit"]
)
def test_fixed_order_routing_successful_first_client(fixture_name: str, request: pytest.FixtureRequest):
    wrapper = request.getfixturevalue(fixture_name)
    response = wrapper.create(messages=[{"role": "user", "content": "Hello"}])
    assert "Response from client1" in response.choices[0].message.content
    assert wrapper._clients[0].call_count == 1
    assert wrapper._clients[1].call_count == 0


def test_round_robin_routing(mock_openai_wrapper_round_robin: OpenAIWrapper):
    # First call
    response1 = mock_openai_wrapper_round_robin.create(messages=[{"role": "user", "content": "Hello 1"}])
    assert "Response from client1" in response1.choices[0].message.content
    assert mock_openai_wrapper_round_robin._clients[0].call_count == 1
    assert mock_openai_wrapper_round_robin._clients[1].call_count == 0
    assert mock_openai_wrapper_round_robin._clients[2].call_count == 0
    assert mock_openai_wrapper_round_robin._round_robin_index == 1

    # Second call
    response2 = mock_openai_wrapper_round_robin.create(messages=[{"role": "user", "content": "Hello 2"}])
    assert "Response from client2" in response2.choices[0].message.content
    assert mock_openai_wrapper_round_robin._clients[0].call_count == 1
    assert mock_openai_wrapper_round_robin._clients[1].call_count == 1
    assert mock_openai_wrapper_round_robin._clients[2].call_count == 0
    assert mock_openai_wrapper_round_robin._round_robin_index == 2

    # Third call
    response3 = mock_openai_wrapper_round_robin.create(messages=[{"role": "user", "content": "Hello 3"}])
    assert "Response from client3" in response3.choices[0].message.content
    assert mock_openai_wrapper_round_robin._clients[0].call_count == 1
    assert mock_openai_wrapper_round_robin._clients[1].call_count == 1
    assert mock_openai_wrapper_round_robin._clients[2].call_count == 1
    assert mock_openai_wrapper_round_robin._round_robin_index == 0

    # Fourth call (wraps around)
    response4 = mock_openai_wrapper_round_robin.create(messages=[{"role": "user", "content": "Hello 4"}])
    assert "Response from client1" in response4.choices[0].message.content
    assert mock_openai_wrapper_round_robin._clients[0].call_count == 2
    assert mock_openai_wrapper_round_robin._clients[1].call_count == 1
    assert mock_openai_wrapper_round_robin._clients[2].call_count == 1
    assert mock_openai_wrapper_round_robin._round_robin_index == 1


def test_round_robin_routing_with_failures(mock_openai_wrapper_round_robin: OpenAIWrapper):
    # Make client2 fail
    mock_openai_wrapper_round_robin._clients[1].config["should_fail"] = True

    # First call (client1)
    response1 = mock_openai_wrapper_round_robin.create(messages=[{"role": "user", "content": "Hello 1"}])
    assert "Response from client1" in response1.choices[0].message.content
    assert mock_openai_wrapper_round_robin._clients[0].call_count == 1
    assert mock_openai_wrapper_round_robin._clients[1].call_count == 0
    assert mock_openai_wrapper_round_robin._clients[2].call_count == 0
    assert mock_openai_wrapper_round_robin._round_robin_index == 1

    # Second call (client2 fails, client3 should be called)
    response2 = mock_openai_wrapper_round_robin.create(messages=[{"role": "user", "content": "Hello 2"}])
    assert "Response from client3" in response2.choices[0].message.content
    assert mock_openai_wrapper_round_robin._clients[0].call_count == 1  # Not called again
    assert mock_openai_wrapper_round_robin._clients[1].call_count == 1  # Called and failed
    assert mock_openai_wrapper_round_robin._clients[2].call_count == 1  # Called
    assert mock_openai_wrapper_round_robin._round_robin_index == 2

    # Third call (client3 is the start of this round)
    # Reset call counts for clarity for this specific call
    client1_prev_calls = mock_openai_wrapper_round_robin._clients[0].call_count
    client2_prev_calls = mock_openai_wrapper_round_robin._clients[1].call_count
    client3_prev_calls = mock_openai_wrapper_round_robin._clients[2].call_count

    response3 = mock_openai_wrapper_round_robin.create(messages=[{"role": "user", "content": "Hello 3"}])
    assert "Response from client3" in response3.choices[0].message.content
    assert mock_openai_wrapper_round_robin._clients[0].call_count == client1_prev_calls
    assert mock_openai_wrapper_round_robin._clients[1].call_count == client2_prev_calls
    assert mock_openai_wrapper_round_robin._clients[2].call_count == client3_prev_calls + 1
    assert mock_openai_wrapper_round_robin._round_robin_index == 0  # Wraps around

    # Fourth call (client1 is the start of this round)
    client1_prev_calls = mock_openai_wrapper_round_robin._clients[0].call_count
    client2_prev_calls = mock_openai_wrapper_round_robin._clients[1].call_count
    client3_prev_calls = mock_openai_wrapper_round_robin._clients[2].call_count
    response4 = mock_openai_wrapper_round_robin.create(messages=[{"role": "user", "content": "Hello 4"}])
    assert "Response from client1" in response4.choices[0].message.content
    assert mock_openai_wrapper_round_robin._clients[0].call_count == client1_prev_calls + 1
    assert mock_openai_wrapper_round_robin._clients[1].call_count == client2_prev_calls
    assert mock_openai_wrapper_round_robin._clients[2].call_count == client3_prev_calls
    assert mock_openai_wrapper_round_robin._round_robin_index == 1


TOOL_ENABLED = False

with optional_import_block() as result:
    import openai
    from openai import AzureOpenAI, OpenAI

    if openai.__version__ >= "1.1.0":
        TOOL_ENABLED = True


@run_for_optional_imports("openai", "openai")
@run_for_optional_imports(["openai"], "openai")
def test_aoai_chat_completion(credentials_azure_gpt_35_turbo: Credentials):
    config_list = credentials_azure_gpt_35_turbo.config_list
    client = OpenAIWrapper(config_list=config_list)
    response = client.create(messages=[{"role": "user", "content": "2+2="}], cache_seed=None)
    print(response)
    print(client.extract_text_or_completion_object(response))

    # test dialect
    config = config_list[0]
    config["azure_deployment"] = config["model"]
    config["azure_endpoint"] = config.pop("base_url")
    client = OpenAIWrapper(**config)
    response = client.create(messages=[{"role": "user", "content": "2+2="}], cache_seed=None)
    print(response)
    print(client.extract_text_or_completion_object(response))


@run_for_optional_imports("openai", "openai")
@run_for_optional_imports(["openai"], "openai")
def test_fallback_kwargs():
    assert set(inspect.getfullargspec(OpenAI.__init__).kwonlyargs) == OPENAI_FALLBACK_KWARGS
    assert set(inspect.getfullargspec(AzureOpenAI.__init__).kwonlyargs) == AOPENAI_FALLBACK_KWARGS


@run_for_optional_imports("openai", "openai")
@pytest.mark.skipif(not TOOL_ENABLED, reason="openai>=1.1.0 not installed")
@run_for_optional_imports(["openai"], "openai")
def test_oai_tool_calling_extraction(credentials_gpt_4o_mini: Credentials):
    client = OpenAIWrapper(config_list=credentials_gpt_4o_mini.config_list)
    response = client.create(
        messages=[
            {
                "role": "user",
                "content": "What is the weather in San Francisco?",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "getCurrentWeather",
                    "description": "Get the weather in location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "The city and state e.g. San Francisco, CA"},
                            "unit": {"type": "string", "enum": ["c", "f"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
    )
    print(response)
    print(client.extract_text_or_completion_object(response))


@run_for_optional_imports("openai", "openai")
@run_for_optional_imports(["openai"], "openai")
def test_chat_completion(credentials_gpt_4o_mini: Credentials):
    client = OpenAIWrapper(config_list=credentials_gpt_4o_mini.config_list)
    response = client.create(messages=[{"role": "user", "content": "1+1="}])
    print(response)
    print(client.extract_text_or_completion_object(response))


@run_for_optional_imports("openai", "openai")
@run_for_optional_imports(["openai"], "openai")
def test_completion(credentials_azure_gpt_35_turbo_instruct: Credentials):
    client = OpenAIWrapper(config_list=credentials_azure_gpt_35_turbo_instruct.config_list)
    response = client.create(prompt="1+1=")
    print(response)
    print(client.extract_text_or_completion_object(response))


@run_for_optional_imports("openai", "openai")
@run_for_optional_imports(["openai"], "openai")
@pytest.mark.parametrize(
    "cache_seed",
    [
        None,
        42,
    ],
)
def test_cost(credentials_azure_gpt_35_turbo_instruct: Credentials, cache_seed):
    client = OpenAIWrapper(config_list=credentials_azure_gpt_35_turbo_instruct.config_list, cache_seed=cache_seed)
    response = client.create(prompt="1+3=")
    print(response.cost)


@run_for_optional_imports("openai", "openai")
@run_for_optional_imports(["openai"], "openai")
def test_customized_cost(credentials_azure_gpt_35_turbo_instruct: Credentials):
    config_list = credentials_azure_gpt_35_turbo_instruct.config_list
    for config in config_list:
        config.update({"price": [1000, 1000]})
    client = OpenAIWrapper(config_list=config_list, cache_seed=None)
    response = client.create(prompt="1+3=")
    assert response.cost >= 4, (
        f"Due to customized pricing, cost should be > 4. Message: {response.choices[0].message.content}"
    )


@run_for_optional_imports("openai", "openai")
@run_for_optional_imports(["openai"], "openai")
def test_usage_summary(credentials_azure_gpt_35_turbo_instruct: Credentials):
    client = OpenAIWrapper(config_list=credentials_azure_gpt_35_turbo_instruct.config_list)
    response = client.create(prompt="1+3=", cache_seed=None)

    # usage should be recorded
    assert client.actual_usage_summary["total_cost"] > 0, "total_cost should be greater than 0"
    assert client.total_usage_summary["total_cost"] > 0, "total_cost should be greater than 0"

    # check print
    client.print_usage_summary()

    # check clear
    client.clear_usage_summary()
    assert client.actual_usage_summary is None, "actual_usage_summary should be None"
    assert client.total_usage_summary is None, "total_usage_summary should be None"

    # actual usage and all usage should be different
    response = client.create(prompt="1+3=", cache_seed=42)
    assert client.total_usage_summary["total_cost"] > 0, "total_cost should be greater than 0"
    client.clear_usage_summary()
    response = client.create(prompt="1+3=", cache_seed=42)
    assert client.actual_usage_summary is None, "No actual cost should be recorded"

    # check update
    response = client.create(prompt="1+3=", cache_seed=42)
    assert client.total_usage_summary["total_cost"] == response.cost * 2, (
        "total_cost should be equal to response.cost * 2"
    )


@run_for_optional_imports(["openai"], "openai")
def test_log_cache_seed_value(mock_credentials: Credentials, monkeypatch: pytest.MonkeyPatch):
    chat_completion = ChatCompletion(**{
        "id": "chatcmpl-B2ZfaI387UgmnNXS69egxeKbDWc0u",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": "The history of human civilization spans thousands of years, beginning with the emergence of Homo sapiens in Africa around 200,000 years ago. Early humans formed hunter-gatherer societies before transitioning to agriculture during the Neolithic Revolution, around 10,000 BCE, leading to the establishment of permanent settlements. The rise of city-states and empires, such as Mesopotamia, Ancient Egypt, and the Indus Valley, marked significant advancements in governance, trade, and culture. The classical era saw the flourish of philosophies and science in Greece and Rome, while the Middle Ages brought feudalism and the spread of religions. The Renaissance sparked exploration and modernization, culminating in the contemporary globalized world.",
                    "refusal": None,
                    "role": "assistant",
                    "audio": None,
                    "function_call": None,
                    "tool_calls": None,
                },
            }
        ],
        "created": 1739953470,
        "model": "gpt-4o-mini-2024-07-18",
        "object": "chat.completion",
        "service_tier": "default",
        "system_fingerprint": "fp_13eed4fce1",
        "usage": {
            "completion_tokens": 142,
            "prompt_tokens": 23,
            "total_tokens": 165,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
        },
        "cost": 8.864999999999999e-05,
    })

    mock_logger = MagicMock()
    mock_cache_get = MagicMock(return_value=chat_completion)
    monkeypatch.setattr("autogen.oai.client.logger", mock_logger)
    monkeypatch.setattr("autogen.cache.disk_cache.DiskCache.get", mock_cache_get)

    prompt = "Write a 100 word summary on the topic of the history of human civilization."

    # Test single client
    wrapper = OpenAIWrapper(config_list=mock_credentials.config_list)
    _ = wrapper.create(messages=[{"role": "user", "content": prompt}], cache_seed=999)

    mock_logger.debug.assert_called_once()
    actual = mock_logger.debug.call_args[0][0]
    expected = "Using cache with seed value 999 for client OpenAIClient"
    assert actual == expected, f"Expected: {expected}, Actual: {actual}"


@run_for_optional_imports("openai", "openai")
@run_for_optional_imports(["openai"], "openai")
def test_legacy_cache(credentials_gpt_4o_mini: Credentials):
    # Prompt to use for testing.
    prompt = "Write a 100 word summary on the topic of the history of human civilization."

    # Clear cache.
    if os.path.exists(LEGACY_CACHE_DIR):
        shutil.rmtree(LEGACY_CACHE_DIR)

    # Test default cache seed.
    client = OpenAIWrapper(config_list=credentials_gpt_4o_mini.config_list, cache_seed=LEGACY_DEFAULT_CACHE_SEED)
    start_time = time.time()
    cold_cache_response = client.create(messages=[{"role": "user", "content": prompt}])
    end_time = time.time()
    duration_with_cold_cache = end_time - start_time

    start_time = time.time()
    warm_cache_response = client.create(messages=[{"role": "user", "content": prompt}])
    end_time = time.time()
    duration_with_warm_cache = end_time - start_time
    assert cold_cache_response == warm_cache_response
    assert duration_with_warm_cache < duration_with_cold_cache
    assert os.path.exists(os.path.join(LEGACY_CACHE_DIR, str(LEGACY_DEFAULT_CACHE_SEED)))

    # Test with cache seed set through constructor
    client = OpenAIWrapper(config_list=credentials_gpt_4o_mini.config_list, cache_seed=13)
    start_time = time.time()
    cold_cache_response = client.create(messages=[{"role": "user", "content": prompt}])
    end_time = time.time()
    duration_with_cold_cache = end_time - start_time

    start_time = time.time()
    warm_cache_response = client.create(messages=[{"role": "user", "content": prompt}])
    end_time = time.time()
    duration_with_warm_cache = end_time - start_time
    assert cold_cache_response == warm_cache_response
    assert duration_with_warm_cache < duration_with_cold_cache
    assert os.path.exists(os.path.join(LEGACY_CACHE_DIR, str(13)))

    # Test with cache seed set through create method
    client = OpenAIWrapper(config_list=credentials_gpt_4o_mini.config_list)
    start_time = time.time()
    cold_cache_response = client.create(messages=[{"role": "user", "content": prompt}], cache_seed=17)
    end_time = time.time()
    duration_with_cold_cache = end_time - start_time

    start_time = time.time()
    warm_cache_response = client.create(messages=[{"role": "user", "content": prompt}], cache_seed=17)
    end_time = time.time()
    duration_with_warm_cache = end_time - start_time
    assert cold_cache_response == warm_cache_response
    assert duration_with_warm_cache < duration_with_cold_cache
    assert os.path.exists(os.path.join(LEGACY_CACHE_DIR, str(17)))

    # Test using a different cache seed through create method.
    start_time = time.time()
    cold_cache_response = client.create(messages=[{"role": "user", "content": prompt}], cache_seed=21)
    end_time = time.time()
    duration_with_cold_cache = end_time - start_time
    assert duration_with_warm_cache < duration_with_cold_cache
    assert os.path.exists(os.path.join(LEGACY_CACHE_DIR, str(21)))


@run_for_optional_imports(["openai"], "openai")
def test_no_default_cache(credentials_gpt_4o_mini: Credentials):
    # Prompt to use for testing.
    prompt = "Write a 100 word summary on the topic of the history of human civilization."

    # Clear cache.
    if os.path.exists(LEGACY_CACHE_DIR):
        shutil.rmtree(LEGACY_CACHE_DIR)

    # Test default cache which is no cache
    client = OpenAIWrapper(config_list=credentials_gpt_4o_mini.config_list)
    start_time = time.time()
    no_cache_response = client.create(messages=[{"role": "user", "content": prompt}])
    end_time = time.time()
    duration_with_no_cache = end_time - start_time

    # Legacy cache should not be used.
    assert not os.path.exists(os.path.join(LEGACY_CACHE_DIR, str(LEGACY_DEFAULT_CACHE_SEED)))

    # Create cold cache
    client = OpenAIWrapper(config_list=credentials_gpt_4o_mini.config_list, cache_seed=LEGACY_DEFAULT_CACHE_SEED)
    start_time = time.time()
    cold_cache_response = client.create(messages=[{"role": "user", "content": prompt}])
    end_time = time.time()
    duration_with_cold_cache = end_time - start_time

    # Create warm cache
    start_time = time.time()
    warm_cache_response = client.create(messages=[{"role": "user", "content": prompt}])
    end_time = time.time()
    duration_with_warm_cache = end_time - start_time

    # Test that warm cache is the same as cold cache.
    assert cold_cache_response == warm_cache_response
    assert no_cache_response != warm_cache_response

    # Test that warm cache is faster than cold cache and no cache.
    assert duration_with_warm_cache < duration_with_cold_cache
    assert duration_with_warm_cache < duration_with_no_cache

    # Test legacy cache is used.
    assert os.path.exists(os.path.join(LEGACY_CACHE_DIR, str(LEGACY_DEFAULT_CACHE_SEED)))


@run_for_optional_imports("openai", "openai")
@run_for_optional_imports(["openai"], "openai")
def test_cache(credentials_gpt_4o_mini: Credentials):
    # Prompt to use for testing.
    prompt = "Write a 100 word summary on the topic of the history of artificial intelligence."

    # Clear cache.
    if os.path.exists(LEGACY_CACHE_DIR):
        shutil.rmtree(LEGACY_CACHE_DIR)
    cache_dir = ".cache_test"
    assert cache_dir != LEGACY_CACHE_DIR
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    # Test cache set through constructor.
    with Cache.disk(cache_seed=49, cache_path_root=cache_dir) as cache:
        client = OpenAIWrapper(config_list=credentials_gpt_4o_mini.config_list, cache=cache)
        start_time = time.time()
        cold_cache_response = client.create(messages=[{"role": "user", "content": prompt}])
        end_time = time.time()
        duration_with_cold_cache = end_time - start_time

        start_time = time.time()
        warm_cache_response = client.create(messages=[{"role": "user", "content": prompt}])
        end_time = time.time()
        duration_with_warm_cache = end_time - start_time
        assert cold_cache_response == warm_cache_response
        assert duration_with_warm_cache < duration_with_cold_cache
        assert os.path.exists(os.path.join(cache_dir, str(49)))
        # Test legacy cache is not used.
        assert not os.path.exists(os.path.join(LEGACY_CACHE_DIR, str(49)))
        assert not os.path.exists(os.path.join(cache_dir, str(LEGACY_DEFAULT_CACHE_SEED)))

    # Test cache set through method.
    client = OpenAIWrapper(config_list=credentials_gpt_4o_mini.config_list)
    with Cache.disk(cache_seed=312, cache_path_root=cache_dir) as cache:
        start_time = time.time()
        cold_cache_response = client.create(messages=[{"role": "user", "content": prompt}], cache=cache)
        end_time = time.time()
        duration_with_cold_cache = end_time - start_time

        start_time = time.time()
        warm_cache_response = client.create(messages=[{"role": "user", "content": prompt}], cache=cache)
        end_time = time.time()
        duration_with_warm_cache = end_time - start_time
        assert cold_cache_response == warm_cache_response
        assert duration_with_warm_cache < duration_with_cold_cache
        assert os.path.exists(os.path.join(cache_dir, str(312)))
        # Test legacy cache is not used.
        assert not os.path.exists(os.path.join(LEGACY_CACHE_DIR, str(312)))
        assert not os.path.exists(os.path.join(cache_dir, str(LEGACY_DEFAULT_CACHE_SEED)))

    # Test different cache seed.
    with Cache.disk(cache_seed=123, cache_path_root=cache_dir) as cache:
        start_time = time.time()
        cold_cache_response = client.create(messages=[{"role": "user", "content": prompt}], cache=cache)
        end_time = time.time()
        duration_with_cold_cache = end_time - start_time
        assert duration_with_warm_cache < duration_with_cold_cache
        # Test legacy cache is not used.
        assert not os.path.exists(os.path.join(LEGACY_CACHE_DIR, str(123)))
        assert not os.path.exists(os.path.join(cache_dir, str(LEGACY_DEFAULT_CACHE_SEED)))


@run_for_optional_imports(["openai"], "openai")
def test_convert_system_role_to_user() -> None:
    messages = [
        {"content": "Your name is Jack and you are a comedian in a two-person comedy show.", "role": "system"},
        {"content": "Jack, tell me a joke.", "role": "user", "name": "user"},
    ]
    OpenAIClient._convert_system_role_to_user(messages)
    expected = [
        {"content": "Your name is Jack and you are a comedian in a two-person comedy show.", "role": "user"},
        {"content": "Jack, tell me a joke.", "role": "user", "name": "user"},
    ]
    assert messages == expected


def test_openai_llm_config_entry():
    openai_llm_config = OpenAILLMConfigEntry(
        model="gpt-4o-mini", api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly"
    )
    assert openai_llm_config.api_type == "openai"
    assert openai_llm_config.model == "gpt-4o-mini"
    assert openai_llm_config.api_key.get_secret_value() == "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly"
    assert openai_llm_config.base_url is None
    expected = {
        "api_type": "openai",
        "model": "gpt-4o-mini",
        "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
        "tags": [],
        "stream": False,
    }
    actual = openai_llm_config.model_dump()
    assert actual == expected, f"Expected: {expected}, Actual: {actual}"


def test_openai_llm_config_entry_with_verbosity():
    openai_llm_config = OpenAILLMConfigEntry(
        model="gpt-5", api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly", verbosity="low"
    )
    assert openai_llm_config.api_type == "openai"
    assert openai_llm_config.model == "gpt-5"
    assert openai_llm_config.api_key.get_secret_value() == "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly"
    assert openai_llm_config.base_url is None
    expected = {
        "api_type": "openai",
        "model": "gpt-5",
        "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
        "tags": [],
        "stream": False,
        "verbosity": "low",
    }
    actual = openai_llm_config.model_dump()
    assert actual == expected, f"Expected: {expected}, Actual: {actual}"


def test_azure_llm_config_entry() -> None:
    azure_llm_config = AzureOpenAILLMConfigEntry(
        model="gpt-4o-mini",
        api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
        base_url="https://api.openai.com/v1",
        user="unique_user_id",
    )
    expected = {
        "api_type": "azure",
        "model": "gpt-4o-mini",
        "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
        "base_url": "https://api.openai.com/v1",
        "user": "unique_user_id",
        "tags": [],
        "stream": False,
    }
    actual = azure_llm_config.model_dump()
    assert actual == expected, f"Expected: {expected}, Actual: {actual}"

    llm_config = LLMConfig(
        config_list=[azure_llm_config],
    )
    assert llm_config.model_dump() == {
        "config_list": [expected],
    }


def test_deepseek_llm_config_entry() -> None:
    deepseek_llm_config = DeepSeekLLMConfigEntry(
        api_key="fake_api_key",
        model="deepseek-chat",
        max_tokens=8192,
        temperature=0.5,
    )

    expected = {
        "api_type": "deepseek",
        "api_key": "fake_api_key",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "max_tokens": 8192,
        "temperature": 0.5,
        "tags": [],
        "stream": False,
    }
    actual = deepseek_llm_config.model_dump()
    assert actual == expected, actual

    llm_config = LLMConfig(
        config_list=[deepseek_llm_config],
    )
    assert llm_config.model_dump() == {
        "config_list": [expected],
    }

    with pytest.raises(ValidationError, match="Value error, temperature and top_p cannot be set at the same time"):
        deepseek_llm_config = DeepSeekLLMConfigEntry(
            model="deepseek-chat",
            temperature=1,
            top_p=0.8,
        )


class TestOpenAIClientBadRequestsError:
    def test_is_agent_name_error_message(self) -> None:
        assert OpenAIClient._is_agent_name_error_message("Invalid 'messages[0].something") is False
        for i in range(5):
            error_message = f"Invalid 'messages[{i}].name': string does not match pattern. Expected a string that matches the pattern ..."
            assert OpenAIClient._is_agent_name_error_message(error_message) is True

    @pytest.mark.parametrize(
        "error_message, raise_new_error",
        [
            (
                "Invalid 'messages[0].name': string does not match pattern. Expected a string that matches the pattern ...",
                True,
            ),
            (
                "Invalid 'messages[1].name': string does not match pattern. Expected a string that matches the pattern ...",
                True,
            ),
            (
                "Invalid 'messages[0].something': string does not match pattern. Expected a string that matches the pattern ...",
                False,
            ),
        ],
    )
    def test_handle_openai_bad_request_error(self, error_message: str, raise_new_error: bool) -> None:
        def raise_bad_request_error(error_message: str) -> None:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "error": {
                    "message": error_message,
                }
            }
            body = {"error": {"message": "Bad Request error occurred"}}
            raise openai.BadRequestError("Bad Request", response=mock_response, body=body)

        # Function raises BadRequestError
        with pytest.raises(openai.BadRequestError):
            raise_bad_request_error(error_message=error_message)

        wrapped_raise_bad_request_error = OpenAIClient._handle_openai_bad_request_error(raise_bad_request_error)
        if raise_new_error:
            with pytest.raises(
                ValueError,
                match="This error typically occurs when the agent name contains invalid characters, such as spaces or special symbols.",
            ):
                wrapped_raise_bad_request_error(error_message=error_message)
        else:
            with pytest.raises(openai.BadRequestError):
                wrapped_raise_bad_request_error(error_message=error_message)


class TestDeepSeekPatch:
    @pytest.mark.parametrize(
        "messages, expected_messages",
        [
            (
                [
                    {"role": "system", "content": "You are an AG2 Agent."},
                    {"role": "user", "content": "Help me with my problem."},
                ],
                [
                    {"role": "system", "content": "You are an AG2 Agent."},
                    {"role": "user", "content": "Help me with my problem."},
                ],
            ),
            (
                [
                    {"role": "user", "content": "You are an AG2 Agent."},
                    {"role": "user", "content": "Help me with my problem."},
                ],
                [
                    {"role": "user", "content": "You are an AG2 Agent."},
                    {"role": "user", "content": "Help me with my problem."},
                ],
            ),
            (
                [
                    {"role": "assistant", "content": "Help me with my problem."},
                    {"role": "system", "content": "You are an AG2 Agent."},
                ],
                [
                    {"role": "system", "content": "You are an AG2 Agent."},
                    {"role": "assistant", "content": "Help me with my problem."},
                ],
            ),
            (
                [
                    {"role": "assistant", "content": "Help me with my problem."},
                    {"role": "system", "content": "You are an AG2 Agent."},
                    {"role": "user", "content": "Help me with my problem."},
                ],
                [
                    {"role": "system", "content": "You are an AG2 Agent."},
                    {"role": "assistant", "content": "Help me with my problem."},
                    {"role": "user", "content": "Help me with my problem."},
                ],
            ),
        ],
    )
    def test_move_system_message_to_beginning(
        self, messages: list[dict[str, str]], expected_messages: list[dict[str, str]]
    ) -> None:
        OpenAIClient._move_system_message_to_beginning(messages)
        assert messages == expected_messages

    @pytest.mark.parametrize(
        "model, should_patch",
        [
            ("deepseek-reasoner", True),
            ("deepseek", False),
            ("something-else", False),
        ],
    )
    def test_patch_messages_for_deepseek_reasoner(self, model: str, should_patch: bool) -> None:
        kwargs = {
            "messages": [
                {"role": "user", "content": "You are an AG2 Agent."},
                {"role": "system", "content": "You are an AG2 Agent System."},
                {"role": "user", "content": "Help me with my problem."},
            ],
            "model": model,
        }

        if should_patch:
            expected_kwargs = {
                "messages": [
                    {"role": "system", "content": "You are an AG2 Agent System."},
                    {"role": "user", "content": "You are an AG2 Agent."},
                    {"role": "assistant", "content": "Help me with my problem."},
                    {"role": "user", "content": "continue"},
                ],
                "model": "deepseek-reasoner",
            }
        else:
            expected_kwargs = copy.deepcopy(kwargs)

        kwargs = OpenAIClient._patch_messages_for_deepseek_reasoner(**kwargs)
        assert kwargs == expected_kwargs


class TestGemini:
    def test_configure_openai_config_for_gemini_updates_proxy(self):
        config_list = [
            {"model": "gemini-2.5-flash", "api_key": "key1", "model_client_cls": "MockModelClient", "name": "client1"}
        ]
        client = OpenAIWrapper(config_list=config_list)
        openai_config = {}
        config = {"proxy": "http://proxy.example.com:8080"}
        client._configure_openai_config_for_gemini(config, openai_config)
        assert openai_config["proxy"] == "http://proxy.example.com:8080"

    def test_configure_openai_config_for_gemini_no_proxy(self):
        config_list = [
            {"model": "gemini-2.5-flash", "api_key": "key1", "model_client_cls": "MockModelClient", "name": "client1"}
        ]
        config = {}
        openai_config = {}
        client = OpenAIWrapper(config_list=config_list)
        client._configure_openai_config_for_gemini(config, openai_config)
        assert "proxy" not in openai_config


class TestO1:
    @pytest.fixture
    def mock_oai_client(self, mock_credentials: Credentials) -> OpenAIClient:
        config = mock_credentials.config_list[0]
        api_key = config["api_key"]
        return OpenAIClient(OpenAI(api_key=api_key), None)

    @pytest.fixture
    def o1_mini_client(self, credentials_o1_mini: Credentials) -> Generator[OpenAIWrapper, None, None]:
        config_list = credentials_o1_mini.config_list
        return OpenAIWrapper(config_list=config_list, cache_seed=42)

    @pytest.fixture
    def o1_client(self, credentials_o1: Credentials) -> Generator[OpenAIWrapper, None, None]:
        config_list = credentials_o1.config_list
        return OpenAIWrapper(config_list=config_list, cache_seed=42)

    def test_reasoning_remove_unsupported_params(self, mock_oai_client: OpenAIClient) -> None:
        """Test that unsupported parameters are removed with appropriate warnings"""
        test_params = {
            "model": "o1-mini",
            "temperature": 0.7,
            "frequency_penalty": 1.0,
            "presence_penalty": 0.5,
            "top_p": 0.9,
            "logprobs": 5,
            "top_logprobs": 3,
            "logit_bias": {1: 2},
            "valid_param": "keep_me",
        }

        with pytest.warns(UserWarning) as warning_records:
            mock_oai_client._process_reasoning_model_params(test_params)

        # Verify all unsupported params were removed
        assert all(
            param not in test_params
            for param in [
                "temperature",
                "frequency_penalty",
                "presence_penalty",
                "top_p",
                "logprobs",
                "top_logprobs",
                "logit_bias",
            ]
        )

        # Verify valid params were kept
        assert "valid_param" in test_params
        assert test_params["valid_param"] == "keep_me"

        # Verify appropriate warnings were raised
        assert len(warning_records) == 7  # One for each unsupported param

    def test_oai_reasoning_max_tokens_replacement(self, mock_oai_client: OpenAIClient) -> None:
        """Test that max_tokens is replaced with max_completion_tokens"""
        test_params = {"api_type": "openai", "model": "o1-mini", "max_tokens": 100}

        mock_oai_client._process_reasoning_model_params(test_params)

        assert "max_tokens" not in test_params
        assert "max_completion_tokens" in test_params
        assert test_params["max_completion_tokens"] == 100

    @pytest.mark.parametrize(
        ["model_name", "should_merge"],
        [
            ("o1-mini", True),  # TODO: Change to False when o1-mini points to a newer model, e.g. 2024-12-...
            ("o1-preview", True),  # TODO: Change to False when o1-preview points to a newer model, e.g. 2024-12-...
            ("o1-mini-2024-09-12", True),
            ("o1-preview-2024-09-12", True),
            ("o1", False),
            ("o1-2024-12-17", False),
        ],
    )
    def test_oai_reasoning_system_message_handling(
        self, model_name: str, should_merge: str, mock_oai_client: OpenAIClient
    ) -> None:
        """Test system message handling for different model types"""
        system_msg = "You are an AG2 Agent."
        user_msg = "Help me with my problem."
        test_params = {
            "model": model_name,
            "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        }

        mock_oai_client._process_reasoning_model_params(test_params)

        assert len(test_params["messages"]) == 2
        if should_merge:
            # Check system message was merged into user message
            assert test_params["messages"][0]["content"] == f"System message: {system_msg}"
            assert test_params["messages"][0]["role"] == "user"
        else:
            # Check messages remained unchanged
            assert test_params["messages"][0]["content"] == system_msg
            assert test_params["messages"][0]["role"] == "system"

    def _test_completion(self, client: OpenAIWrapper, messages: list[dict[str, str]]) -> None:
        assert isinstance(client, OpenAIWrapper)
        response = client.create(messages=messages, cache_seed=123)

        assert response
        print(f"{response=}")

        text_or_completion_object = client.extract_text_or_completion_object(response)
        print(f"{text_or_completion_object=}")
        assert text_or_completion_object
        assert isinstance(text_or_completion_object[0], str)
        assert "4" in text_or_completion_object[0]

    @pytest.mark.parametrize(
        "messages",
        [
            [{"role": "system", "content": "You are an assistant"}, {"role": "user", "content": "2+2="}],
            [{"role": "user", "content": "2+2="}],
        ],
    )
    @run_for_optional_imports("openai", "openai")
    def test_completion_o1_mini(self, o1_mini_client: OpenAIWrapper, messages: list[dict[str, str]]) -> None:
        self._test_completion(o1_mini_client, messages)

    @pytest.mark.parametrize(
        "messages",
        [
            [{"role": "system", "content": "You are an assistant"}, {"role": "user", "content": "2+2="}],
            [{"role": "user", "content": "2+2="}],
        ],
    )
    @run_for_optional_imports("openai", "openai")
    @pytest.mark.skip(reason="Wait for o1 to be available in CI")
    def test_completion_o1(self, o1_client: OpenAIWrapper, messages: list[dict[str, str]]) -> None:
        self._test_completion(o1_client, messages)


if __name__ == "__main__":
    pass
    # test_aoai_chat_completion()
    # test_oai_tool_calling_extraction()
    # test_chat_completion()
    # test_completion()
    # test_cost()
    # test_usage_summary()
    # test_legacy_cache()
    # test_cache()
