# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
#!/usr/bin/env python3 -m pytest

import os
import shutil
import time
from collections.abc import Generator

import pytest

from autogen import OpenAIWrapper
from autogen.cache.cache import Cache
from autogen.oai.client import LEGACY_CACHE_DIR, LEGACY_DEFAULT_CACHE_SEED, OpenAIClient

from ..conftest import Credentials

TOOL_ENABLED = False
try:
    import openai
    from openai import OpenAI  # noqa: F401

    if openai.__version__ >= "1.1.0":
        TOOL_ENABLED = True
except ImportError:
    skip = True
else:
    skip = False


@pytest.mark.openai
@pytest.mark.skipif(skip, reason="openai>=1 not installed")
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


@pytest.mark.openai
@pytest.mark.skipif(skip or not TOOL_ENABLED, reason="openai>=1.1.0 not installed")
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


@pytest.mark.openai
@pytest.mark.skipif(skip, reason="openai>=1 not installed")
def test_chat_completion(credentials_gpt_4o_mini: Credentials):
    client = OpenAIWrapper(config_list=credentials_gpt_4o_mini.config_list)
    response = client.create(messages=[{"role": "user", "content": "1+1="}])
    print(response)
    print(client.extract_text_or_completion_object(response))


@pytest.mark.openai
@pytest.mark.skipif(skip, reason="openai>=1 not installed")
def test_completion(credentials_azure_gpt_35_turbo_instruct: Credentials):
    client = OpenAIWrapper(config_list=credentials_azure_gpt_35_turbo_instruct.config_list)
    response = client.create(prompt="1+1=")
    print(response)
    print(client.extract_text_or_completion_object(response))


@pytest.mark.openai
@pytest.mark.skipif(skip, reason="openai>=1 not installed")
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


@pytest.mark.openai
@pytest.mark.skipif(skip, reason="openai>=1 not installed")
def test_customized_cost(credentials_azure_gpt_35_turbo_instruct: Credentials):
    config_list = credentials_azure_gpt_35_turbo_instruct.config_list
    for config in config_list:
        config.update({"price": [1000, 1000]})
    client = OpenAIWrapper(config_list=config_list, cache_seed=None)
    response = client.create(prompt="1+3=")
    assert response.cost >= 4, (
        f"Due to customized pricing, cost should be > 4. Message: {response.choices[0].message.content}"
    )


@pytest.mark.openai
@pytest.mark.skipif(skip, reason="openai>=1 not installed")
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


@pytest.mark.openai
@pytest.mark.skipif(skip, reason="openai>=1 not installed")
def test_legacy_cache(credentials_gpt_4o_mini: Credentials):
    # Prompt to use for testing.
    prompt = "Write a 100 word summary on the topic of the history of human civilization."

    # Clear cache.
    if os.path.exists(LEGACY_CACHE_DIR):
        shutil.rmtree(LEGACY_CACHE_DIR)

    # Test default cache seed.
    client = OpenAIWrapper(config_list=credentials_gpt_4o_mini.config_list)
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


@pytest.mark.openai
@pytest.mark.skipif(skip, reason="openai>=1 not installed")
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


class TestO1:
    @pytest.fixture
    def mock_oai_client(self, mock_credentials: Credentials) -> OpenAIClient:
        config = mock_credentials.config_list[0]
        api_key = config["api_key"]
        return OpenAIClient(OpenAI(api_key=api_key), None)

    @pytest.fixture
    def o1_mini_client(self, credentials_o1_mini: Credentials) -> Generator[OpenAIWrapper, None, None]:
        config_list = credentials_o1_mini.config_list
        yield OpenAIWrapper(config_list=config_list, cache_seed=42)

    @pytest.fixture
    def o1_client(self, credentials_o1: Credentials) -> Generator[OpenAIWrapper, None, None]:
        config_list = credentials_o1.config_list
        yield OpenAIWrapper(config_list=config_list, cache_seed=42)

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
        test_params = {"model": "o1-mini", "max_tokens": 100}

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

    def _test_completition(self, client: OpenAIWrapper, messages: list[dict[str, str]]) -> None:
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
    @pytest.mark.openai
    def test_completition_o1_mini(self, o1_mini_client: OpenAIWrapper, messages: list[dict[str, str]]) -> None:
        self._test_completition(o1_mini_client, messages)

    @pytest.mark.parametrize(
        "messages",
        [
            [{"role": "system", "content": "You are an assistant"}, {"role": "user", "content": "2+2="}],
            [{"role": "user", "content": "2+2="}],
        ],
    )
    @pytest.mark.openai
    @pytest.mark.skip(reason="Wait for o1 to be available in CI")
    def test_completition_o1(self, o1_client: OpenAIWrapper, messages: list[dict[str, str]]) -> None:
        self._test_completition(o1_client, messages)


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
