# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import warnings

import httpx2
import pytest
from dirty_equals import IsPartialDict

from ag2 import Agent
from ag2.config import OpenAICompletionsConfig, OpenAIConfig, OpenAIResponsesConfig
from ag2.config.input_acceptance import accepts_input, input_modalities
from ag2.config.openai import OpenAICompletionsConfig as ProviderCompletionsConfig
from ag2.config.openai import OpenAIConfig as ProviderConfig
from ag2.events import BinaryInput, BinaryType

from ._helpers import message, response


def _chat_response(request: httpx2.Request) -> httpx2.Response:
    assert request.url.path == "/v1/chat/completions"
    assert json.loads(request.content) == IsPartialDict({
        "model": "test-model",
        "max_completion_tokens": 20,
        "extra_option": True,
    })
    return httpx2.Response(
        200,
        json={
            "id": "chat-1",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
        },
    )


def _responses_response(request: httpx2.Request) -> httpx2.Response:
    assert request.url.path == "/v1/responses"
    assert json.loads(request.content) == IsPartialDict({"model": "test-model", "max_output_tokens": 20})
    return httpx2.Response(200, json=response(message("Hello")))


@pytest.mark.parametrize("config_class", [OpenAIConfig, ProviderConfig])
def test_openai_config_is_the_responses_config(config_class: type[OpenAIResponsesConfig]) -> None:
    assert config_class is OpenAIResponsesConfig


@pytest.mark.parametrize("config_class", [OpenAICompletionsConfig, ProviderCompletionsConfig, OpenAIConfig])
def test_configs_do_not_warn(config_class: type[OpenAICompletionsConfig] | type[OpenAIResponsesConfig]) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        config = config_class("test-model", api_key="test")
        copied = config.copy(model="other-model")
        config.create()
        copied.create()

    assert not caught
    assert type(copied) is config_class
    assert copied.model == "other-model"


@pytest.mark.asyncio
async def test_openai_config_calls_the_responses_endpoint() -> None:
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(_responses_response)) as http_client:
        config = OpenAIConfig("test-model", api_key="test", http_client=http_client, max_output_tokens=20)
        reply = await Agent("test", config=config).ask("Hi")

    assert reply.body == "Hello"


@pytest.mark.asyncio
async def test_completions_config_keeps_the_chat_endpoint() -> None:
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(_chat_response)) as http_client:
        config = OpenAICompletionsConfig(
            "test-model",
            api_key="test",
            http_client=http_client,
            max_completion_tokens=20,
            extra_body={"extra_option": True},
        )
        reply = await Agent("test", config=config).ask("Hi")

    assert reply.body == "Hello"


def test_chat_completions_parameters_are_rejected_by_openai_config() -> None:
    with pytest.raises(TypeError, match="max_completion_tokens"):
        OpenAIConfig("test-model", max_completion_tokens=20)  # type: ignore[call-arg]


def test_input_acceptance_follows_the_bound_api() -> None:
    completions = OpenAICompletionsConfig("test-model")
    responses = OpenAIConfig("test-model")

    assert input_modalities(completions) == {"image": True, "audio": True, "video": False, "pdf": True}
    assert accepts_input(completions, "user", BinaryInput(b"audio", media_type="audio/wav", kind=BinaryType.AUDIO))
    assert input_modalities(responses) == input_modalities(OpenAIResponsesConfig("test-model"))
    assert not accepts_input(responses, "user", BinaryInput(b"audio", media_type="audio/wav", kind=BinaryType.AUDIO))
