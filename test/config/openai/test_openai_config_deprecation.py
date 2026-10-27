# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import warnings
from pathlib import Path

import httpx2
import pytest
from dirty_equals import IsPartialDict

from ag2 import Agent
from ag2.config import OpenAIChatConfig, OpenAIConfig, OpenAIResponsesConfig
from ag2.config.input_acceptance import accepts_input, input_modalities
from ag2.config.openai import OpenAIChatConfig as ProviderChatConfig
from ag2.config.openai import OpenAIConfig as ProviderConfig
from ag2.events import BinaryInput, BinaryType


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


@pytest.mark.parametrize("config_class", [OpenAIChatConfig, ProviderChatConfig, OpenAIResponsesConfig])
def test_explicit_configs_do_not_warn(config_class: type[OpenAIChatConfig] | type[OpenAIResponsesConfig]) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        config = config_class("test-model", api_key="test")
        copied = config.copy(model="other-model")
        config.create()
        copied.create()

    assert not caught
    assert type(copied) is config_class
    assert config.model == "test-model"
    assert copied.model == "other-model"


@pytest.mark.parametrize("config_class", [OpenAIConfig, ProviderConfig])
def test_legacy_construction_and_copy_do_not_warn(config_class: type[OpenAIConfig]) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        config = config_class("test-model", api_key="test", max_completion_tokens=20)
        copied = config.copy(temperature=0.5)

    assert not caught
    assert type(copied) is config_class
    assert copied.max_completion_tokens == 20
    assert copied.temperature == 0.5
    assert config.max_completion_tokens == 20
    with pytest.warns(DeprecationWarning, match="OpenAIChatConfig.*OpenAIResponsesConfig.*1.4.0") as emitted:
        copied.create()
    assert len(emitted) == 1
    assert Path(emitted[0].filename) == Path(__file__)


@pytest.mark.asyncio
@pytest.mark.parametrize("config_class", [OpenAIChatConfig, OpenAIConfig, ProviderConfig])
async def test_chat_endpoint_is_preserved(config_class: type[OpenAIChatConfig]) -> None:
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(_chat_response)) as http_client:
        config = config_class(
            "test-model",
            api_key="test",
            http_client=http_client,
            max_completion_tokens=20,
            extra_body={"extra_option": True},
        )
        agent = Agent("test", config=config)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            reply = await agent.ask("Hi")

    assert reply.body == "Hello"
    assert len(caught) == (0 if config_class is OpenAIChatConfig else 1)
    if caught:
        assert issubclass(caught[0].category, DeprecationWarning)
        assert "1.4.0" in str(caught[0].message)


@pytest.mark.parametrize("config_class", [OpenAIChatConfig, OpenAIConfig])
def test_chat_input_acceptance_is_preserved(config_class: type[OpenAIChatConfig]) -> None:
    config = config_class("test-model")

    assert input_modalities(config) == {"image": True, "audio": True, "video": False, "pdf": True}
    assert accepts_input(config, "user", BinaryInput(b"audio", media_type="audio/wav", kind=BinaryType.AUDIO))
    assert not accepts_input(config, "user", BinaryInput(b"audio", media_type="audio/ogg", kind=BinaryType.AUDIO))
    assert not accepts_input(config, "tool", BinaryInput(b"image", media_type="image/png", kind=BinaryType.IMAGE))
