# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls

from ag2.config import MistralConfig, ModelProvider
from ag2.config.mistral import MistralClient, MistralFilesClient
from ag2.events import ModelRequest, TextInput
from test.config.mistral._helpers import FakeChat, FakeMistralClient, make_call_context


def test_provider() -> None:
    assert MistralConfig(model="mistral-small-latest").provider is ModelProvider.MISTRAL


def test_defaults_are_unset() -> None:
    config = MistralConfig(model="mistral-small-latest")

    assert config.api_key is None
    assert config.server_url is None
    assert config.streaming is False
    assert config.temperature is None


def test_copy_overrides_without_mutating_original() -> None:
    config = MistralConfig(model="mistral-small-latest", temperature=0.1)

    copy = config.copy(model="mistral-large-latest", streaming=True)

    assert (copy.model, copy.streaming, copy.temperature) == ("mistral-large-latest", True, 0.1)
    assert (config.model, config.streaming) == ("mistral-small-latest", False)


def test_create_returns_client() -> None:
    assert isinstance(MistralConfig(model="mistral-small-latest").create(), MistralClient)


def test_create_files_client() -> None:
    config = MistralConfig(model="mistral-small-latest", api_key="test-key")

    assert isinstance(config.create_files_client(), MistralFilesClient)


@pytest.mark.asyncio
async def test_configured_options_reach_the_api() -> None:
    """Set options are forwarded; unset ones are dropped, not sent as ``None``."""
    chat = FakeChat()
    client = MistralConfig(model="mistral-small-latest", temperature=0, max_tokens=64).create()
    client._client = FakeMistralClient(chat)

    await client(
        messages=[ModelRequest([TextInput("hello")])],
        context=make_call_context(),
        tools=(),
        response_schema=None,
        serializer=SerializerCls,
    )

    assert chat.kwargs == IsPartialDict({
        "model": "mistral-small-latest",
        "temperature": 0,
        "max_tokens": 64,
    })
    assert "top_p" not in chat.kwargs
