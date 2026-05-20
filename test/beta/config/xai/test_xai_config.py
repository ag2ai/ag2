# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fast_depends.use import SerializerCls
from xai_sdk.chat import chat_pb2

from autogen.beta import Context
from autogen.beta.config import XAIConfig
from autogen.beta.config.xai import XAIClient
from autogen.beta.events import ModelRequest, TextInput
from autogen.beta.stream import MemoryStream


def _fake_sample_response() -> SimpleNamespace:
    return SimpleNamespace(
        content="",
        reasoning_content="",
        tool_calls=[],
        model="grok-4-fast",
        finish_reason="FINISH_REASON_STOP",
        usage=None,
        proto=chat_pb2.GetChatCompletionResponse(),
    )


async def _invoke_client(config: XAIConfig, *, streaming: bool = False) -> tuple[MagicMock, MagicMock]:
    """Invoke ``config.create()`` against a mocked xai-sdk and return the (xai_client_mock, stub_chat) pair."""
    stub_chat = MagicMock()

    if streaming:

        async def empty_stream() -> AsyncIterator[tuple[Any, Any]]:
            if False:
                yield  # pragma: no cover

        stub_chat.stream = MagicMock(return_value=empty_stream())
    else:
        stub_chat.sample = AsyncMock(return_value=_fake_sample_response())

    with patch("autogen.beta.config.xai.xai_client.AsyncClient") as mock_async_client:
        instance = MagicMock()
        instance.chat.create.return_value = stub_chat
        mock_async_client.return_value = instance

        client = config.create()

        await client(
            messages=[ModelRequest([TextInput("hi")])],
            context=Context(stream=MemoryStream(persist_all=True)),
            tools=[],
            response_schema=None,
            serializer=SerializerCls,
        )

    return instance, stub_chat


def test_copy_without_overrides_returns_new_equal_instance() -> None:
    config = XAIConfig(model="grok-4-fast", temperature=0.2, streaming=True)

    copied = config.copy()

    assert copied == config
    assert copied is not config


def test_copy_applies_overrides_without_mutating_original() -> None:
    config = XAIConfig(model="grok-4-fast", temperature=0.2, streaming=False)

    copied = config.copy(model="grok-4.20", temperature=0.8, streaming=True, reasoning_effort="high")

    assert copied.model == "grok-4.20"
    assert copied.temperature == 0.8
    assert copied.streaming is True
    assert copied.reasoning_effort == "high"

    assert config.model == "grok-4-fast"
    assert config.temperature == 0.2
    assert config.streaming is False
    assert config.reasoning_effort is None


def test_create_returns_xai_client() -> None:
    config = XAIConfig(model="grok-4-fast", api_key="test-key")
    client = config.create()

    assert isinstance(client, XAIClient)


def test_defaults() -> None:
    config = XAIConfig(model="grok-4-fast")

    assert config.api_host == "api.x.ai"
    assert config.timeout == 1620.0
    assert config.api_key is None
    assert config.streaming is False
    assert config.temperature is None
    assert config.max_tokens is None
    assert config.reasoning_effort is None
    assert config.include is None


def test_api_host_can_be_overridden() -> None:
    config = XAIConfig(model="grok-4-fast", api_host="custom.x.ai")
    assert config.api_host == "custom.x.ai"


@pytest.mark.asyncio
async def test_reasoning_effort_forwarded_to_chat_create() -> None:
    instance, _ = await _invoke_client(XAIConfig(model="grok-4-fast", reasoning_effort="high"))

    assert instance.chat.create.call_args.kwargs.get("reasoning_effort") == "high"


@pytest.mark.asyncio
async def test_include_forwarded_to_chat_create() -> None:
    instance, _ = await _invoke_client(
        XAIConfig(model="grok-4-fast", include=["verbose_streaming", "code_execution_call_output"])
    )

    assert instance.chat.create.call_args.kwargs.get("include") == [
        "verbose_streaming",
        "code_execution_call_output",
    ]


@pytest.mark.asyncio
async def test_streaming_flag_routes_to_chat_stream() -> None:
    _, stub_chat = await _invoke_client(XAIConfig(model="grok-4-fast", streaming=True), streaming=True)

    assert stub_chat.stream.called
    assert not stub_chat.sample.called


@pytest.mark.asyncio
async def test_non_streaming_routes_to_chat_sample() -> None:
    _, stub_chat = await _invoke_client(XAIConfig(model="grok-4-fast", streaming=False))

    assert stub_chat.sample.called
    assert not stub_chat.stream.called


@pytest.mark.asyncio
async def test_none_fields_are_dropped_from_chat_create_kwargs() -> None:
    instance, _ = await _invoke_client(XAIConfig(model="grok-4-fast", temperature=0.5))
    kwargs = instance.chat.create.call_args.kwargs

    assert kwargs.get("temperature") == 0.5
    assert "top_p" not in kwargs
    assert "max_tokens" not in kwargs


def test_files_client_not_supported() -> None:
    config = XAIConfig(model="grok-4-fast")

    with pytest.raises(NotImplementedError, match="Files API"):
        config.create_files_client()
