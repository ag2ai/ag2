# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Anthropic client usage normalization (input_tokens → prompt_tokens, cache keys)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from autogen.beta.config.anthropic import AnthropicClient
from autogen.beta.events import ModelResponse, Usage


def _make_usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> MagicMock:
    """Create a mock Anthropic Usage object with model_dump()."""
    usage = MagicMock()
    usage.model_dump.return_value = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
    }
    return usage


def _make_response(usage: Any, content_text: str = "Hello") -> MagicMock:
    """Create a mock Anthropic Message response."""
    text_block = MagicMock()
    text_block.__class__ = type("TextBlock", (), {})
    # Match isinstance checks by using real types
    from anthropic.types import TextBlock

    text_block = MagicMock(spec=TextBlock)
    text_block.text = content_text

    response = MagicMock()
    response.content = [text_block]
    response.usage = usage
    response.model = "claude-sonnet-4-6"
    response.stop_reason = "end_turn"
    return response


def _make_context() -> AsyncMock:
    ctx = AsyncMock()
    ctx.send = AsyncMock()
    return ctx


@pytest.mark.asyncio()
async def test_process_response_normalizes_usage():
    client = AnthropicClient(api_key="test", prompt_caching=False)
    usage = _make_usage(input_tokens=100, output_tokens=25)
    response = _make_response(usage)

    result = await client._process_response(response, _make_context())

    assert isinstance(result, ModelResponse)
    assert result.usage == Usage(prompt_tokens=100, completion_tokens=25)


@pytest.mark.asyncio()
async def test_process_response_includes_cache_creation_tokens():
    client = AnthropicClient(api_key="test", prompt_caching=False)
    usage = _make_usage(input_tokens=3, output_tokens=10, cache_creation_input_tokens=5058)
    response = _make_response(usage)

    result = await client._process_response(response, _make_context())

    assert result.usage == Usage(
        prompt_tokens=3,
        completion_tokens=10,
        cache_creation_input_tokens=5058,
    )
    assert result.usage.cache_read_input_tokens is None


@pytest.mark.asyncio()
async def test_process_response_includes_cache_read_tokens():
    client = AnthropicClient(api_key="test", prompt_caching=False)
    usage = _make_usage(input_tokens=3, output_tokens=14, cache_read_input_tokens=5043)
    response = _make_response(usage)

    result = await client._process_response(response, _make_context())

    assert result.usage == Usage(
        prompt_tokens=3,
        completion_tokens=14,
        cache_read_input_tokens=5043,
    )
    assert result.usage.cache_creation_input_tokens is None


@pytest.mark.asyncio()
async def test_process_response_no_usage():
    client = AnthropicClient(api_key="test", prompt_caching=False)
    response = _make_response(usage=None)
    response.usage = None

    result = await client._process_response(response, _make_context())

    assert result.usage == Usage(prompt_tokens=0, completion_tokens=0)


@pytest.mark.asyncio()
async def test_process_stream_normalizes_usage():
    client = AnthropicClient(api_key="test", prompt_caching=False)
    usage = _make_usage(input_tokens=50, output_tokens=12, cache_read_input_tokens=4000)

    final_message = MagicMock()
    final_message.usage = usage
    final_message.model = "claude-sonnet-4-6"
    final_message.stop_reason = "end_turn"

    # Mock stream: emits one text_delta then stops
    text_delta = MagicMock()
    text_delta.type = "content_block_delta"
    text_delta.delta = MagicMock()
    text_delta.delta.type = "text_delta"
    text_delta.delta.text = "Hi"

    stream = AsyncMock()
    stream.__aiter__ = lambda self: _async_iter([text_delta])
    stream.get_final_message = AsyncMock(return_value=final_message)

    result = await client._process_stream(stream, _make_context())

    assert isinstance(result, ModelResponse)
    assert result.usage == Usage(
        prompt_tokens=50,
        completion_tokens=12,
        cache_read_input_tokens=4000,
    )
    assert result.usage.cache_creation_input_tokens is None


async def _async_iter(items):
    for item in items:
        yield item
