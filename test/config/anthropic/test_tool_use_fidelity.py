# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``ToolUseBlock`` carries more than id/name/input.

``caller`` distinguishes a call the model made directly from one a server tool
made (``bash_20250124`` accepts ``allowed_callers``), and ``toolset_name`` is
mandatory on the tool_result of a browser/computer member tool. Both must survive
the round trip, so both paths park them on ``vendor_metadata``.
"""

from types import SimpleNamespace

import pytest
from anthropic.types import Message, ToolUseBlock, Usage

from ag2 import Context, MemoryStream
from ag2.config.anthropic import AnthropicClient
from ag2.events import ModelResponse
from test.config.anthropic._helpers import FakeStream

BLOCK = ToolUseBlock.model_construct(
    id="toolu_1",
    name="navigate",
    input={"url": "https://example.com"},
    type="tool_use",
    caller={"type": "direct"},
    toolset_name="browser",
)


async def _non_streaming() -> ModelResponse:
    client = AnthropicClient(api_key="test", prompt_caching=False)
    message = Message.model_construct(
        id="m1",
        type="message",
        role="assistant",
        model="claude-sonnet-5",
        content=[BLOCK],
        usage=Usage(input_tokens=1, output_tokens=1),
        stop_reason="tool_use",
    )
    return await client._process_response(message, Context(stream=MemoryStream()))


async def _streaming() -> ModelResponse:
    client = AnthropicClient(api_key="test", prompt_caching=False)
    events = [
        SimpleNamespace(type="content_block_start", content_block=BLOCK),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"url": "https://example.com"}'),
        ),
        SimpleNamespace(type="content_block_stop"),
    ]
    return await client._process_stream(FakeStream(events, stop_reason="tool_use"), Context(stream=MemoryStream()))


@pytest.mark.asyncio
async def test_non_streaming_keeps_caller_and_toolset_name() -> None:
    [call] = (await _non_streaming()).tool_calls.calls

    assert call.vendor_metadata == {"caller": {"type": "direct"}, "toolset_name": "browser"}


@pytest.mark.asyncio
async def test_streaming_keeps_caller_and_toolset_name() -> None:
    [call] = (await _streaming()).tool_calls.calls

    assert call.vendor_metadata == {"caller": {"type": "direct"}, "toolset_name": "browser"}


@pytest.mark.asyncio
async def test_plain_tool_use_stays_empty() -> None:
    client = AnthropicClient(api_key="test", prompt_caching=False)
    plain = ToolUseBlock(id="toolu_2", name="my_func", input={"x": 1}, type="tool_use")
    message = Message.model_construct(
        id="m1",
        type="message",
        role="assistant",
        model="claude-sonnet-5",
        content=[plain],
        usage=Usage(input_tokens=1, output_tokens=1),
        stop_reason="tool_use",
    )

    [call] = (await client._process_response(message, Context(stream=MemoryStream()))).tool_calls.calls
    assert call.vendor_metadata == {}
