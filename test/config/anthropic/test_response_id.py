# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Anthropic names every message it produces; ag2 keeps that name on the response."""

import json
from typing import Any

import httpx2
import pytest
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config.anthropic import AnthropicConfig
from ag2.events import ModelRequest, ModelResponse, TextInput

MESSAGE: dict[str, Any] = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}

STREAM_EVENTS: tuple[dict[str, Any], ...] = (
    {"type": "message_start", "message": {**MESSAGE, "content": [], "stop_reason": None}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 1},
    },
    {"type": "message_stop"},
)


def _config(handler: Any, *, streaming: bool = False) -> AnthropicConfig:
    return AnthropicConfig(
        model="claude-haiku-4-5",
        api_key="test",
        streaming=streaming,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


def _message_config() -> AnthropicConfig:
    """A config whose transport replays one crafted message."""
    return _config(lambda request: httpx2.Response(200, json=MESSAGE))


def _streaming_config() -> AnthropicConfig:
    """A config that streams that same message as SSE."""
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in STREAM_EVENTS)

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})

    return _config(handler, streaming=True)


async def _ask(config: AnthropicConfig) -> ModelResponse:
    return await config.create()(
        messages=[ModelRequest([TextInput("hi")])],
        context=Context(stream=MemoryStream()),
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )


@pytest.mark.asyncio
async def test_a_completed_message_carries_its_id() -> None:
    result = await _ask(_message_config())

    assert result.response_id == "msg_1"


@pytest.mark.asyncio
async def test_a_streamed_message_carries_its_id() -> None:
    result = await _ask(_streaming_config())

    assert result.response_id == "msg_1"
