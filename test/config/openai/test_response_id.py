# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""OpenAI names the response it produced; ag2 keeps that name on the response.

Both surfaces and both modes are covered because each assembles its own final
response: chat completions repeat the id on every chunk, the Responses API
carries it on the terminating event.
"""

import json
from typing import Any

import httpx2
import pytest
from fast_depends.use import SerializerCls

from ag2 import Agent, Context, MemoryStream
from ag2.config.openai import OpenAIClient
from ag2.events import ModelRequest, ModelResponse, TextInput

from ._helpers import ask, config, message, response, streaming_config

COMPLETION: dict[str, Any] = {
    "id": "chatcmpl_1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

CHUNKS: list[dict[str, Any]] = [
    {
        "id": "chatcmpl_1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}],
    },
    {
        "id": "chatcmpl_1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    },
]

STREAMED_RESPONSE: list[dict[str, Any]] = [
    {"type": "response.output_item.done", "sequence_number": 1, "output_index": 0, "item": message("ok")},
    {"type": "response.completed", "sequence_number": 2, "response": response(message("ok"))},
]


def _chat_client(handler: Any, **create_options: Any) -> OpenAIClient:
    """A chat completions client whose transport replays one crafted payload."""
    return OpenAIClient(
        api_key="test",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        create_options={"model": "gpt-4o", **create_options},
    )


def _streaming_chat_client() -> OpenAIClient:
    """A chat completions client that replays `CHUNKS` as one SSE stream."""
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in CHUNKS).encode() + b"data: [DONE]\n\n"

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})

    return _chat_client(handler, stream=True)


async def _chat(client: OpenAIClient) -> ModelResponse:
    return await client(
        messages=[ModelRequest([TextInput("go")])],
        context=Context(stream=MemoryStream()),
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )


@pytest.mark.asyncio
async def test_a_chat_completion_carries_its_id() -> None:
    result = await _chat(_chat_client(lambda request: httpx2.Response(200, json=COMPLETION)))

    assert result.response_id == "chatcmpl_1"


@pytest.mark.asyncio
async def test_a_streamed_chat_completion_carries_its_id() -> None:
    result = await _chat(_streaming_chat_client())

    assert result.response_id == "chatcmpl_1"


@pytest.mark.asyncio
async def test_a_response_carries_its_id() -> None:
    result = await ask(config(response(message("ok"))), stream=MemoryStream())

    assert result.response_id == "resp_1"


@pytest.mark.asyncio
async def test_a_streamed_response_carries_its_id() -> None:
    result = await ask(streaming_config(STREAMED_RESPONSE), stream=MemoryStream())

    assert result.response_id == "resp_1"


@pytest.mark.asyncio
async def test_an_ordinary_run_leaves_the_id_on_the_stream() -> None:
    """No provider-specific event to subscribe to: the id rides the response itself.

    Driven by a real client over the mock transport rather than ``TestConfig``,
    which has no provider id to supply.
    """
    stream = MemoryStream()

    await Agent("a", config=config(response(message("ok")))).ask("go", stream=stream)

    [result] = [e for e in await stream.history.get_events() if isinstance(e, ModelResponse)]
    assert result.response_id == "resp_1"
