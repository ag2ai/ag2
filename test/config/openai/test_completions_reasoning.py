# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Chat Completions reasoning extraction across provider field names.

OpenAI-compatible servers split on where they attach reasoning tokens:
DeepSeek and SGLang use ``reasoning_content``; vLLM, OpenRouter, and
Ollama-compatible endpoints use ``reasoning``. Both non-streaming messages
and streaming deltas must check both fields and emit ``ModelReasoning`` once.
"""

import json
from typing import Any

import httpx2
import pytest
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config import OpenAICompletionsConfig
from ag2.events import ModelMessage, ModelMessageChunk, ModelReasoning, ModelRequest, TextInput

from ._helpers import recording


def _completion(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl_1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl_1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _sse(chunks: list[dict[str, Any]]) -> bytes:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"


def _http_client_json(payload: dict[str, Any]) -> httpx2.AsyncClient:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=payload)

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


def _http_client_stream(chunks: list[dict[str, Any]]) -> httpx2.AsyncClient:
    body = _sse(chunks)

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


async def _run(config: OpenAICompletionsConfig) -> list[ModelReasoning]:
    events = await _run_all(config)
    return [e for e in events if isinstance(e, ModelReasoning)]


async def _run_all(config: OpenAICompletionsConfig) -> list[Any]:
    stream = MemoryStream()
    captured = recording(stream)
    client = config.create()
    await client(
        messages=[ModelRequest([TextInput("go")])],
        context=Context(stream=stream),
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )
    return captured


@pytest.mark.asyncio
async def test_non_streaming_reasoning_field() -> None:
    config = OpenAICompletionsConfig(
        model="test-model",
        api_key="test",
        http_client=_http_client_json(_completion({"role": "assistant", "content": "ok", "reasoning": "thought"})),
    )

    assert await _run(config) == [ModelReasoning("thought")]


@pytest.mark.asyncio
async def test_non_streaming_reasoning_content_field() -> None:
    # DeepSeek / SGLang shape: non-streaming message carries reasoning_content.
    config = OpenAICompletionsConfig(
        model="test-model",
        api_key="test",
        http_client=_http_client_json(
            _completion({"role": "assistant", "content": "ok", "reasoning_content": "thought"})
        ),
    )

    assert await _run(config) == [ModelReasoning("thought")]


@pytest.mark.asyncio
async def test_streaming_reasoning_field() -> None:
    # vLLM / OpenRouter shape: streaming delta carries reasoning.
    config = OpenAICompletionsConfig(
        model="test-model",
        api_key="test",
        streaming=True,
        http_client=_http_client_stream([
            _chunk({"role": "assistant", "reasoning": "step 1 "}),
            _chunk({"reasoning": "step 2"}),
            _chunk({"content": "ok"}, finish_reason="stop"),
        ]),
    )

    events = await _run_all(config)
    assert [e for e in events if isinstance(e, ModelReasoning)] == [
        ModelReasoning("step 1 "),
        ModelReasoning("step 2"),
    ]
    assert [e for e in events if isinstance(e, ModelMessageChunk)] == [ModelMessageChunk("ok")]
    assert [e for e in events if isinstance(e, ModelMessage)] == [ModelMessage("ok")]


@pytest.mark.asyncio
async def test_streaming_reasoning_content_field() -> None:
    config = OpenAICompletionsConfig(
        model="test-model",
        api_key="test",
        streaming=True,
        http_client=_http_client_stream([
            _chunk({"role": "assistant", "reasoning_content": "step 1 "}),
            _chunk({"reasoning_content": "step 2"}),
            _chunk({"content": "ok"}, finish_reason="stop"),
        ]),
    )

    assert await _run(config) == [ModelReasoning("step 1 "), ModelReasoning("step 2")]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_both_fields_present_emits_reasoning_once(streaming: bool) -> None:
    payload_fields = {"reasoning_content": "preferred", "reasoning": "fallback"}
    if streaming:
        http_client = _http_client_stream([
            _chunk({"role": "assistant", **payload_fields}),
            _chunk({"content": "ok"}, finish_reason="stop"),
        ])
    else:
        http_client = _http_client_json(_completion({"role": "assistant", "content": "ok", **payload_fields}))

    config = OpenAICompletionsConfig(
        model="test-model",
        api_key="test",
        streaming=streaming,
        http_client=http_client,
    )

    assert await _run(config) == [ModelReasoning("preferred")]


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_value", [None, ""])
@pytest.mark.parametrize("streaming", [False, True])
async def test_empty_reasoning_content_falls_back_to_reasoning(streaming: bool, empty_value: Any) -> None:
    payload_fields = {"reasoning_content": empty_value, "reasoning": "fallback"}
    if streaming:
        http_client = _http_client_stream([
            _chunk({"role": "assistant", **payload_fields}),
            _chunk({"content": "ok"}, finish_reason="stop"),
        ])
    else:
        http_client = _http_client_json(_completion({"role": "assistant", "content": "ok", **payload_fields}))

    config = OpenAICompletionsConfig(
        model="test-model",
        api_key="test",
        streaming=streaming,
        http_client=http_client,
    )

    assert await _run(config) == [ModelReasoning("fallback")]
