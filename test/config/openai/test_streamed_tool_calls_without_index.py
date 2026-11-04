# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Streamed tool calls from OpenAI-compatible endpoints that leave out `index`.

Chat Completions names each streamed tool call by `index`, and the client keys its
accumulator on it. Compatible endpoints (Gemini's OpenAI-compatible API, for one) send
whole tool calls with no `index`; each one is still a call of its own.
"""

import json
from typing import Any

import httpx2
import pytest
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config.openai import OpenAIClient
from ag2.events import ModelRequest, TextInput


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl_1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gemini-3.8-flash",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


async def _stream(chunks: list[dict[str, Any]]) -> Any:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = OpenAIClient(
        api_key="test",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        create_options={"model": "gemini-3.8-flash", "stream": True},
    )
    return await client(
        messages=[ModelRequest([TextInput("go")])],
        context=Context(stream=MemoryStream()),
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )


@pytest.mark.asyncio
async def test_a_tool_call_without_an_index_is_kept() -> None:
    result = await _stream([
        _chunk({"role": "assistant", "tool_calls": [_call("call_1", "get_weather", '{"city": "Paris"}')]}),
        _chunk({}, finish_reason="tool_calls"),
    ])

    assert [(c.id, c.name, c.arguments) for c in result.tool_calls.calls] == [
        ("call_1", "get_weather", '{"city": "Paris"}')
    ]


@pytest.mark.asyncio
async def test_parallel_tool_calls_without_an_index_stay_separate() -> None:
    result = await _stream([
        _chunk({
            "role": "assistant",
            "tool_calls": [
                _call("call_1", "get_weather", '{"city": "Paris"}'),
                _call("call_2", "get_weather", '{"city": "Rome"}'),
            ],
        }),
        _chunk({}, finish_reason="tool_calls"),
    ])

    assert [(c.id, c.arguments) for c in result.tool_calls.calls] == [
        ("call_1", '{"city": "Paris"}'),
        ("call_2", '{"city": "Rome"}'),
    ]


@pytest.mark.asyncio
async def test_indexed_tool_calls_are_still_accumulated_by_index() -> None:
    first = {"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"ci'}}
    rest = {"index": 0, "function": {"arguments": 'ty": "Paris"}'}}
    result = await _stream([
        _chunk({"role": "assistant", "tool_calls": [first]}),
        _chunk({"tool_calls": [rest]}),
        _chunk({}, finish_reason="tool_calls"),
    ])

    assert [(c.id, c.arguments) for c in result.tool_calls.calls] == [("call_1", '{"city": "Paris"}')]
