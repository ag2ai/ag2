# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible servers may stream tool calls without the ``index`` field."""

import json
from typing import Any

import httpx2
import pytest
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config.openai import OpenAIClient
from ag2.events import ModelRequest, TextInput


def _chunk(tool_calls: list[dict[str, Any]], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl_1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gemini",
        "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": finish_reason}],
    }


def _call(call_id: str | None, name: str | None, arguments: str) -> dict[str, Any]:
    function: dict[str, Any] = {"arguments": arguments}
    if name:
        function["name"] = name
    call: dict[str, Any] = {"type": "function", "function": function}
    if call_id:
        call["id"] = call_id
    return call


async def _stream(chunks: list[dict[str, Any]]) -> Any:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = OpenAIClient(
        api_key="test",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        create_options={"model": "gemini", "stream": True},
    )
    return await client(
        messages=[ModelRequest([TextInput("go")])],
        context=Context(stream=MemoryStream()),
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )


@pytest.mark.asyncio
async def test_parallel_tool_calls_without_index_stay_separate() -> None:
    result = await _stream([
        _chunk([_call("call_a", "get_weather", '{"city": "Paris"}')]),
        _chunk([_call("call_b", "get_time", '{"tz": "UTC"}')], "tool_calls"),
    ])

    assert [(c.id, c.name, c.arguments) for c in result.tool_calls.calls] == [
        ("call_a", "get_weather", '{"city": "Paris"}'),
        ("call_b", "get_time", '{"tz": "UTC"}'),
    ]


@pytest.mark.asyncio
async def test_argument_fragments_without_index_join_the_last_call() -> None:
    result = await _stream([
        _chunk([_call("call_a", "get_weather", '{"city": ')]),
        _chunk([_call(None, None, '"Paris"}')], "tool_calls"),
    ])

    assert [(c.id, c.name, c.arguments) for c in result.tool_calls.calls] == [
        ("call_a", "get_weather", '{"city": "Paris"}'),
    ]
