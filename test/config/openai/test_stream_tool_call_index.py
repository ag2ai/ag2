# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Streamed tool calls are assembled by ``index``, or without it.

OpenAI tags every tool-call delta with ``index``. Some OpenAI-compatible servers
reached through ``base_url`` (Gemini's compat endpoint) omit it: there a delta
carrying an ``id`` starts a new call and one without continues the last.
"""

import json
from typing import Any

import httpx2
import pytest
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls

from ag2 import Agent, Context, MemoryStream
from ag2.config import OpenAICompletionsConfig
from ag2.config.openai import OpenAIClient
from ag2.events import ModelRequest, ModelResponse, TextInput, ToolCallEvent


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl_1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _tool_delta(
    arguments: str,
    *,
    index: int | None = None,
    call_id: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    function: dict[str, Any] = {"arguments": arguments}
    if name is not None:
        function["name"] = name
    call: dict[str, Any] = {"function": function}
    if index is not None:
        call["index"] = index
    if call_id is not None:
        call["id"] = call_id
        call["type"] = "function"
    return _chunk({"tool_calls": [call]})


def _sse(chunks: list[dict[str, Any]]) -> bytes:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"


def _streaming_http_client(*turns: list[dict[str, Any]], bodies: list[dict[str, Any]]) -> httpx2.AsyncClient:
    """Replays one chunk list per request and records each request body into `bodies`."""
    remaining = list(turns)

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, content=_sse(remaining.pop(0)), headers={"content-type": "text/event-stream"})

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


async def _complete(chunks: list[dict[str, Any]]) -> ModelResponse:
    client = OpenAIClient(
        api_key="test",
        http_client=_streaming_http_client(chunks, bodies=[]),
        create_options={"model": "test-model", "stream": True},
    )
    return await client(
        messages=[ModelRequest([TextInput("go")])],
        context=Context(stream=MemoryStream()),
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )


@pytest.mark.asyncio
async def test_indexed_argument_fragments_assemble_per_call() -> None:
    # The shape OpenAI streams: id and name on the first delta only, arguments in pieces.
    result = await _complete([
        _tool_delta("", index=0, call_id="call_a", name="get_weather"),
        _tool_delta('{"ci', index=0),
        _tool_delta('ty": "Paris"}', index=0),
        _tool_delta("", index=1, call_id="call_b", name="get_time"),
        _tool_delta('{"tz": ', index=1),
        _tool_delta('"UTC"}', index=1),
        _chunk({}, "tool_calls"),
    ])

    assert result.tool_calls.calls == [
        ToolCallEvent(id="call_a", name="get_weather", arguments='{"city": "Paris"}'),
        ToolCallEvent(id="call_b", name="get_time", arguments='{"tz": "UTC"}'),
    ]


@pytest.mark.asyncio
async def test_interleaved_indexed_fragments_go_to_their_own_call() -> None:
    result = await _complete([
        _tool_delta("", index=0, call_id="call_a", name="get_weather"),
        _tool_delta("", index=1, call_id="call_b", name="get_time"),
        _tool_delta('{"city": ', index=0),
        _tool_delta('{"tz": ', index=1),
        _tool_delta('"Paris"}', index=0),
        _tool_delta('"UTC"}', index=1),
        _chunk({}, "tool_calls"),
    ])

    assert result.tool_calls.calls == [
        ToolCallEvent(id="call_a", name="get_weather", arguments='{"city": "Paris"}'),
        ToolCallEvent(id="call_b", name="get_time", arguments='{"tz": "UTC"}'),
    ]


@pytest.mark.asyncio
async def test_parallel_calls_without_index_stay_separate() -> None:
    # The shape Gemini's OpenAI-compatible endpoint streams: no index, each call whole.
    result = await _complete([
        _tool_delta('{"city": "Paris"}', call_id="call_a", name="get_weather"),
        _tool_delta('{"tz": "UTC"}', call_id="call_b", name="get_time"),
        _chunk({}, "tool_calls"),
    ])

    assert result.tool_calls.calls == [
        ToolCallEvent(id="call_a", name="get_weather", arguments='{"city": "Paris"}'),
        ToolCallEvent(id="call_b", name="get_time", arguments='{"tz": "UTC"}'),
    ]


@pytest.mark.asyncio
async def test_argument_fragments_without_index_join_the_last_call() -> None:
    result = await _complete([
        _tool_delta('{"city": ', call_id="call_a", name="get_weather"),
        _tool_delta('"Paris"}'),
        _chunk({}, "tool_calls"),
    ])

    assert result.tool_calls.calls == [
        ToolCallEvent(id="call_a", name="get_weather", arguments='{"city": "Paris"}'),
    ]


def get_weather(city: str) -> str:
    return f"Sunny in {city}"


def get_time(tz: str) -> str:
    return f"12:00 {tz}"


@pytest.mark.asyncio
async def test_agent_runs_parallel_tools_streamed_without_index() -> None:
    bodies: list[dict[str, Any]] = []
    config = OpenAICompletionsConfig(
        model="gemini-2.5-flash",
        api_key="test",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        streaming=True,
        http_client=_streaming_http_client(
            [
                _tool_delta('{"city": "Paris"}', call_id="call_a", name="get_weather"),
                _tool_delta('{"tz": "UTC"}', call_id="call_b", name="get_time"),
                _chunk({}, "tool_calls"),
            ],
            [_chunk({"content": "Sunny, 12:00."}, "stop")],
            bodies=bodies,
        ),
    )
    agent = Agent("assistant", config=config, tools=[get_weather, get_time])

    reply = await agent.ask("Weather in Paris and the time in UTC?")

    assert reply.body == "Sunny, 12:00."
    assert [m for m in bodies[1]["messages"] if m["role"] == "tool"] == [
        IsPartialDict({"tool_call_id": "call_a", "content": "Sunny in Paris"}),
        IsPartialDict({"tool_call_id": "call_b", "content": "12:00 UTC"}),
    ]
