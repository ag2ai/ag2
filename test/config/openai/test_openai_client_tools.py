# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config.openai import OpenAIClient
from ag2.events import ModelRequest, TextInput
from ag2.tools import tool


def _capturing_client(captured: dict[str, object]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_empty_tools_omits_tools_field() -> None:
    captured: dict[str, object] = {}
    client = OpenAIClient(
        api_key="test",
        http_client=_capturing_client(captured),
        create_options={"model": "gpt-4o"},
    )

    await client(
        messages=[ModelRequest([TextInput("capital of France?")])],
        context=Context(stream=MemoryStream()),
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )

    assert "tools" not in captured["body"]


@pytest.mark.asyncio
async def test_non_empty_tools_serialized() -> None:
    captured: dict[str, object] = {}
    context = Context(stream=MemoryStream())

    @tool(description="Get weather")
    def get_weather(city: str) -> str:
        return f"Weather in {city}"

    [weather_schema] = await get_weather.schemas(context)

    client = OpenAIClient(
        api_key="test",
        http_client=_capturing_client(captured),
        create_options={"model": "gpt-4o"},
    )

    await client(
        messages=[ModelRequest([TextInput("weather?")])],
        context=context,
        tools=[weather_schema],
        response_schema=None,
        serializer=SerializerCls,
    )

    tools = captured["body"]["tools"]
    assert isinstance(tools, list)
    assert [t["function"]["name"] for t in tools] == ["get_weather"]
