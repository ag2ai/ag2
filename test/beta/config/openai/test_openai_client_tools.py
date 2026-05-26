# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest
from fast_depends.use import SerializerCls

from autogen.beta import Context, MemoryStream
from autogen.beta.config.openai import OpenAIClient
from autogen.beta.events import ModelRequest, TextInput
from autogen.beta.tools.final.function_tool import FunctionDefinition, FunctionToolSchema


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


def _weather_tool() -> FunctionToolSchema:
    return FunctionToolSchema(
        function=FunctionDefinition(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        )
    )


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
    client = OpenAIClient(
        api_key="test",
        http_client=_capturing_client(captured),
        create_options={"model": "gpt-4o"},
    )

    await client(
        messages=[ModelRequest([TextInput("weather?")])],
        context=Context(stream=MemoryStream()),
        tools=[_weather_tool()],
        response_schema=None,
        serializer=SerializerCls,
    )

    tools = captured["body"]["tools"]
    assert isinstance(tools, list)
    assert [t["function"]["name"] for t in tools] == ["get_weather"]
