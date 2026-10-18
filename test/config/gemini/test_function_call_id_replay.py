# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
import pytest
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream, ToolResult
from ag2.config.gemini import GeminiClient
from ag2.events import ModelRequest, TextInput, ToolResultEvent, ToolResultsEvent


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("cities", [("Paris",), ("Paris", "Tokyo")])
async def test_provider_function_call_ids_survive_the_next_request(streaming: bool, cities: tuple[str, ...]) -> None:
    captured: list[dict[str, Any]] = []
    call_parts = [
        {"functionCall": {"id": f"call-{city}", "name": "get_weather", "args": {"city": city}}} for city in cities
    ]
    call_parts[0]["thoughtSignature"] = "c2ln"

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        parts = call_parts if len(captured) == 1 else [{"text": "Done"}]
        response = {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}]}
        if streaming:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(response)}\n\n",
            )
        return httpx.Response(200, json=response)

    context = Context(stream=MemoryStream())
    initial = ModelRequest([TextInput("Check the weather.")])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = GeminiClient(
            model="gemini-3-flash-preview", api_key="test", http_client=http_client, streaming=streaming
        )
        response = await client(
            messages=[initial], context=context, tools=[], response_schema=None, serializer=SerializerCls
        )
        assert [call.id for call in response.tool_calls.calls] == [f"call-{city}" for city in cities]

        results = ToolResultsEvent(
            results=[
                ToolResultEvent.from_call(call, ToolResult(f"{city}: sunny"))
                for call, city in zip(response.tool_calls.calls, cities, strict=True)
            ]
        )
        await client(
            messages=[initial, response, results],
            context=context,
            tools=[],
            response_schema=None,
            serializer=SerializerCls,
        )

    assert captured[1]["contents"] == [
        {"role": "user", "parts": [{"text": "Check the weather."}]},
        {"role": "model", "parts": call_parts},
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "id": f"call-{city}",
                        "name": "get_weather",
                        "response": {"result": f"{city}: sunny"},
                    }
                }
                for city in cities
            ],
        },
    ]
