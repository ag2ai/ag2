# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
import pytest
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config.gemini import GeminiClient
from ag2.events import ModelRequest, TextInput
from ag2.tools import WebSearchTool, tool
from ag2.tools.tool import Tool

_MODES = [("auto", "AUTO"), ("none", "NONE"), ("required", "ANY")]


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return city


async def _body(choice: str, tools: list[Tool]) -> dict[str, Any]:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
            },
        )

    client = GeminiClient(
        model="gemini-3.6-flash",
        api_key="test",
        vertexai=False,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    context = Context(stream=MemoryStream()).with_tool_choice(choice)
    await client(
        messages=[ModelRequest([TextInput("weather?")])],
        context=context,
        tools=[s for t in tools for s in await t.schemas(context)],
        response_schema=None,
        serializer=SerializerCls,
    )
    [body] = bodies
    return body


@pytest.mark.asyncio
class TestToolChoice:
    @pytest.mark.parametrize(("mode", "expected"), _MODES)
    async def test_mode_is_sent_with_the_tools(self, mode: str, expected: str) -> None:
        body = await _body(mode, [get_weather])

        assert body == IsPartialDict({
            "toolConfig": {"functionCallingConfig": {"mode": expected}},
            "tools": [{"functionDeclarations": [IsPartialDict({"name": "get_weather"})]}],
        })

    @pytest.mark.parametrize(("mode", "expected"), _MODES)
    async def test_existing_tool_config_is_kept(self, mode: str, expected: str) -> None:
        body = await _body(mode, [WebSearchTool(), get_weather])

        assert body["toolConfig"] == {
            "includeServerSideToolInvocations": True,
            "functionCallingConfig": {"mode": expected},
        }

    async def test_no_tools_sends_no_tool_config(self) -> None:
        body = await _body("required", [])

        assert "toolConfig" not in body
        assert "tools" not in body
