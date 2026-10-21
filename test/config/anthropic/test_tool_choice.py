# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx2
import pytest
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config.anthropic import AnthropicConfig
from ag2.events import ModelRequest, TextInput
from ag2.tools import tool
from ag2.tools.schemas import ToolSchema

_MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return city


async def _body(choice: str, *, with_tools: bool = True, drop_tools_for_none: bool = False) -> dict[str, Any]:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=_MESSAGE)

    config = AnthropicConfig(
        model="claude-haiku-4-5",
        api_key="test",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        drop_tools_for_none=drop_tools_for_none,
    )
    context = Context(stream=MemoryStream()).with_tool_choice(choice)
    tools: list[ToolSchema] = await get_weather.schemas(context) if with_tools else []
    await config.create()(
        messages=[ModelRequest([TextInput("weather?")])],
        context=context,
        tools=tools,
        response_schema=None,
        serializer=SerializerCls,
    )
    [body] = bodies
    return body


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "expected"), [("auto", "auto"), ("none", "none"), ("required", "any")])
async def test_mode_is_sent_with_the_tools(mode: str, expected: str) -> None:
    body = await _body(mode)

    assert body == IsPartialDict({
        "tool_choice": {"type": expected},
        "tools": [IsPartialDict({"name": "get_weather"})],
    })


@pytest.mark.asyncio
async def test_no_tools_sends_no_tool_choice() -> None:
    body = await _body("required", with_tools=False)

    assert "tool_choice" not in body
    assert "tools" not in body


@pytest.mark.asyncio
async def test_drop_tools_for_none_sends_no_tools() -> None:
    body = await _body("none", drop_tools_for_none=True)

    assert "tool_choice" not in body
    assert "tools" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "expected"), [("auto", "auto"), ("required", "any")])
async def test_drop_tools_for_none_keeps_other_modes(mode: str, expected: str) -> None:
    body = await _body(mode, drop_tools_for_none=True)

    assert body == IsPartialDict({"tool_choice": {"type": expected}, "tools": [IsPartialDict({"name": "get_weather"})]})
