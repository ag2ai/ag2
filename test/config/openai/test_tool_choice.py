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
from ag2.config import OpenAIConfig
from ag2.events import ModelRequest, TextInput
from ag2.tools import tool
from ag2.tools.schemas import ToolSchema
from test.config.openai._helpers import capturing_config, message, response

_COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return city


async def _call(client: Any, choice: str | None, *, with_tools: bool = True) -> None:
    context = Context(stream=MemoryStream())
    if choice is not None:
        context = context.with_tool_choice(choice)
    tools: list[ToolSchema] = await get_weather.schemas(context) if with_tools else []
    await client(
        messages=[ModelRequest([TextInput("weather?")])],
        context=context,
        tools=tools,
        response_schema=None,
        serializer=SerializerCls,
    )


async def _chat_body(choice: str | None, *, with_tools: bool = True, **config: Any) -> dict[str, Any]:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=_COMPLETION)

    model_config = OpenAIConfig(
        model="gpt-4o",
        api_key="test",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        **config,
    )
    await _call(model_config.create(), choice, with_tools=with_tools)
    [body] = bodies
    return body


@pytest.mark.asyncio
class TestChatCompletions:
    @pytest.mark.parametrize("mode", ["auto", "none", "required"])
    async def test_mode_is_sent_with_the_tools(self, mode: str) -> None:
        body = await _chat_body(mode)

        assert body == IsPartialDict({"tool_choice": mode, "tools": [IsPartialDict({"type": "function"})]})

    @pytest.mark.parametrize("mode", ["auto", "none"])
    async def test_per_call_choice_overrides_config(self, mode: str) -> None:
        body = await _chat_body(mode, tool_choice="required")

        assert body["tool_choice"] == mode

    async def test_config_choice_kept_without_per_call_choice(self) -> None:
        body = await _chat_body(None, tool_choice="required")

        assert body["tool_choice"] == "required"

    async def test_no_tools_sends_no_tool_choice(self) -> None:
        body = await _chat_body("required", with_tools=False)

        assert "tool_choice" not in body
        assert "tools" not in body

    async def test_drop_tools_for_none_sends_no_tools(self) -> None:
        body = await _chat_body("none", drop_tools_for_none=True, tool_choice="required", parallel_tool_calls=True)

        assert "tools" not in body
        assert "tool_choice" not in body
        assert "parallel_tool_calls" not in body

    @pytest.mark.parametrize("mode", ["auto", "required"])
    async def test_drop_tools_for_none_keeps_tools_for_other_modes(self, mode: str) -> None:
        body = await _chat_body(mode, drop_tools_for_none=True, parallel_tool_calls=True)

        assert body == IsPartialDict({"tool_choice": mode, "parallel_tool_calls": True})
        assert len(body["tools"]) == 1


@pytest.mark.asyncio
class TestResponses:
    @pytest.mark.parametrize("mode", ["auto", "none", "required"])
    async def test_mode_is_sent_with_the_tools(self, mode: str) -> None:
        config, bodies = capturing_config(response(message("ok")))

        await _call(config.create(), mode)

        [body] = bodies
        assert body == IsPartialDict({"tool_choice": mode, "tools": [IsPartialDict({"type": "function"})]})

    async def test_no_tools_sends_no_tool_choice(self) -> None:
        config, bodies = capturing_config(response(message("ok")))

        await _call(config.create(), "required", with_tools=False)

        [body] = bodies
        assert "tool_choice" not in body
