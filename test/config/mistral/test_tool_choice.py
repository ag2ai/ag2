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
from ag2.config.mistral import MistralClient
from ag2.events import ModelRequest, TextInput
from test.config._helpers import make_tool

_COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "created": 0,
    "model": "mistral-test",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


async def _body(choice: str, *, with_tools: bool = True) -> dict[str, Any]:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=_COMPLETION)

    client = MistralClient(
        "mistral-test",
        api_key="test",
        async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    await client(
        messages=[ModelRequest([TextInput("hello")])],
        context=Context(stream=MemoryStream()).with_tool_choice(choice),
        tools=[make_tool().schema] if with_tools else [],
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
        "tool_choice": expected,
        "tools": [IsPartialDict({"function": IsPartialDict({"name": "search_docs"})})],
    })


@pytest.mark.asyncio
async def test_no_tools_sends_no_tool_choice() -> None:
    body = await _body("required", with_tools=False)

    assert "tool_choice" not in body
    assert "tools" not in body
