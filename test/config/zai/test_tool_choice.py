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
from ag2.config.zai import ZAIClient
from ag2.events import ModelRequest, TextInput
from ag2.exceptions import UnsupportedToolChoiceError
from test.config._helpers import make_tool

_COMPLETION = {
    "id": "c1",
    "created": 0,
    "model": "glm-test",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class _Recorder:
    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_COMPLETION)


async def _ask(recorder: _Recorder, choice: str, *, with_tools: bool = True, **options: Any) -> None:
    client = ZAIClient(
        api_key="test.secret",
        http_client=httpx.Client(transport=httpx.MockTransport(recorder.handle)),
        create_options={"model": "glm-test", **options},
    )
    await client(
        messages=[ModelRequest([TextInput("hello")])],
        context=Context(stream=MemoryStream()).with_tool_choice(choice),
        tools=[make_tool().schema] if with_tools else [],
        response_schema=None,
        serializer=SerializerCls,
    )


@pytest.mark.asyncio
class TestToolChoice:
    async def test_auto_keeps_the_tools(self) -> None:
        recorder = _Recorder()

        await _ask(recorder, "auto")

        [body] = recorder.bodies
        assert body == IsPartialDict({"tools": [IsPartialDict({"type": "function"})]})

    async def test_none_sends_no_tools_and_no_tool_choice(self) -> None:
        recorder = _Recorder()

        await _ask(recorder, "none", tool_choice="auto")

        [body] = recorder.bodies
        assert "tools" not in body
        assert "tool_choice" not in body

    async def test_required_is_unsupported(self) -> None:
        recorder = _Recorder()

        with pytest.raises(UnsupportedToolChoiceError, match="zai"):
            await _ask(recorder, "required")

        assert recorder.bodies == []

    async def test_no_tools_keeps_configured_tool_choice(self) -> None:
        recorder = _Recorder()

        await _ask(recorder, "none", with_tools=False, tool_choice="auto")

        [body] = recorder.bodies
        assert "tools" not in body
        assert body["tool_choice"] == "auto"
