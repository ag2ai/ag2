# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator
from typing import Any

import dashscope
import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config.dashscope import DashScopeClient
from ag2.events import ModelRequest, TextInput
from ag2.exceptions import UnsupportedToolChoiceError
from test.config._helpers import make_tool

_RESPONSE = {
    "request_id": "r1",
    "output": {
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": [{"text": "ok"}]}}],
    },
    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
}


class FakeDashScope:
    """Local DashScope endpoint: records each request's `parameters`, answers with one message."""

    def __init__(self) -> None:
        self.url = ""
        self.parameters: list[dict[str, Any]] = []

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/{tail:.*}", self._generate)
        return app

    async def _generate(self, request: web.Request) -> web.Response:
        self.parameters.append((await request.json()).get("parameters", {}))
        return web.json_response(_RESPONSE)


@pytest_asyncio.fixture
async def dashscope_server() -> AsyncGenerator[FakeDashScope]:
    fake = FakeDashScope()
    # The client points the SDK's global base URL at the server; put it back after.
    base_url = dashscope.base_http_api_url
    async with TestServer(fake.app()) as server:
        fake.url = str(server.make_url("")).rstrip("/")
        yield fake
    dashscope.base_http_api_url = base_url


async def _ask(server: FakeDashScope, choice: str, *, with_tools: bool = True) -> None:
    client = DashScopeClient("qwen-test", api_key="test", base_url=server.url)
    await client(
        messages=[ModelRequest([TextInput("hello")])],
        context=Context(stream=MemoryStream()).with_tool_choice(choice),
        tools=[make_tool().schema] if with_tools else [],
        response_schema=None,
        serializer=SerializerCls,
    )


@pytest.mark.asyncio
class TestToolChoice:
    @pytest.mark.parametrize("mode", ["auto", "none"])
    async def test_mode_is_sent_with_the_tools(self, dashscope_server: FakeDashScope, mode: str) -> None:
        await _ask(dashscope_server, mode)

        [parameters] = dashscope_server.parameters
        assert parameters == IsPartialDict({"tool_choice": mode, "tools": [IsPartialDict({"type": "function"})]})

    async def test_required_is_unsupported(self, dashscope_server: FakeDashScope) -> None:
        with pytest.raises(UnsupportedToolChoiceError, match="dashscope"):
            await _ask(dashscope_server, "required")

        assert dashscope_server.parameters == []

    async def test_no_tools_sends_no_tool_choice(self, dashscope_server: FakeDashScope) -> None:
        await _ask(dashscope_server, "required", with_tools=False)

        [parameters] = dashscope_server.parameters
        assert "tool_choice" not in parameters
        assert "tools" not in parameters
