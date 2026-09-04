# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import httpx
import pytest
from fast_depends.use import SerializerCls
from google.oauth2.credentials import Credentials

from ag2 import Context, MemoryStream
from ag2.config.gemini import GeminiClient
from ag2.events import ModelRequest, TextInput
from ag2.tools import WebSearchTool, tool
from ag2.tools.tool import Tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{city}: 22C, sunny"


async def _send(client: GeminiClient, tools: list[Tool]) -> None:
    context = Context(stream=MemoryStream())
    schemas = [s for t in tools for s in await t.schemas(context)]

    await client(
        messages=[ModelRequest([TextInput("capital of France?")])],
        context=context,
        tools=schemas,
        response_schema=None,
        serializer=SerializerCls,
    )


@pytest.mark.asyncio
class TestToolConfigOnTheWire:
    async def test_builtin_with_function_tool_sends_flag(
        self,
        capturing_http_client: httpx.AsyncClient,
        captured_request: dict[str, Any],
    ) -> None:
        client = GeminiClient(
            model="gemini-3.6-flash",
            api_key="test",
            vertexai=False,
            http_client=capturing_http_client,
        )

        await _send(client, [WebSearchTool(), get_weather])

        assert captured_request["body"]["toolConfig"] == {"includeServerSideToolInvocations": True}

    async def test_builtin_alone_sends_no_tool_config(
        self,
        capturing_http_client: httpx.AsyncClient,
        captured_request: dict[str, Any],
    ) -> None:
        client = GeminiClient(
            model="gemini-3.6-flash",
            api_key="test",
            vertexai=False,
            http_client=capturing_http_client,
        )

        await _send(client, [WebSearchTool()])

        assert "toolConfig" not in captured_request["body"]

    async def test_vertexai_omits_flag(
        self,
        capturing_http_client: httpx.AsyncClient,
        captured_request: dict[str, Any],
    ) -> None:
        client = GeminiClient(
            model="gemini-3.6-flash",
            vertexai=True,
            credentials=Credentials(token="fake-token"),
            project="test-project",
            location="us-central1",
            http_client=capturing_http_client,
        )

        await _send(client, [WebSearchTool(), get_weather])

        assert "toolConfig" not in captured_request["body"]
