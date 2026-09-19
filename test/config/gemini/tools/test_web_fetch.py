# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
import pytest
from fast_depends.use import SerializerCls
from google.genai import types

from ag2 import Context, MemoryStream
from ag2.config.gemini import GeminiClient
from ag2.config.gemini.mappers import build_tools
from ag2.events import ModelRequest, TextInput
from ag2.tools.builtin.web_fetch import WebFetchTool


@pytest.mark.asyncio
async def test_defaults(context: Context) -> None:
    tool = WebFetchTool()

    [schema] = await tool.schemas(context)

    assert build_tools([schema]) == [
        types.Tool(url_context=types.UrlContext()),
    ]


@pytest.mark.asyncio
async def test_version_gated_options_are_ignored(context: Context) -> None:
    """Gemini honours none of these, and refuses none of them either."""
    tool = WebFetchTool(
        strict=True,
        use_cache=False,
        response_inclusion="excluded",
        version="web_fetch_20250910",
    )

    [schema] = await tool.schemas(context)

    assert build_tools([schema]) == [
        types.Tool(url_context=types.UrlContext()),
    ]


def _capturing_client(captured: dict[str, Any]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
            },
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_a_run_carrying_version_gated_options_still_reaches_the_api() -> None:
    """Gemini ignores the options rather than refusing them, so the run goes out as usual."""
    captured: dict[str, Any] = {}
    run_context = Context(stream=MemoryStream())
    client = GeminiClient(
        model="gemini-3.6-flash",
        api_key="test",
        vertexai=False,
        http_client=_capturing_client(captured),
    )
    schemas = await WebFetchTool(
        strict=True,
        use_cache=False,
        response_inclusion="excluded",
        version="web_fetch_20250910",
    ).schemas(run_context)

    await client(
        messages=[ModelRequest([TextInput("hi")])],
        context=run_context,
        tools=schemas,
        response_schema=None,
        serializer=SerializerCls,
    )

    assert captured["body"]["tools"] == [{"urlContext": {}}]
