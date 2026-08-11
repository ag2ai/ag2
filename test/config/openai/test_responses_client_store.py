# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config.openai import OpenAIResponsesConfig
from ag2.events import ModelRequest, TextInput
from ag2.tools.builtin.file_search import FileSearchTool

ENCRYPTED_REASONING = "reasoning.encrypted_content"


def _capturing_config(captured: dict[str, object], **kwargs: object) -> OpenAIResponsesConfig:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "model": "m",
                "status": "completed",
                "output": [],
                "usage": None,
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "instructions": None,
                "metadata": {},
                "text": {"format": {"type": "text"}},
            },
        )

    return OpenAIResponsesConfig(
        model="m",
        api_key="test",
        base_url="http://test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **kwargs,  # type: ignore[arg-type]
    )


async def _request_body(captured: dict[str, object], tools: list, **kwargs: object) -> dict[str, object]:
    context = Context(stream=MemoryStream())
    schemas = []
    for t in tools:
        schemas.extend(await t.schemas(context))
    client = _capturing_config(captured, **kwargs).create()
    await client(
        messages=[ModelRequest([TextInput("hi")])],
        context=context,
        tools=schemas,
        response_schema=None,
        serializer=SerializerCls,
    )
    return captured["body"]  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_store_false_requests_encrypted_reasoning() -> None:
    """With store=False nothing is persisted server-side, so a reasoning item replayed
    by id on the next turn 404s. The encrypted payload makes it self-contained."""
    captured: dict[str, object] = {}

    body = await _request_body(captured, [], store=False)

    assert body["include"] == [ENCRYPTED_REASONING]


@pytest.mark.asyncio
async def test_store_true_does_not_request_encrypted_reasoning() -> None:
    """Stored responses resolve reasoning items by id, so the payload is dead weight."""
    captured: dict[str, object] = {}

    body = await _request_body(captured, [], store=True)

    assert ENCRYPTED_REASONING not in (body.get("include") or [])


@pytest.mark.asyncio
async def test_store_false_appends_to_tool_derived_includes() -> None:
    """The reasoning include is additive — it must not clobber tool-derived ones."""
    captured: dict[str, object] = {}

    body = await _request_body(
        captured,
        [FileSearchTool(vector_store_ids=["vs_1"], include_results=True)],
        store=False,
    )

    assert body["include"] == ["file_search_call.results", ENCRYPTED_REASONING]
