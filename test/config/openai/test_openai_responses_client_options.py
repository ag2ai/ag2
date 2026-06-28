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

# Optional generation params that must never be serialized as an explicit
# `null`: strict OpenAI-compatible servers (e.g. vLLM) reject e.g.
# `service_tier: null` / `truncation: null`. When unset they must be omitted.
NULLABLE_OPTIONS = (
    "temperature",
    "top_p",
    "max_output_tokens",
    "max_tool_calls",
    "top_logprobs",
    "metadata",
    "service_tier",
    "truncation",
)


def _capturing_config(captured: dict[str, object], **overrides: object) -> OpenAIResponsesConfig:
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
        **overrides,
    )


async def _request_body(captured: dict[str, object], **overrides: object) -> dict[str, object]:
    client = _capturing_config(captured, **overrides).create()
    await client(
        messages=[ModelRequest([TextInput("capital of France?")])],
        context=Context(stream=MemoryStream(), prompt=["You are a helpful assistant."]),
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )
    return captured["body"]  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_unset_options_omitted_not_null() -> None:
    captured: dict[str, object] = {}
    body = await _request_body(captured)

    for key in NULLABLE_OPTIONS:
        assert key not in body, f"{key} should be omitted when unset, got {body.get(key)!r}"
    assert [k for k, v in body.items() if v is None] == []


@pytest.mark.asyncio
async def test_set_options_serialized() -> None:
    captured: dict[str, object] = {}
    body = await _request_body(
        captured,
        temperature=0.7,
        top_p=0.95,
        service_tier="auto",
        truncation="disabled",
    )

    assert body["temperature"] == 0.7
    assert body["top_p"] == 0.95
    assert body["service_tier"] == "auto"
    assert body["truncation"] == "disabled"
