# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Prompt caching is what the application configures, not what the API defaults to.

Both OpenAI surfaces, because an application that moves between them should not
have to rediscover its own cache identity.
"""

import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx2
import pytest
from fast_depends.pydantic import PydanticSerializer

from ag2 import Context, MemoryStream
from ag2.config import OpenAIConfig, OpenAIResponsesConfig
from ag2.events import ModelRequest, TextInput

CHAT_COMPLETION: dict[str, Any] = {
    "id": "chatcmpl_1",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

RESPONSE: dict[str, Any] = {
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
}


def _capturing_client(captured: dict[str, Any], payload: dict[str, Any]) -> httpx2.AsyncClient:
    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json=payload)

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


async def _ask(config: OpenAIConfig | OpenAIResponsesConfig) -> None:
    await config.create()(
        messages=[ModelRequest([TextInput("capital of France?")])],
        context=Context(stream=MemoryStream()),
        tools=[],
        response_schema=None,
        serializer=PydanticSerializer(),
    )


async def _chat_body(**overrides: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    await _ask(
        OpenAIConfig(
            model="m",
            api_key="test",
            base_url="http://test/v1",
            http_client=_capturing_client(captured, CHAT_COMPLETION),
            **overrides,
        )
    )
    body: dict[str, Any] = captured["body"]
    return body


async def _responses_body(**overrides: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    await _ask(
        OpenAIResponsesConfig(
            model="m",
            api_key="test",
            base_url="http://test/v1",
            http_client=_capturing_client(captured, RESPONSE),
            **overrides,
        )
    )
    body: dict[str, Any] = captured["body"]
    return body


BOTH_SURFACES = pytest.mark.parametrize("body_of", (_chat_body, _responses_body), ids=("chat", "responses"))

Surface = Callable[..., Awaitable[dict[str, Any]]]


@BOTH_SURFACES
@pytest.mark.asyncio
async def test_prompt_cache_fields_are_absent_when_unset(body_of: Surface) -> None:
    body = await body_of()

    assert "prompt_cache_key" not in body
    assert "prompt_cache_options" not in body


@BOTH_SURFACES
@pytest.mark.asyncio
async def test_an_empty_options_object_is_not_a_setting(body_of: Surface) -> None:
    """Setting neither ``mode`` nor ``ttl`` is not a request to cache differently.

    An empty object on the wire is not harmless: a model without the feature refuses
    the whole request over it.
    """
    body = await body_of(prompt_cache_options={})

    assert "prompt_cache_options" not in body


@BOTH_SURFACES
@pytest.mark.asyncio
async def test_the_cache_key_reaches_the_request_unchanged(body_of: Surface) -> None:
    body = await body_of(prompt_cache_key="checkout-agent")

    assert body["prompt_cache_key"] == "checkout-agent"


@BOTH_SURFACES
@pytest.mark.asyncio
async def test_the_cache_options_reach_the_request_unchanged(body_of: Surface) -> None:
    body = await body_of(prompt_cache_options={"mode": "explicit", "ttl": "30m"})

    assert body["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}


@BOTH_SURFACES
@pytest.mark.asyncio
async def test_one_option_travels_without_the_other(body_of: Surface) -> None:
    body = await body_of(prompt_cache_options={"ttl": "30m"})

    assert body["prompt_cache_options"] == {"ttl": "30m"}


@BOTH_SURFACES
@pytest.mark.asyncio
async def test_a_cache_key_does_not_drag_in_an_end_user_identifier(body_of: Surface) -> None:
    body = await body_of(prompt_cache_key="checkout-agent")

    assert "user" not in body


@BOTH_SURFACES
@pytest.mark.asyncio
async def test_the_cache_and_the_end_user_are_named_separately(body_of: Surface) -> None:
    body = await body_of(prompt_cache_key="checkout-agent", user="u-42")

    assert body["prompt_cache_key"] == "checkout-agent"
    assert body["user"] == "u-42"


def test_chat_cache_settings_survive_a_copy() -> None:
    config = OpenAIConfig(model="m").copy(
        prompt_cache_key="checkout-agent",
        prompt_cache_options={"mode": "explicit"},
    )

    assert config.prompt_cache_key == "checkout-agent"
    assert config.prompt_cache_options == {"mode": "explicit"}


def test_responses_cache_settings_survive_a_copy() -> None:
    config = OpenAIResponsesConfig(model="m").copy(
        prompt_cache_key="checkout-agent",
        prompt_cache_options={"mode": "explicit"},
    )

    assert config.prompt_cache_key == "checkout-agent"
    assert config.prompt_cache_options == {"mode": "explicit"}
