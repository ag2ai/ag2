# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import httpx
import pytest
from dirty_equals import IsPartialDict, IsStr
from fast_depends.pydantic import PydanticSerializer
from zai.core import APIStatusError

from ag2.config.zai import ZAIClient, ZAIConfig, ZAIFilesClient
from ag2.events import ModelRequest, TextInput
from ag2.tools.schemas import ToolSchema
from test.config._helpers import WireRecorder, json_response, make_tool, sse_response
from test.config.zai._helpers import chunk_json, completion_json, make_call_context, wire_config

# Long enough that the SDK's JWT signing does not warn about a short HMAC key.
_JWT_API_KEY = "key-id.a-secret-that-is-at-least-thirty-two-bytes"


async def _ask(config: ZAIConfig, tools: list[ToolSchema] | None = None) -> None:
    await config.create()(
        messages=[ModelRequest([TextInput("hello")])],
        context=make_call_context(),
        tools=tools or [],
        response_schema=None,
        serializer=PydanticSerializer(),
    )


def test_defaults() -> None:
    config = ZAIConfig(model="glm-5.2")

    assert config.model == "glm-5.2"
    assert config.api_key is None
    assert config.base_url is None
    assert config.streaming is False


def test_copy_returns_equal_new_instance() -> None:
    config = ZAIConfig(model="glm-5.2", temperature=0.2)

    copied = config.copy()

    assert copied == config
    assert copied is not config


def test_copy_applies_overrides_without_mutating_original() -> None:
    config = ZAIConfig(model="glm-5.2", temperature=0.2)

    copied = config.copy(model="glm-5.1", temperature=0.7)

    assert copied == ZAIConfig(model="glm-5.1", temperature=0.7)
    assert config == ZAIConfig(model="glm-5.2", temperature=0.2)


def test_create_returns_client() -> None:
    assert isinstance(ZAIConfig(model="glm-5.2").create(), ZAIClient)


@pytest.mark.asyncio
async def test_inference_params_reach_the_request() -> None:
    recorder = WireRecorder(sse_response(chunk_json(content="ok")))

    with wire_config(
        recorder,
        model="glm-5.2",
        streaming=True,
        max_tokens=100,
        temperature=0.2,
        top_p=0.9,
        stop=["END"],
        seed=42,
        tool_choice="auto",
        request_id="req-1",
        user_id="user-1",
        do_sample=True,
        meta={"trace": "abc"},
        request_timeout=30.0,
        watermark_enabled=False,
        tool_stream=True,
        reasoning_effort="high",
    ) as config:
        await _ask(config)

    assert recorder.bodies == [
        {
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "max_tokens": 100,
            "temperature": 0.2,
            "top_p": 0.9,
            "stop": ["END"],
            "seed": 42,
            "tool_choice": "auto",
            "request_id": "req-1",
            "user_id": "user-1",
            "do_sample": True,
            "meta": {"trace": "abc"},
            "watermark_enabled": False,
            "tool_stream": True,
            "reasoning_effort": "high",
            "response_format": None,
            "thinking": None,
        }
    ]
    [request] = recorder.requests
    # `request_timeout` is the per-call timeout, which httpx carries on the request.
    assert request.extensions["timeout"] == httpx.Timeout(30.0).as_dict()
    # The token cache is off by default, so the key travels as-is.
    assert request.headers["authorization"] == "Bearer id.secret"


@pytest.mark.asyncio
async def test_unset_params_are_omitted_and_extra_body_is_forwarded_unshadowed() -> None:
    recorder = WireRecorder(json_response(completion_json()))

    with wire_config(
        recorder,
        model="glm-5.2",
        thinking=True,
        extra_body={"thinking": {"type": "disabled"}, "request_id": "abc"},
    ) as config:
        await _ask(config)

    assert recorder.bodies == [
        {
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "response_format": None,
            "thinking": {"type": "enabled"},
            # Merged in by the SDK, less the key a typed option already sets.
            "request_id": "abc",
        }
    ]


@pytest.mark.asyncio
async def test_extra_body_keys_reach_the_request_body_without_binding_to_create() -> None:
    # `extra_body` is the SDK's escape hatch: its keys join the request JSON without being
    # parameters of `create`, so a field the SDK does not know yet still reaches the API.
    recorder = WireRecorder(json_response(completion_json()))

    with wire_config(recorder, model="glm-5.2", extra_body={"future_param": {"on": True}}) as config:
        await _ask(config)

    assert recorder.bodies == [IsPartialDict({"future_param": {"on": True}})]


@pytest.mark.asyncio
async def test_a_typed_option_wins_over_the_same_key_in_extra_body() -> None:
    # The user guide's precedence; the SDK merges `extra_body` last, so the config has to keep it.
    recorder = WireRecorder(json_response(completion_json()))

    with wire_config(recorder, model="glm-5.2", thinking=True, extra_body={"thinking": {"type": "disabled"}}) as config:
        await _ask(config)

    assert recorder.bodies == [IsPartialDict({"thinking": {"type": "enabled"}})]


@pytest.mark.asyncio
async def test_the_agents_tools_win_over_tools_in_extra_body() -> None:
    recorder = WireRecorder(json_response(completion_json()))

    with wire_config(recorder, model="glm-5.2", extra_body={"tools": [], "future_param": 1}) as config:
        await _ask(config, tools=[make_tool().schema])

    assert recorder.bodies == [
        IsPartialDict({"tools": [IsPartialDict({"type": "function"})], "future_param": 1}),
    ]


@pytest.mark.asyncio
async def test_thinking_false_maps_to_disabled() -> None:
    recorder = WireRecorder(json_response(completion_json()))

    with wire_config(recorder, model="glm-5.2", thinking=False) as config:
        await _ask(config)

    assert recorder.bodies == [
        {
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "response_format": None,
            "thinking": {"type": "disabled"},
        }
    ]


@pytest.mark.asyncio
async def test_client_connection_params_reach_the_request() -> None:
    recorder = WireRecorder(json_response(completion_json()))

    with wire_config(
        recorder,
        model="glm-5.2",
        api_key=_JWT_API_KEY,
        base_url="https://example.test/api/paas/v4/",
        timeout=12.0,
        custom_headers={"x-test": "1"},
        disable_token_cache=False,
        source_channel="ag2-test",
    ) as config:
        await _ask(config)

    [request] = recorder.requests
    assert str(request.url) == "https://example.test/api/paas/v4/chat/completions"
    assert dict(request.headers) == IsPartialDict({
        "x-test": "1",
        "x-source-channel": "ag2-test",
        # With the token cache on, the SDK signs a JWT from the key instead of sending it raw.
        "authorization": IsStr(regex=r"Bearer ey[\w-]+\.[\w-]+\.[\w-]+"),
    })
    assert request.extensions["timeout"] == httpx.Timeout(12.0).as_dict()


@pytest.mark.asyncio
async def test_max_retries_reaches_the_sdk_client() -> None:
    # The SDK retries a 5xx up to `max_retries` times; zero means the first failure is final.
    recorder = WireRecorder(json_response({"error": {"code": "500", "message": "boom"}}, status_code=500))

    with wire_config(recorder, model="glm-5.2", max_retries=0) as config, pytest.raises(APIStatusError):
        await _ask(config)

    assert len(recorder.requests) == 1


@patch("ag2.config.zai.files.ZaiClient")
def test_create_files_client(_mock_zai_client: MagicMock) -> None:
    config = ZAIConfig(model="glm-5.2")

    client = config.create_files_client()

    assert isinstance(client, ZAIFilesClient)


def test_unsupported_penalty_fields_are_rejected() -> None:
    with pytest.raises(TypeError):
        ZAIConfig(model="glm-5.2", frequency_penalty=0.1)  # type: ignore[call-arg]  # the refusal under test

    with pytest.raises(TypeError):
        ZAIConfig(model="glm-5.2", presence_penalty=0.2)  # type: ignore[call-arg]  # the refusal under test


@pytest.mark.asyncio
async def test_sdk_accepts_every_option_the_config_exposes() -> None:
    # Regression: every generation param ZAIConfig exposes must be one the real zai-sdk
    # `Completions.create` accepts, or the SDK raises TypeError before any request is sent.
    recorder = WireRecorder(sse_response(chunk_json(content="ok")))

    with wire_config(
        recorder,
        model="glm-5.2",
        streaming=True,
        max_tokens=100,
        temperature=0.2,
        top_p=0.9,
        stop=["END"],
        seed=42,
        tool_choice="auto",
        request_id="req-1",
        user_id="user-1",
        do_sample=True,
        meta={"trace": "abc"},
        sensitive_word_check={"type": "ALL", "status": "DISABLE"},
        extra={"target": {"language": "python", "code_prefix": "def f(", "code_suffix": ")"}},
        request_timeout=30.0,
        watermark_enabled=False,
        tool_stream=True,
        reasoning_effort="high",
        thinking=True,
        extra_headers={"x-extra": "1"},
        extra_body={"future_param": 1},
    ) as config:
        await _ask(config)

    assert recorder.bodies == [
        {
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "max_tokens": 100,
            "temperature": 0.2,
            "top_p": 0.9,
            "stop": ["END"],
            "seed": 42,
            "tool_choice": "auto",
            "request_id": "req-1",
            "user_id": "user-1",
            "do_sample": True,
            "meta": {"trace": "abc"},
            "sensitive_word_check": {"type": "ALL", "status": "DISABLE"},
            "extra": {"target": {"language": "python", "code_prefix": "def f(", "code_suffix": ")"}},
            "watermark_enabled": False,
            "tool_stream": True,
            "reasoning_effort": "high",
            "response_format": None,
            "thinking": {"type": "enabled"},
            "future_param": 1,
        }
    ]
    [request] = recorder.requests
    assert request.headers["x-extra"] == "1"
