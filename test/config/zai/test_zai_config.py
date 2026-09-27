# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from dirty_equals import IsPartialDict
from fast_depends.pydantic import PydanticSerializer
from zai.api_resource.chat.completions import Completions as RealCompletions

import ag2.config.zai.zai_client as zai_client_module
from ag2.config.zai import ZAIClient, ZAIConfig, ZAIFilesClient
from ag2.events import ModelRequest, TextInput
from ag2.tools.schemas import ToolSchema
from test.config._helpers import make_tool
from test.config.zai._helpers import FakeCompletions, FakeZAIClient, make_call_context


class FakeZAIClientFactory:
    def __init__(self, completions: FakeCompletions) -> None:
        self.completions = completions
        self.kwargs: dict[str, object] | None = None

    def __call__(self, **kwargs: object) -> FakeZAIClient:
        self.kwargs = kwargs
        return FakeZAIClient(self.completions)


class _BodyRecorder:
    """An `httpx.MockTransport` handler that keeps each request's JSON body — what the API sees."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "created": 1,
                "model": "glm-5.2",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )


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
async def test_inference_params_reach_sdk_call(monkeypatch: pytest.MonkeyPatch) -> None:
    completions = FakeCompletions()
    factory = FakeZAIClientFactory(completions)
    monkeypatch.setattr(zai_client_module, "ZaiClient", factory)
    client = ZAIConfig(
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
    ).create()

    await client(
        messages=[ModelRequest([TextInput("hello")])],
        context=make_call_context(),
        tools=[],
        response_schema=None,
        serializer=PydanticSerializer(),
    )

    assert completions.kwargs == {
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
        "timeout": 30.0,
        "watermark_enabled": False,
        "tool_stream": True,
        "reasoning_effort": "high",
    }

    assert factory.kwargs == {"disable_token_cache": True, "max_retries": 3}


@pytest.mark.asyncio
async def test_unset_params_are_omitted_and_extra_body_is_forwarded_unshadowed(monkeypatch: pytest.MonkeyPatch) -> None:
    completions = FakeCompletions()
    factory = FakeZAIClientFactory(completions)
    monkeypatch.setattr(zai_client_module, "ZaiClient", factory)
    client = ZAIConfig(
        model="glm-5.2",
        thinking=True,
        extra_body={"thinking": {"type": "disabled"}, "request_id": "abc"},
    ).create()

    await client(
        messages=[ModelRequest([TextInput("hello")])],
        context=make_call_context(),
        tools=[],
        response_schema=None,
        serializer=PydanticSerializer(),
    )

    assert completions.kwargs == {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "thinking": {"type": "enabled"},
        # Forwarded for the SDK to merge into the body, less the key a typed option already sets.
        "extra_body": {"request_id": "abc"},
    }


@pytest.mark.asyncio
async def test_extra_body_keys_reach_the_request_body_without_binding_to_create() -> None:
    # `extra_body` is the SDK's escape hatch: its keys join the request JSON without being
    # parameters of `create`, so a field the SDK does not know yet still reaches the API.
    recorder = _BodyRecorder()
    with httpx.Client(transport=httpx.MockTransport(recorder)) as http_client:
        config = ZAIConfig(
            model="glm-5.2", api_key="id.secret", http_client=http_client, extra_body={"future_param": {"on": True}}
        )
        await _ask(config)

    assert recorder.bodies == [IsPartialDict({"future_param": {"on": True}})]


@pytest.mark.asyncio
async def test_a_typed_option_wins_over_the_same_key_in_extra_body() -> None:
    # The user guide's precedence; the SDK merges `extra_body` last, so the config has to keep it.
    recorder = _BodyRecorder()
    with httpx.Client(transport=httpx.MockTransport(recorder)) as http_client:
        config = ZAIConfig(
            model="glm-5.2",
            api_key="id.secret",
            http_client=http_client,
            thinking=True,
            extra_body={"thinking": {"type": "disabled"}},
        )
        await _ask(config)

    assert recorder.bodies == [IsPartialDict({"thinking": {"type": "enabled"}})]


@pytest.mark.asyncio
async def test_the_agents_tools_win_over_tools_in_extra_body() -> None:
    recorder = _BodyRecorder()
    with httpx.Client(transport=httpx.MockTransport(recorder)) as http_client:
        config = ZAIConfig(
            model="glm-5.2", api_key="id.secret", http_client=http_client, extra_body={"tools": [], "future_param": 1}
        )
        await _ask(config, tools=[make_tool().schema])

    assert recorder.bodies == [
        IsPartialDict({"tools": [IsPartialDict({"type": "function"})], "future_param": 1}),
    ]


@pytest.mark.asyncio
async def test_thinking_false_maps_to_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    completions = FakeCompletions()
    monkeypatch.setattr(zai_client_module, "ZaiClient", FakeZAIClientFactory(completions))
    client = ZAIConfig(model="glm-5.2", thinking=False).create()

    await client(
        messages=[ModelRequest([TextInput("hello")])],
        context=make_call_context(),
        tools=[],
        response_schema=None,
        serializer=PydanticSerializer(),
    )

    assert completions.kwargs == {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "thinking": {"type": "disabled"},
    }


@pytest.mark.asyncio
async def test_client_connection_params_reach_sdk_client(monkeypatch: pytest.MonkeyPatch) -> None:
    completions = FakeCompletions()
    factory = FakeZAIClientFactory(completions)
    with httpx.Client() as http_client:
        monkeypatch.setattr(zai_client_module, "ZaiClient", factory)
        client = ZAIConfig(
            model="glm-5.2",
            api_key="key",
            base_url="https://example.test/api/paas/v4/",
            timeout=12.0,
            max_retries=5,
            http_client=http_client,
            custom_headers={"x-test": "1"},
            disable_token_cache=False,
            source_channel="ag2-test",
        ).create()

        await client(
            messages=[ModelRequest([TextInput("hello")])],
            context=make_call_context(),
            tools=[],
            response_schema=None,
            serializer=PydanticSerializer(),
        )

        assert factory.kwargs == {
            "api_key": "key",
            "base_url": "https://example.test/api/paas/v4/",
            "timeout": 12.0,
            "max_retries": 5,
            "http_client": http_client,
            "custom_headers": {"x-test": "1"},
            "disable_token_cache": False,
            "source_channel": "ag2-test",
        }


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
async def test_sdk_only_receives_kwargs_it_accepts(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: every generation param ZAIConfig exposes must bind against the
    # real zai-sdk Completions.create signature, or the SDK raises TypeError at runtime.
    completions = FakeCompletions()
    factory = FakeZAIClientFactory(completions)
    monkeypatch.setattr(zai_client_module, "ZaiClient", factory)
    client = ZAIConfig(
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
    ).create()

    await client(
        messages=[ModelRequest([TextInput("hello")])],
        context=make_call_context(),
        tools=[],
        response_schema=None,
        serializer=PydanticSerializer(),
    )

    assert completions.kwargs is not None
    inspect.signature(RealCompletions.create).bind_partial(object(), **completions.kwargs)
