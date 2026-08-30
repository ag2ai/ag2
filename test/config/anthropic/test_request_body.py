# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""What actually goes out on the wire, pinned through an ``httpx2`` mock transport.

The `anthropic` 1.x floor moved two things AG2 depends on: the sampling
parameters left the method signature, and header names started matching
case-insensitively. Both are only observable in the request the SDK builds, so
they are asserted there rather than against AG2's own kwargs.
"""

import json

import httpx2
import pytest
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config.anthropic import AnthropicClient, AnthropicConfig
from ag2.events import ModelRequest, TextInput
from ag2.tools.builtin.mcp_server import MCPServerTool

_MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}

_STREAM_EVENTS = (
    {"type": "message_start", "message": {**_MESSAGE, "content": [], "stop_reason": None}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 1},
    },
    {"type": "message_stop"},
)


def _capturing_client(captured: dict[str, object], *, stream: bool = False) -> httpx2.AsyncClient:
    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        captured["headers"] = request.headers
        if not stream:
            return httpx2.Response(200, json=_MESSAGE)
        body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in _STREAM_EVENTS)
        return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


async def _ask(config: AnthropicConfig, **kwargs: object) -> None:
    await config.create()(
        messages=[ModelRequest([TextInput("hi")])],
        context=Context(stream=MemoryStream()),
        response_schema=None,
        serializer=SerializerCls,
        **{"tools": [], **kwargs},  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_sampling_fields_still_reach_the_api() -> None:
    captured: dict[str, object] = {}
    config = AnthropicConfig(
        model="claude-haiku-4-5",
        api_key="test",
        temperature=0.2,
        top_p=0.9,
        top_k=5,
        http_client=_capturing_client(captured),
    )

    await _ask(config)

    assert captured["body"] == IsPartialDict({"temperature": 0.2, "top_p": 0.9, "top_k": 5})


@pytest.mark.asyncio
async def test_a_zero_sampling_field_is_sent_not_dropped() -> None:
    """Zero is the value determinism is asked for with, and it is falsy."""
    captured: dict[str, object] = {}
    config = AnthropicConfig(
        model="claude-haiku-4-5",
        api_key="test",
        temperature=0,
        top_p=0,
        top_k=0,
        http_client=_capturing_client(captured),
    )

    await _ask(config)

    assert captured["body"] == IsPartialDict({"temperature": 0, "top_p": 0, "top_k": 0})


@pytest.mark.asyncio
async def test_sampling_fields_reach_the_api_when_streaming() -> None:
    captured: dict[str, object] = {}
    config = AnthropicConfig(
        model="claude-haiku-4-5",
        api_key="test",
        temperature=0.2,
        streaming=True,
        http_client=_capturing_client(captured, stream=True),
    )

    await _ask(config)

    assert captured["body"] == IsPartialDict({"temperature": 0.2, "stream": True})


@pytest.mark.asyncio
async def test_user_extra_body_wins_over_a_sampling_field() -> None:
    captured: dict[str, object] = {}
    config = AnthropicConfig(
        model="claude-haiku-4-5",
        api_key="test",
        temperature=0.2,
        extra_body={"temperature": 0.9},
        http_client=_capturing_client(captured),
    )

    await _ask(config)

    assert captured["body"] == IsPartialDict({"temperature": 0.9})


@pytest.mark.asyncio
async def test_sampling_fields_join_the_mcp_servers_in_the_extra_body(context: Context) -> None:
    """Both are folded into the one ``extra_body`` the client sends; neither displaces the other."""
    captured: dict[str, object] = {}
    config = AnthropicConfig(
        model="claude-haiku-4-5",
        api_key="test",
        temperature=0.2,
        http_client=_capturing_client(captured),
    )
    schemas = await MCPServerTool(server_url="https://mcp.example.com/x", server_label="x").schemas(context)

    await _ask(config, tools=schemas)

    assert captured["body"] == IsPartialDict({
        "temperature": 0.2,
        "mcp_servers": [{"type": "url", "url": "https://mcp.example.com/x", "name": "x"}],
    })


@pytest.mark.asyncio
async def test_no_sampling_fields_leaves_the_body_alone() -> None:
    captured: dict[str, object] = {}
    config = AnthropicConfig(model="claude-haiku-4-5", api_key="test", http_client=_capturing_client(captured))

    await _ask(config)

    body = captured["body"]
    assert isinstance(body, dict)
    assert not {"temperature", "top_p", "top_k"} & body.keys()


@pytest.mark.asyncio
async def test_a_users_default_beta_survives_the_per_request_pin() -> None:
    """1.x matches header names case-insensitively, so only one line goes out.

    AG2 sets the MCP pin per request while the user may set betas as defaults on
    the config. On the 0.x floor both lines were sent; now the per-request value
    replaces the default outright, so AG2 folds the user's betas into the value it
    writes rather than dropping them.
    """
    captured: dict[str, object] = {}
    config = AnthropicConfig(
        model="claude-haiku-4-5",
        api_key="test",
        default_headers={"Anthropic-Beta": "user-beta-2026-01-01"},
        http_client=_capturing_client(captured),
    )
    [mcp_schema] = await MCPServerTool(
        server_url="https://mcp.example.com/sse",
        server_label="example-mcp",
    ).schemas(Context(stream=MemoryStream()))

    await _ask(config, tools=[mcp_schema])

    headers = captured["headers"]
    assert isinstance(headers, httpx2.Headers)
    assert headers.get_list("anthropic-beta") == ["mcp-client-2025-11-20,user-beta-2026-01-01"]


@pytest.mark.asyncio
async def test_a_beta_the_user_already_set_is_not_repeated() -> None:
    captured: dict[str, object] = {}
    config = AnthropicConfig(
        model="claude-haiku-4-5",
        api_key="test",
        default_headers={"anthropic-beta": "mcp-client-2025-11-20"},
        http_client=_capturing_client(captured),
    )
    [mcp_schema] = await MCPServerTool(
        server_url="https://mcp.example.com/sse",
        server_label="example-mcp",
    ).schemas(Context(stream=MemoryStream()))

    await _ask(config, tools=[mcp_schema])

    headers = captured["headers"]
    assert isinstance(headers, httpx2.Headers)
    assert headers.get_list("anthropic-beta") == ["mcp-client-2025-11-20"]


@pytest.mark.asyncio
async def test_default_beta_header_survives_when_nothing_is_set_per_request() -> None:
    captured: dict[str, object] = {}
    config = AnthropicConfig(
        model="claude-haiku-4-5",
        api_key="test",
        default_headers={"Anthropic-Beta": "user-beta-2026-01-01"},
        http_client=_capturing_client(captured),
    )

    await _ask(config)

    headers = captured["headers"]
    assert isinstance(headers, httpx2.Headers)
    assert headers.get_list("anthropic-beta") == ["user-beta-2026-01-01"]


async def _ask_client(client: AnthropicClient) -> None:
    await client(
        messages=[ModelRequest([TextInput("hi")])],
        context=Context(stream=MemoryStream()),
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )


@pytest.mark.asyncio
async def test_a_direct_client_caller_gets_the_extra_body_route_not_a_type_error() -> None:
    """`AnthropicClient` is public and `CreateOptions` used to type these keys.

    Without its own handling the keys would reach `messages.create()`, which raises
    a bare `TypeError` on the 1.x floor — at request time, with nothing said about
    what to do instead.
    """
    captured: dict[str, object] = {}
    client = AnthropicClient(
        api_key="test",
        prompt_caching=False,
        http_client=_capturing_client(captured),
        create_options={"model": "claude-haiku-4-5", "max_tokens": 16, "temperature": 0.2},  # type: ignore[typeddict-unknown-key]
    )

    await _ask_client(client)

    assert captured["body"] == IsPartialDict({"temperature": 0.2})


@pytest.mark.asyncio
async def test_a_direct_caller_may_still_spell_a_sampling_field_none() -> None:
    """The pre-1.x ``CreateOptions`` typed these ``float | None``.

    A key left behind because its value happened to be ``None`` would reach
    ``messages.create()`` all the same, and the argument is what 1.x removed.
    """
    captured: dict[str, object] = {}
    client = AnthropicClient(
        api_key="test",
        prompt_caching=False,
        http_client=_capturing_client(captured),
        create_options={"model": "claude-haiku-4-5", "max_tokens": 16, "temperature": None},  # type: ignore[typeddict-unknown-key]
    )

    await _ask_client(client)

    body = captured["body"]
    assert isinstance(body, dict)
    assert "temperature" not in body


@pytest.mark.asyncio
async def test_a_direct_callers_extra_body_still_wins() -> None:
    captured: dict[str, object] = {}
    client = AnthropicClient(
        api_key="test",
        prompt_caching=False,
        http_client=_capturing_client(captured),
        create_options={"model": "claude-haiku-4-5", "max_tokens": 16, "top_k": 5},  # type: ignore[typeddict-unknown-key]
        extra_body={"top_k": 9},
    )

    await _ask_client(client)

    assert captured["body"] == IsPartialDict({"top_k": 9})


@pytest.mark.asyncio
async def test_a_direct_client_caller_without_sampling_leaves_the_body_alone() -> None:
    captured: dict[str, object] = {}
    client = AnthropicClient(
        api_key="test",
        prompt_caching=False,
        http_client=_capturing_client(captured),
        create_options={"model": "claude-haiku-4-5", "max_tokens": 16},
    )

    await _ask_client(client)

    body = captured["body"]
    assert isinstance(body, dict)
    assert not {"temperature", "top_p", "top_k"} & body.keys()
