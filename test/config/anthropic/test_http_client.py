# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""`http_client` reaches the SDK untouched, and the SDK decides what it takes.

ag2 forwards the client and adds no policy of its own. The public parameter is
typed `httpx2.AsyncClient` because that is how `anthropic>=1` types it, and the
type is the whole of ag2's opinion: a static checker rejects a legacy `httpx`
client here for the same reason it rejects one at `AsyncAnthropic(...)`.

Unlike `openai>=3`, the `anthropic` SDK keeps no compatibility path for a legacy
client — it checks the type and raises. The legacy case below pins that this stays
the SDK's `TypeError`, raised where the caller can read which package it wants,
rather than something ag2 wraps, warns about, or quietly adapts.
"""

import json

import httpx
import httpx2
import pytest
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config.anthropic import AnthropicConfig
from ag2.events import ModelRequest, TextInput

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


async def _ask(config: AnthropicConfig) -> None:
    await config.create()(
        messages=[ModelRequest([TextInput("hi")])],
        context=Context(stream=MemoryStream()),
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )


@pytest.mark.asyncio
async def test_the_clients_own_transport_serves_the_request() -> None:
    seen: list[tuple[str, bytes]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append((str(request.url), request.content))
        return httpx2.Response(200, json=_MESSAGE)

    config = AnthropicConfig(
        model="claude-haiku-4-5",
        api_key="secret-key",  # pragma: allowlist secret
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )

    await _ask(config)

    [(url, body)] = seen
    assert url == "https://api.anthropic.com/v1/messages"
    assert b'"claude-haiku-4-5"' in body


@pytest.mark.asyncio
async def test_the_client_streams_server_sent_events() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in _STREAM_EVENTS)
        return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})

    config = AnthropicConfig(
        model="claude-haiku-4-5",
        api_key="test",
        streaming=True,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )

    await _ask(config)


def test_the_client_is_handed_over_as_is() -> None:
    """Not rebuilt or wrapped, so the SDK reads the caller's own settings off it."""
    http_client = httpx2.AsyncClient(timeout=httpx2.Timeout(33.0, connect=5.0))

    config = AnthropicConfig(model="claude-haiku-4-5", api_key="test", http_client=http_client)

    assert config.create()._client._client is http_client


def test_the_files_client_takes_the_same_client() -> None:
    http_client = httpx2.AsyncClient()

    config = AnthropicConfig(model="claude-haiku-4-5", api_key="test", http_client=http_client)

    assert config.create_files_client()._client._client is http_client


def test_a_legacy_httpx_client_is_refused_by_the_sdk() -> None:
    """ag2 neither adapts it nor preempts the error — the SDK names the package."""
    config = AnthropicConfig(model="claude-haiku-4-5", api_key="test", http_client=httpx.AsyncClient())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="httpx2.AsyncClient"):
        config.create()


def test_no_http_client_is_left_to_the_sdk() -> None:
    assert AnthropicConfig(model="claude-haiku-4-5", api_key="test").create() is not None
