# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""`http_client` reaches the SDK untouched, whichever HTTP package built it.

ag2 forwards the client and adds no policy of its own. The public parameter is
typed `httpx2.AsyncClient` because that is how `openai>=3` types it, and the type
is the whole of ag2's opinion: a static checker rejects a legacy client here for
the same reason it rejects one at `AsyncOpenAI(...)`.

What happens at runtime is the SDK's call, not ours. `openai>=3` depends on
`httpx2` alone but still accepts a legacy `httpx` client — it detects one through
`sys.modules` and normalizes the URL, timeout, and response types around it — and
documents that path as a migration aid that may be discontinued. So a legacy
client works today, and the day it stops working the SDK will say so. The legacy
cases below are the guard on ag2 staying out of that: they fail if ag2 ever
starts rejecting or warning about a client the SDK itself would have accepted.
"""

import json

import httpx
import httpx2
import pytest
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config import OpenAIConfig, OpenAIResponsesConfig
from ag2.events import ModelRequest, TextInput

_COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

_RESPONSE = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "model": "gpt-4o",
    "status": "completed",
    "output": [
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": {
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}


def _modern_client(captured: dict[str, object], payload: dict[str, object]) -> httpx2.AsyncClient:
    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json=payload)

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


def _legacy_client(captured: dict[str, object], payload: dict[str, object]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=payload)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _ask(config: OpenAIConfig | OpenAIResponsesConfig) -> None:
    client = config.create()
    await client(
        messages=[ModelRequest([TextInput("capital of France?")])],
        context=Context(stream=MemoryStream()),
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )


_COMPLETION_BODY = IsPartialDict({
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": ""},
        {"role": "user", "content": "capital of France?"},
    ],
})


@pytest.mark.asyncio
class TestHttpx2Client:
    async def test_completions(self) -> None:
        captured: dict[str, object] = {}

        await _ask(OpenAIConfig(model="gpt-4o", api_key="test", http_client=_modern_client(captured, _COMPLETION)))

        assert captured["body"] == _COMPLETION_BODY

    async def test_responses(self) -> None:
        captured: dict[str, object] = {}

        await _ask(
            OpenAIResponsesConfig(model="gpt-4o", api_key="test", http_client=_modern_client(captured, _RESPONSE))
        )

        assert captured["body"] == IsPartialDict({"model": "gpt-4o"})


@pytest.mark.asyncio
class TestLegacyHttpxClient:
    """Accepted for as long as the SDK accepts it — ag2 neither blocks nor warns."""

    async def test_completions(self) -> None:
        captured: dict[str, object] = {}

        await _ask(OpenAIConfig(model="gpt-4o", api_key="test", http_client=_legacy_client(captured, _COMPLETION)))

        assert captured["body"] == _COMPLETION_BODY

    async def test_responses(self) -> None:
        captured: dict[str, object] = {}

        await _ask(
            OpenAIResponsesConfig(model="gpt-4o", api_key="test", http_client=_legacy_client(captured, _RESPONSE))
        )

        assert captured["body"] == IsPartialDict({"model": "gpt-4o"})
