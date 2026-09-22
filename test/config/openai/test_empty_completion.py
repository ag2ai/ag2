# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A completion with no choices is still a response.

The Chat Completions API can answer with an empty `choices` list — a content filter
does — and the turn has still spent tokens. The client owes the agent loop a
`ModelResponse` either way, because everything downstream reads one.
"""

import json
from typing import Any

import httpx2
import pytest
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config.openai import OpenAIClient
from ag2.events import ModelRequest, ModelResponse, TextInput

_EMPTY: dict[str, Any] = {
    "id": "chatcmpl_1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o",
    "choices": [],
    "usage": {"prompt_tokens": 7, "completion_tokens": 0, "total_tokens": 7},
}


def _client(payload: dict[str, Any]) -> OpenAIClient:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=json.dumps(payload).encode())

    return OpenAIClient(
        api_key="test",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        create_options={"model": "gpt-4o"},
    )


@pytest.mark.asyncio
async def test_a_completion_with_no_choices_answers_a_response() -> None:
    result = await _client(_EMPTY)(
        messages=[ModelRequest([TextInput("hi")])],
        context=Context(stream=MemoryStream()),
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )

    assert isinstance(result, ModelResponse)
    assert result.message is None
    assert len(result.tool_calls) == 0
    assert result.finish_reason is None
    # The tokens were spent, so they are reported.
    assert result.usage.prompt_tokens == 7
    assert result.model == "gpt-4o"
    assert result.response_id == "chatcmpl_1"
