# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""What the chat client does with a completion it cannot read as a normal turn.

The API can answer with an empty `choices` list — a content filter does — and the
turn has still spent tokens, so the client owes the agent loop a `ModelResponse`
either way. A tool call of a kind ag2 never asked for is the opposite case: there
is no reading of it, so it is refused by name.
"""

import json
from typing import Any

import httpx2
import pytest
from fast_depends.pydantic import PydanticSerializer

from ag2 import Context, MemoryStream
from ag2.config.openai import OpenAIClient
from ag2.events import ModelRequest, ModelResponse, TextInput
from ag2.exceptions import UnsupportedToolError

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
        serializer=PydanticSerializer(),
    )

    assert isinstance(result, ModelResponse)
    assert result.message is None
    assert len(result.tool_calls) == 0
    assert result.finish_reason is None
    # The tokens were spent, so they are reported.
    assert result.usage.prompt_tokens == 7
    assert result.model == "gpt-4o"
    assert result.response_id == "chatcmpl_1"


_CUSTOM_TOOL_CALL: dict[str, Any] = {
    **_EMPTY,
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "custom",
                        "custom": {"name": "grep", "input": "-rn needle"},
                    }
                ],
            },
        }
    ],
}


@pytest.mark.asyncio
async def test_a_tool_call_of_a_kind_ag2_never_sent_is_refused_by_name() -> None:
    """ag2 sends function tools only, so a custom call has no `function` to read."""
    with pytest.raises(UnsupportedToolError, match="custom"):
        await _client(_CUSTOM_TOOL_CALL)(
            messages=[ModelRequest([TextInput("hi")])],
            context=Context(stream=MemoryStream()),
            tools=[],
            response_schema=None,
            serializer=PydanticSerializer(),
        )
