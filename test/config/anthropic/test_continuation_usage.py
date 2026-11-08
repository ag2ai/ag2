# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx2
import pytest
from dirty_equals import IsPartialDict

from ag2 import Agent, MemoryStream
from ag2.config.anthropic import AnthropicConfig
from ag2.events import ModelResponse, Usage, UsageEvent


def _message(index: int, paused: bool) -> dict[str, Any]:
    return {
        "id": f"msg_{index}",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [{"type": "text", "text": f"response {index}"}],
        "stop_reason": "pause_turn" if paused else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10 * index,
            "output_tokens": 2 * index,
            "cache_creation_input_tokens": 3 * index,
            "cache_read_input_tokens": 4 * index,
            "output_tokens_details": {"thinking_tokens": index},
        },
    }


def _streaming_response(message: dict[str, Any]) -> httpx2.Response:
    usage = message["usage"]
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                **message,
                "content": [],
                "stop_reason": None,
                "usage": {**usage, "output_tokens": 1, "output_tokens_details": {"thinking_tokens": 0}},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": message["content"][0]["text"]},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
            "usage": {"output_tokens": usage["output_tokens"], "output_tokens_details": usage["output_tokens_details"]},
        },
        {"type": "message_stop"},
    ]
    body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("pauses", [0, 1, 2, 6], ids=["no-pause", "one-pause", "two-pauses", "continuation-limit"])
async def test_usage_includes_every_continuation(streaming: bool, pauses: int) -> None:
    """Use the real SDK's HTTP seam: TestConfig cannot exercise provider continuations."""
    requests: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        index = len(requests)
        response = _message(index, index <= pauses)
        responses.append(response)
        return _streaming_response(response) if streaming else httpx2.Response(200, json=response)

    stream = MemoryStream()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http_client:
        agent = Agent(
            "assistant",
            config=AnthropicConfig(
                model="claude-haiku-4-5",
                api_key="test",
                streaming=streaming,
                http_client=http_client,
                prompt_caching=False,
            ),
        )
        reply = await agent.ask("hello", stream=stream)

    count = min(pauses + 1, 6)
    assert len(requests) == count
    for index, request in enumerate(requests[1:]):
        assert request == IsPartialDict({
            "messages": [
                *requests[index]["messages"],
                {"role": "assistant", "content": responses[index]["content"]},
            ],
        })
    assert reply.body == f"response {count}"
    events = await stream.history.get_events()
    [response] = [event for event in events if isinstance(event, ModelResponse)]
    assert response.finish_reason == ("pause_turn" if pauses >= 6 else "end_turn")
    assert response.response_id == f"msg_{count}"
    total = count * (count + 1) // 2
    expected = Usage(
        prompt_tokens=10 * total,
        completion_tokens=2 * total,
        total_tokens=12 * total,
        cache_creation_input_tokens=3 * total,
        cache_read_input_tokens=4 * total,
        thinking_tokens=total,
    )
    assert response.usage == expected
    assert [event.usage for event in events if isinstance(event, UsageEvent)] == [expected]
    assert (await reply.usage()).total == expected
