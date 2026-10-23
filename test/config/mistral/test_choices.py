# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx2
import pytest
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config.mistral import MistralConfig
from ag2.events import (
    BaseEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    Usage,
)


async def _complete(
    choices: list[dict[str, Any]], streaming: bool, *, reverse_stream: bool = False
) -> tuple[ModelResponse, list[BaseEvent]]:
    usage = {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
    payload = {
        "id": "completion-1",
        "object": "chat.completion",
        "created": 0,
        "model": "mistral-small-latest",
        "choices": choices,
        "usage": usage,
    }
    # Choices arrive in separate chunks, possibly out of order. The final
    # chunk carries aggregate usage without a choice.
    chunks = [
        {
            **payload,
            "object": "chat.completion.chunk",
            "usage": None,
            "choices": [{"index": c["index"], "delta": c["message"], "finish_reason": c["finish_reason"]}],
        }
        for c in (reversed(choices) if reverse_stream else choices)
    ]
    chunks.append({**payload, "object": "chat.completion.chunk", "choices": []})

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert json.loads(request.content) == IsPartialDict({"n": 2, "stream": streaming})
        if streaming:
            body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})
        return httpx2.Response(200, json=payload)

    stream = MemoryStream(persist_all=True)
    assert SerializerCls is not None
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http_client:
        client = MistralConfig(
            model="mistral-small-latest",
            api_key="test",
            async_client=http_client,
            streaming=streaming,
            n=2,
        ).create()
        response = await client(
            messages=[ModelRequest([TextInput("hello")])],
            context=Context(stream=stream),
            tools=[],
            response_schema=None,
            serializer=SerializerCls,
        )

    assert response.usage == Usage(prompt_tokens=10, completion_tokens=8, total_tokens=18)
    return response, list(await stream.history.get_events())


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_only_first_choice_text_and_reasoning_are_emitted(streaming: bool) -> None:
    choices = [
        {
            "index": index,
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": [{"type": "text", "text": f"reasoning {index}"}]},
                    {"type": "text", "text": f"answer {index}"},
                ],
            },
            "finish_reason": "stop" if index == 0 else "length",
        }
        for index in range(2)
    ]

    response, events = await _complete(choices, streaming)

    assert response.message == ModelMessage("answer 0")
    assert response.finish_reason == "stop"
    assert events == [
        ModelReasoning("reasoning 0"),
        *([ModelMessageChunk("answer 0")] if streaming else []),
        ModelMessage("answer 0"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_tool_calls_from_different_choices_do_not_share_an_accumulator(streaming: bool) -> None:
    choices = [
        {
            "index": index,
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {"name": "search", "arguments": f'{{"query":"choice {index}"}}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
        for index in range(2)
    ]

    response, events = await _complete(choices, streaming, reverse_stream=True)

    assert response.tool_calls == ToolCallsEvent([
        ToolCallEvent(id="call_0", name="search", arguments='{"query":"choice 0"}')
    ])
    assert response.message is None
    assert response.finish_reason == "tool_calls"
    assert events == []
