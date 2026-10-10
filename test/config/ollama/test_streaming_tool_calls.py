# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
from fast_depends.use import SerializerCls
from ollama import AsyncClient, ChatResponse, Message

from ag2 import Context
from ag2.config.ollama import OllamaClient
from ag2.events import ModelRequest, TextInput
from ag2.stream import MemoryStream


async def response_chunks(chunk_sizes: tuple[int, ...]) -> AsyncIterator[ChatResponse]:
    call_number = 0
    for size in chunk_sizes:
        tool_calls = []
        for _ in range(size):
            tool_calls.append(
                Message.ToolCall(
                    function=Message.ToolCall.Function(name=f"tool_{call_number}", arguments={"value": call_number})
                )
            )
            call_number += 1
        yield ChatResponse(model="test-model", message=Message(role="assistant", tool_calls=tool_calls))
    yield ChatResponse(
        model="test-model",
        message=Message(role="assistant"),
        done=True,
        done_reason="stop",
        prompt_eval_count=4,
        eval_count=6,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_sizes", [(2, 1), (2, 2), (4, 2), (3, 1), (1, 2), (1, 1)])
async def test_streaming_tool_call_ids_are_unique_across_chunks(
    monkeypatch: pytest.MonkeyPatch, chunk_sizes: tuple[int, ...]
) -> None:
    chat = AsyncMock(return_value=response_chunks(chunk_sizes))
    monkeypatch.setattr(AsyncClient, "chat", chat)
    client = OllamaClient(model="test-model", streaming=True)
    assert SerializerCls is not None

    response = await client(
        messages=[ModelRequest([TextInput("Run the tools")])],
        context=Context(stream=MemoryStream()),
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )

    calls = response.tool_calls.calls
    expected_count = sum(chunk_sizes)
    assert len(calls) == expected_count
    assert len({call.id for call in calls}) == expected_count
    assert [call.name for call in calls] == [f"tool_{i}" for i in range(expected_count)]
    assert [json.loads(call.arguments) for call in calls] == [{"value": i} for i in range(expected_count)]
    assert response.usage.total_tokens == 10
    assert response.model == "test-model"
    assert response.finish_reason == "stop"
    chat.assert_awaited_once()
    assert chat.call_args.kwargs["stream"] is True
