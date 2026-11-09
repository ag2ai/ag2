# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fast_depends.pydantic.serializer import PydanticSerializer

from ag2.config.dashscope import DashScopeClient
from ag2.events import ModelRequest
from ag2.events.input_events import TextInput


def _chunk(tool_calls: list[dict[str, Any]]) -> SimpleNamespace:
    message = {"role": "assistant", "content": "", "tool_calls": tool_calls}
    choice = SimpleNamespace(message=message, get=lambda key, default=None: None)
    return SimpleNamespace(
        status_code=200,
        usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        output=SimpleNamespace(choices=[choice]),
    )


async def _run(chunks: list[SimpleNamespace]) -> list[tuple[str, str, str]]:
    async def stream() -> AsyncIterator[SimpleNamespace]:
        for chunk in chunks:
            yield chunk

    context = AsyncMock()
    context.prompt = []
    client = DashScopeClient("qwen-plus", api_key="k", streaming=True)
    with patch(
        "ag2.config.dashscope.dashscope_client.AioMultiModalConversation.call", AsyncMock(return_value=stream())
    ):
        response = await client(
            messages=[ModelRequest([TextInput("hi")])],
            context=context,
            tools=(),
            response_schema=None,
            serializer=PydanticSerializer,
        )
    return [(c.id, c.name, c.arguments) for c in response.tool_calls.calls]


@pytest.mark.asyncio
async def test_fragments_with_index_are_merged() -> None:
    calls = await _run([
        _chunk([{"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": '{"city": '}}]),
        _chunk([{"index": 0, "function": {"arguments": '"Paris"}'}}]),
    ])

    assert calls == [("call_1", "get_weather", '{"city": "Paris"}')]


@pytest.mark.asyncio
async def test_empty_id_and_none_name_in_later_fragments_do_not_split_the_call() -> None:
    calls = await _run([
        _chunk([{"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": '{"a":'}}]),
        _chunk([{"index": 0, "id": "", "function": {"name": None, "arguments": " 1}"}}]),
    ])

    assert calls == [("call_1", "get_weather", '{"a": 1}')]


@pytest.mark.asyncio
async def test_interleaved_indexes_yield_one_call_each_in_order() -> None:
    calls = await _run([
        _chunk([{"index": 0, "id": "a", "function": {"name": "f", "arguments": '{"x":'}}]),
        _chunk([{"index": 1, "id": "b", "function": {"name": "g", "arguments": '{"y":'}}]),
        _chunk([{"index": 0, "function": {"arguments": "1}"}}]),
        _chunk([{"index": 1, "function": {"arguments": "2}"}}]),
    ])

    assert calls == [("a", "f", '{"x":1}'), ("b", "g", '{"y":2}')]


@pytest.mark.asyncio
async def test_whole_call_in_one_chunk_without_index() -> None:
    calls = await _run([_chunk([{"id": "a", "function": {"name": "f", "arguments": '{"x": 1}'}}])])

    assert calls == [("a", "f", '{"x": 1}')]


@pytest.mark.asyncio
async def test_fallback_keys_do_not_collide_across_chunks() -> None:
    calls = await _run([
        _chunk([
            {"id": "a", "function": {"name": "f", "arguments": "{}"}},
            {"id": "b", "function": {"name": "g", "arguments": "{}"}},
        ]),
        _chunk([{"id": "c", "function": {"name": "h", "arguments": "{}"}}]),
    ])

    assert calls == [("a", "f", "{}"), ("b", "g", "{}"), ("c", "h", "{}")]
