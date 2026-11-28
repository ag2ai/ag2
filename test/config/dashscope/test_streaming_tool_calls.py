# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ag2.config.dashscope import DashScopeClient


def _chunk(tool_calls: list[dict[str, Any]], finish_reason: str | None = None) -> SimpleNamespace:
    choice = SimpleNamespace(
        finish_reason=finish_reason,
        message={"content": "", "tool_calls": tool_calls},
    )
    # DashScope's choices are dict-like; the client only uses `.get()` on them and attribute access on `.message`.
    choice_dict = MagicMock()
    choice_dict.get = lambda key, default=None: getattr(choice, key, default)
    choice_dict.message = choice.message
    return SimpleNamespace(
        status_code=200,
        usage=None,
        output=SimpleNamespace(choices=[choice_dict]),
    )


async def _stream(chunks: list[SimpleNamespace]) -> AsyncIterator[SimpleNamespace]:
    for c in chunks:
        yield c


def _tc(index: int, *, id: str = "", name: str = "", arguments: str = "") -> dict[str, Any]:
    return {"index": index, "id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


@pytest.mark.asyncio()
async def test_streamed_tool_call_fragments_are_merged() -> None:
    chunks = [
        _chunk([_tc(0, id="call_1", name="get_weather", arguments="")]),
        _chunk([_tc(0, arguments='{"city": ')]),
        _chunk([_tc(0, arguments='"Paris"}')], finish_reason="tool_calls"),
    ]
    client = DashScopeClient("qwen-plus", api_key="k", streaming=True)
    context = MagicMock()
    context.send = AsyncMock()

    with patch(
        "ag2.config.dashscope.dashscope_client.AioMultiModalConversation.call",
        new=AsyncMock(return_value=_stream(chunks)),
    ):
        response = await client._call_streaming([], {}, context)

    assert [(c.id, c.name, c.arguments) for c in response.tool_calls.calls] == [
        ("call_1", "get_weather", '{"city": "Paris"}')
    ]


@pytest.mark.asyncio()
async def test_streamed_parallel_tool_calls_are_kept_apart() -> None:
    chunks = [
        _chunk([_tc(0, id="call_1", name="a", arguments='{"x":')]),
        _chunk([_tc(0, arguments=" 1}"), _tc(1, id="call_2", name="b", arguments='{"y":')]),
        _chunk([_tc(1, arguments=" 2}")], finish_reason="tool_calls"),
    ]
    client = DashScopeClient("qwen-plus", api_key="k", streaming=True)
    context = MagicMock()
    context.send = AsyncMock()

    with patch(
        "ag2.config.dashscope.dashscope_client.AioMultiModalConversation.call",
        new=AsyncMock(return_value=_stream(chunks)),
    ):
        response = await client._call_streaming([], {}, context)

    assert [(c.id, c.name, c.arguments) for c in response.tool_calls.calls] == [
        ("call_1", "a", '{"x": 1}'),
        ("call_2", "b", '{"y": 2}'),
    ]
