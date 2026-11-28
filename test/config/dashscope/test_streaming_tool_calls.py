# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ag2.config.dashscope.dashscope_client import DashScopeClient


class _Dict(dict):  # mimics the SDK's DictMixin: keys readable as attributes
    def __getattr__(self, name: str) -> Any:
        return self[name]


def _chunk(tool_calls: list[dict[str, Any]], finish_reason: str | None = None) -> SimpleNamespace:
    message = _Dict(role="assistant", content="", tool_calls=tool_calls)
    choice = _Dict(message=message, finish_reason=finish_reason)
    return SimpleNamespace(status_code=200, usage=None, output=SimpleNamespace(choices=[choice]))


async def _stream(chunks: list[SimpleNamespace]) -> Any:
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_streaming_tool_call_fragments_are_accumulated_by_index() -> None:
    chunks = [
        _chunk([
            {"index": 0, "id": "call-1", "type": "function", "function": {"name": "get_weather", "arguments": ""}}
        ]),
        _chunk([{"index": 0, "function": {"arguments": '{"city": '}}]),
        _chunk([{"index": 0, "function": {"arguments": '"Paris"}'}}], finish_reason="tool_calls"),
    ]
    context = MagicMock()
    context.prompt = []
    context.send = AsyncMock()

    client = DashScopeClient(model="qwen-plus", api_key="k", base_url="http://x", streaming=True)
    with patch(
        "ag2.config.dashscope.dashscope_client.AioMultiModalConversation.call",
        new=AsyncMock(return_value=_stream(chunks)),
    ):
        response = await client._call_streaming([], {}, context)

    assert [(c.id, c.name, c.arguments) for c in response.tool_calls.calls] == [
        ("call-1", "get_weather", '{"city": "Paris"}')
    ]
