# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.events import ToolCallEvent, Usage
from test.config.ollama._helpers import FakeOllama, ask, make_chunk


def _calls(first: int, count: int) -> list[tuple[str, dict[str, int]]]:
    return [(f"tool_{n}", {"value": n}) for n in range(first, first + count)]


def _expected(count: int) -> list[ToolCallEvent]:
    return [ToolCallEvent(id=f"call_{n}", name=f"tool_{n}", arguments=f'{{"value": {n}}}') for n in range(count)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunk_sizes",
    [(2, 1), (2, 2), (4, 2), (3, 1), (1, 2), (1, 1), (2,), (0, 2, 0, 1), (1, 0, 0, 3)],
)
async def test_streaming_tool_call_ids_are_unique_across_chunks(
    ollama: FakeOllama, chunk_sizes: tuple[int, ...]
) -> None:
    firsts = [sum(chunk_sizes[:i]) for i in range(len(chunk_sizes))]
    ollama.chunks = [
        *(make_chunk(tool_calls=_calls(first, size)) for first, size in zip(firsts, chunk_sizes, strict=True)),
        make_chunk(done=True),
    ]

    result, _ = await ask(ollama.config(streaming=True))

    assert result.tool_calls.calls == _expected(sum(chunk_sizes))
    assert result.usage == Usage(prompt_tokens=4, completion_tokens=6, total_tokens=10)
    assert result.model == "m1"
    assert result.finish_reason == "stop"
    assert ollama.body["stream"] is True


@pytest.mark.asyncio
async def test_streaming_tool_call_ids_are_unique_alongside_text(ollama: FakeOllama) -> None:
    ollama.chunks = [
        make_chunk(content="Running ", tool_calls=_calls(0, 2)),
        make_chunk(content="tools", tool_calls=_calls(2, 1)),
        make_chunk(done=True),
    ]

    result, _ = await ask(ollama.config(streaming=True))

    assert result.tool_calls.calls == _expected(3)
    assert result.content == "Running tools"


@pytest.mark.asyncio
async def test_streaming_tool_call_ids_are_unique_when_final_chunk_carries_calls(ollama: FakeOllama) -> None:
    ollama.chunks = [
        make_chunk(tool_calls=_calls(0, 2)),
        make_chunk(tool_calls=_calls(2, 1), done=True),
    ]

    result, _ = await ask(ollama.config(streaming=True))

    assert result.tool_calls.calls == _expected(3)
    assert result.usage == Usage(prompt_tokens=4, completion_tokens=6, total_tokens=10)
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_non_streaming_tool_call_ids_are_unique(ollama: FakeOllama) -> None:
    ollama.chunks = [make_chunk(tool_calls=_calls(0, 3), done=True)]

    result, _ = await ask(ollama.config(streaming=False))

    assert result.tool_calls.calls == _expected(3)
    assert ollama.body["stream"] is False
