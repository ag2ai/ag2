# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsStr

from ag2 import Agent, Context, MemoryStream
from ag2.events import ToolResult, ToolResultEvent, ToolResultsEvent, Usage
from ag2.exceptions import HumanInputNotProvidedError
from ag2.history import HUMAN_INPUT_ABANDONED_TOOL_RESULT
from ag2.testing import TestConfig
from test.config.ollama._helpers import FakeOllama, ask, make_chunk


def _calls(first: int, count: int) -> list[tuple[str, dict[str, int]]]:
    return [(f"tool_{n}", {"value": n}) for n in range(first, first + count)]


def _expected(count: int) -> list[dict[str, object]]:
    return [
        {
            "id": IsStr(min_length=1),
            "type": "function",
            "function": {"name": f"tool_{n}", "arguments": f'{{"value": {n}}}'},
        }
        for n in range(count)
    ]


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

    assert result.tool_calls.to_api() == _expected(sum(chunk_sizes))
    assert len({call.id for call in result.tool_calls.calls}) == sum(chunk_sizes)
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

    assert result.tool_calls.to_api() == _expected(3)
    assert len({call.id for call in result.tool_calls.calls}) == 3
    assert result.content == "Running tools"


@pytest.mark.asyncio
async def test_streaming_tool_call_ids_are_unique_when_final_chunk_carries_calls(ollama: FakeOllama) -> None:
    ollama.chunks = [
        make_chunk(tool_calls=_calls(0, 2)),
        make_chunk(tool_calls=_calls(2, 1), done=True),
    ]

    result, _ = await ask(ollama.config(streaming=True))

    assert result.tool_calls.to_api() == _expected(3)
    assert len({call.id for call in result.tool_calls.calls}) == 3
    assert result.usage == Usage(prompt_tokens=4, completion_tokens=6, total_tokens=10)
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_non_streaming_tool_call_ids_are_unique(ollama: FakeOllama) -> None:
    ollama.chunks = [make_chunk(tool_calls=_calls(0, 3), done=True)]

    result, _ = await ask(ollama.config(streaming=False))

    assert result.tool_calls.to_api() == _expected(3)
    assert len({call.id for call in result.tool_calls.calls}) == 3
    assert ollama.body["stream"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_tool_call_ids_are_unique_across_responses(ollama: FakeOllama, streaming: bool) -> None:
    ollama.chunks = [make_chunk(tool_calls=_calls(0, 2), done=True)]
    config = ollama.config(streaming=streaming)

    first, _ = await ask(config)
    second, _ = await ask(config)

    calls = [*first.tool_calls.calls, *second.tool_calls.calls]
    assert len({call.id for call in calls}) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_later_unanswered_tool_call_is_not_hidden_by_an_earlier_result(
    ollama: FakeOllama, streaming: bool
) -> None:
    def finish() -> str:
        return "done"

    async def ask_human(ctx: Context) -> str:
        return await ctx.input("Confirm?")

    config = ollama.config(streaming=streaming)
    ollama.chunks = [make_chunk(tool_calls=[("finish", {})], done=True)]
    first, _ = await ask(config)
    ollama.chunks = [make_chunk(tool_calls=[("ask_human", {})], done=True)]
    second, _ = await ask(config)
    stream = MemoryStream()
    agent = Agent("", config=TestConfig(first, "completed"), tools=[finish, ask_human])

    await agent.ask("First turn", stream=stream)
    with pytest.raises(HumanInputNotProvidedError):
        await agent.ask("Second turn", stream=stream, config=TestConfig(second, raise_tool_errors=False))

    results = [
        result
        for event in await stream.history.get_events()
        if isinstance(event, ToolResultsEvent)
        for result in event.results
    ]
    assert results == [
        ToolResultEvent(parent_id=first.tool_calls.calls[0].id, name="finish", result=ToolResult("done")),
        ToolResultEvent(
            parent_id=second.tool_calls.calls[0].id,
            name="ask_human",
            result=ToolResult(HUMAN_INPUT_ABANDONED_TOOL_RESULT),
        ),
    ]
