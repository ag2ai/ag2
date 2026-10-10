# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls

from ag2.config.ollama.mappers import convert_messages
from ag2.events import (
    BaseEvent,
    ModelRequest,
    TextInput,
    ToolCallEvent,
    ToolNotFoundEvent,
    ToolResult,
    ToolResultEvent,
    ToolResultsEvent,
)
from ag2.exceptions import ToolNotFoundError
from test.config.ollama._helpers import FakeOllama, ask


def _round(*results: ToolResultEvent) -> list[BaseEvent]:
    return [ModelRequest([TextInput("go")]), ToolResultsEvent(results=list(results))]


def _result(name: str | None, text: str, call_id: str = "c") -> ToolResultEvent:
    return ToolResultEvent(parent_id=call_id, name=name, result=ToolResult(TextInput(text)))


def _tool_messages(ollama: FakeOllama) -> list[dict[str, object]]:
    return [m for m in ollama.body["messages"] if m["role"] == "tool"]


@pytest.mark.asyncio
async def test_single_result_names_its_tool(ollama: FakeOllama) -> None:
    await ask(ollama.config(), _round(_result("get_weather", "sunny")))

    assert _tool_messages(ollama) == [{"role": "tool", "content": "sunny", "tool_name": "get_weather"}]


@pytest.mark.asyncio
async def test_different_tools_in_one_turn_each_carry_their_name(ollama: FakeOllama) -> None:
    await ask(ollama.config(), _round(_result("get_weather", "sunny", "a"), _result("get_time", "noon", "b")))

    assert _tool_messages(ollama) == [
        {"role": "tool", "content": "sunny", "tool_name": "get_weather"},
        {"role": "tool", "content": "noon", "tool_name": "get_time"},
    ]


@pytest.mark.asyncio
async def test_same_tool_twice_keeps_call_order_and_names_both(ollama: FakeOllama) -> None:
    await ask(ollama.config(), _round(_result("get_weather", "Paris", "a"), _result("get_weather", "Tokyo", "b")))

    assert _tool_messages(ollama) == [
        {"role": "tool", "content": "Paris", "tool_name": "get_weather"},
        {"role": "tool", "content": "Tokyo", "tool_name": "get_weather"},
    ]


@pytest.mark.asyncio
async def test_result_without_a_name_sends_no_tool_name(ollama: FakeOllama) -> None:
    await ask(ollama.config(), _round(_result(None, "sunny")))

    assert _tool_messages(ollama) == [{"role": "tool", "content": "sunny"}]


@pytest.mark.asyncio
async def test_error_for_hallucinated_tool_names_that_tool(ollama: FakeOllama) -> None:
    call = ToolCallEvent(id="c", name="ghost_tool")

    await ask(ollama.config(), _round(ToolNotFoundEvent.from_call(call, ToolNotFoundError("ghost_tool"))))

    assert _tool_messages(ollama) == [IsPartialDict({"role": "tool", "tool_name": "ghost_tool"})]


def test_multi_part_result_names_its_tool() -> None:
    # Mapper-level: the SDK rejects list `content`, so a multi-part result cannot cross the seam.
    result = ToolResultEvent(parent_id="c", name="get_weather", result=ToolResult(TextInput("a"), TextInput("b")))

    [message] = convert_messages([], _round(result)[1:], SerializerCls)

    assert message == {
        "role": "tool",
        "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        "tool_name": "get_weather",
    }
