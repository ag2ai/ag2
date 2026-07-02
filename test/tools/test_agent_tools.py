# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import asdict
from unittest.mock import MagicMock

import pytest
from dirty_equals import IsPartialDict
from pydantic import BaseModel

from ag2 import Agent, ToolResult, tool
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent
from ag2.testing import TestConfig

DEFAULT_SCHEMA = {
    "function": {
        "description": "Tool description.",
        "name": "my_tool",
        "parameters": {
            "properties": {
                "a": {
                    "title": "A",
                    "type": "string",
                },
                "b": {
                    "title": "B",
                    "type": "integer",
                },
            },
            "required": [
                "a",
                "b",
            ],
            "type": "object",
        },
    },
    "type": "function",
    "defer_loading": False,
}


def test_agent_with_function(mock: MagicMock) -> None:
    def my_tool(a: str, b: int) -> str:
        """Tool description."""
        return ""

    agent = Agent("", config=mock, tools=[my_tool])

    assert asdict(list(agent.tools)[0].schema) == DEFAULT_SCHEMA


def test_agent_with_tool(mock: MagicMock) -> None:
    @tool
    def my_tool(a: str, b: int) -> str:
        """Tool description."""
        return ""

    agent = Agent("", config=mock, tools=[my_tool])

    assert asdict(list(agent.tools)[0].schema) == DEFAULT_SCHEMA


def test_agent_with_tool_decorator(mock: MagicMock) -> None:
    agent = Agent("", config=mock)

    @agent.tool
    def my_tool(a: str, b: int) -> str:
        """Tool description."""
        return ""

    assert asdict(list(agent.tools)[0].schema) == DEFAULT_SCHEMA


def test_agent_with_tool_decorator_options_override(mock: MagicMock) -> None:
    agent = Agent("", config=mock)

    @agent.tool(name="another_name", description="another_description")
    def my_tool(a: str, b: int) -> str:
        """Tool description."""
        return ""

    assert asdict(list(agent.tools)[0].schema) == {
        "function": IsPartialDict({
            "description": "another_description",
            "name": "another_name",
        }),
        "type": "function",
        "defer_loading": False,
    }


@pytest.mark.asyncio()
async def test_final_tool() -> None:
    class DataModel(BaseModel):
        data: str

    def my_tool() -> ToolResult:
        return ToolResult({"data": "result"}, final=True)

    agent = Agent(
        "",
        tools=[my_tool],
        config=TestConfig(ToolCallEvent(name="my_tool")),
    )

    result = await agent.ask("Hi!")
    assert DataModel.model_validate_json(result.body) == DataModel(data="result")


@pytest.mark.asyncio()
async def test_concurrent_tool_execution() -> None:
    """Test that multiple tools are executed concurrently, not sequentially.

    Each tool blocks on a shared barrier that only releases once all three
    have started. Concurrent execution releases it immediately; sequential
    execution leaves the earlier tools waiting, so their bounded wait expires
    and they never record completion — failing the ``finished`` assertion
    fast instead of hanging.
    """
    started = 0
    all_started = asyncio.Event()
    finished: list[str] = []

    async def _run(name: str) -> str:
        nonlocal started
        started += 1
        if started == 3:
            all_started.set()
        try:
            await asyncio.wait_for(all_started.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            return "not-concurrent"
        finished.append(name)
        return f"result_{name}"

    async def slow_tool_a() -> str:
        return await _run("a")

    async def slow_tool_b() -> str:
        return await _run("b")

    async def slow_tool_c() -> str:
        return await _run("c")

    agent = Agent(
        "test_agent",
        tools=[slow_tool_a, slow_tool_b, slow_tool_c],
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent(
                    calls=[
                        ToolCallEvent(name="slow_tool_a"),
                        ToolCallEvent(name="slow_tool_b"),
                        ToolCallEvent(name="slow_tool_c"),
                    ]
                )
            ),
            "result",
        ),
    )

    result = await agent.ask("Execute all tools")
    assert result.body == "result"
    assert started == 3
    assert sorted(finished) == ["a", "b", "c"]
