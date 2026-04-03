# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import asdict
from unittest.mock import MagicMock

import pytest
from dirty_equals import IsPartialDict
from pydantic import BaseModel

from autogen.beta import Agent, ToolResult, tool
from autogen.beta.events import ModelResponse, ToolCallEvent, ToolCallsEvent
from autogen.beta.testing import TestConfig

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
    }


@pytest.mark.asyncio()
async def test_final_tool() -> None:
    class DataModel(BaseModel):
        data: str

    def my_tool() -> ToolResult[DataModel]:
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
    """Test that multiple tools are executed concurrently, not sequentially."""
    execution_order: list[str] = []

    async def slow_tool_a() -> str:
        execution_order.append("a_start")
        await asyncio.sleep(0.005)  # Simulate slow operation
        execution_order.append("a_end")
        return "result_a"

    async def slow_tool_b() -> str:
        execution_order.append("b_start")
        await asyncio.sleep(0.005)  # Simulate slow operation
        execution_order.append("b_end")
        return "result_b"

    async def slow_tool_c() -> str:
        execution_order.append("c_start")
        await asyncio.sleep(0.005)  # Simulate slow operation
        execution_order.append("c_end")
        return "result_c"

    # Create an agent with multiple slow tools
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

    # Trigger tool calls - they should execute concurrently
    result = await agent.ask("Execute all tools")
    assert result.body == "result"

    # Verify all tools were executed
    assert "a_start" in execution_order
    assert "b_start" in execution_order
    assert "c_start" in execution_order
    assert "a_end" in execution_order
    assert "b_end" in execution_order
    assert "c_end" in execution_order

    # Verify tools started before any finished (concurrent execution)
    # All starts should happen before all ends
    start_indices = [
        execution_order.index("a_start"),
        execution_order.index("b_start"),
        execution_order.index("c_start"),
    ]
    end_indices = [
        execution_order.index("a_end"),
        execution_order.index("b_end"),
        execution_order.index("c_end"),
    ]

    # The last start should happen before the first end for true concurrency
    assert max(start_indices) < min(end_indices), "Tools should start executing concurrently"
