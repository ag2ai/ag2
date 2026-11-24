# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Agent, Context, TextInput, tool
from ag2.events import ToolCallEvent, ToolResultEvent
from ag2.testing import TestConfig


class Multiplier:
    """Multiply an integer by a configured factor."""

    def __init__(self, factor: int) -> None:
        self.factor = factor

    def __call__(self, value: int) -> str:
        return str(value * self.factor)


class AsyncMultiplier:
    """Multiply an integer asynchronously by a configured factor."""

    def __init__(self, factor: int) -> None:
        self.factor = factor

    async def __call__(self, value: int) -> str:
        return str(value * self.factor)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", [None, "renamed"])
async def test_ordinary_function_keeps_dunder_name(name: str | None, context: Context) -> None:
    def add(value: int) -> str:
        """Return the value as text."""
        return str(value)

    add_tool = tool(add, name=name)
    [schema] = await add_tool.schemas(context)
    assert schema.function.name == (name or "add")


@pytest.mark.asyncio
@pytest.mark.parametrize("callable_type", [Multiplier, AsyncMultiplier])
@pytest.mark.parametrize("sync_to_thread", [True, False])
@pytest.mark.parametrize("name", [None, "multiply"])
async def test_callable_instance_tool(
    callable_type: type[Multiplier] | type[AsyncMultiplier],
    sync_to_thread: bool,
    name: str | None,
    context: Context,
) -> None:
    multiply = tool(callable_type(3), name=name, sync_to_thread=sync_to_thread)
    expected_name = name or callable_type.__name__
    [schema] = await multiply.schemas(context)
    assert schema.function.name == expected_name
    assert schema.function.description == callable_type.__doc__
    assert schema.function.parameters == {
        "type": "object",
        "properties": {"value": {"title": "Value", "type": "integer"}},
        "required": ["value"],
    }

    agent = Agent(
        "calculator",
        tools=[multiply],
        config=TestConfig(ToolCallEvent(name=expected_name, arguments='{"value": 7}'), "done"),
    )
    reply = await agent.ask("Multiply seven by three.")

    assert reply.body == "done"
    results = await reply.context.stream.history.where(ToolResultEvent).all()
    assert [result.result.parts for result in results] == [[TextInput("21")]]
