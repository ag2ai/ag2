# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from autogen.beta import Agent, Context, tool
from autogen.beta.events import ToolCallEvent
from autogen.beta.testing import TestConfig
from autogen.beta.tools import Toolkit


@pytest.mark.asyncio
async def test_toolkit_schemas(async_mock: AsyncMock) -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    toolkit = Toolkit(add, multiply)
    schemas = list(await toolkit.schemas(Context(async_mock)))

    assert len(schemas) == 2
    assert schemas[0].function.name == "add"
    assert schemas[1].function.name == "multiply"


@pytest.mark.asyncio()
async def test_toolkit_executes_tool(mock: MagicMock) -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        mock(a=a, b=b)
        return a + b

    toolkit = Toolkit(add)

    config = TestConfig(
        ToolCallEvent(name="add", arguments=json.dumps({"a": 2, "b": 3})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    result = await agent.ask("Hi!")

    mock.assert_called_once_with(a=2, b=3)
    assert result.body == "done"


@pytest.mark.asyncio()
async def test_toolkit_multiple_tools(mock: MagicMock) -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        mock.add(a=a, b=b)
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        mock.multiply(a=a, b=b)
        return a * b

    toolkit = Toolkit(add, multiply)

    config = TestConfig(
        ToolCallEvent(name="multiply", arguments=json.dumps({"a": 4, "b": 5})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    await agent.ask("Hi!")

    mock.add.assert_not_called()
    mock.multiply.assert_called_once_with(a=4, b=5)


@pytest.mark.asyncio()
async def test_toolkit_mixed_with_standalone_tool(mock: MagicMock) -> None:
    @tool
    def bundled(a: str) -> str:
        """Bundled tool."""
        mock.bundled(a)
        return a

    @tool
    def standalone(b: str) -> str:
        """Standalone tool."""
        mock.standalone(b)
        return b

    toolkit = Toolkit(bundled)

    config = TestConfig(
        ToolCallEvent(name="standalone", arguments=json.dumps({"b": "hello"})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit, standalone])
    await agent.ask("Hi!")

    mock.bundled.assert_not_called()
    mock.standalone.assert_called_once_with("hello")


@pytest.mark.asyncio()
async def test_toolkit_with_context(mock: MagicMock) -> None:
    from autogen.beta import Context

    @tool
    def greet(name: str, ctx: Context) -> str:
        """Greet someone."""
        mock(ctx.dependencies["lang"])
        return f"hello {name}"

    toolkit = Toolkit(greet)

    config = TestConfig(
        ToolCallEvent(name="greet", arguments=json.dumps({"name": "world"})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit], dependencies={"lang": "en"})
    await agent.ask("Hi!")

    mock.assert_called_once_with("en")


@pytest.mark.asyncio()
async def test_toolkit_with_plain_functions(mock: MagicMock) -> None:
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        mock(a=a, b=b)
        return a + b

    toolkit = Toolkit(add)

    config = TestConfig(
        ToolCallEvent(name="add", arguments=json.dumps({"a": 1, "b": 2})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    await agent.ask("Hi!")

    mock.assert_called_once_with(a=1, b=2)


@pytest.mark.asyncio()
async def test_toolkit_mixed_functions_and_tools(mock: MagicMock) -> None:
    @tool
    def decorated(a: str) -> str:
        """Decorated tool."""
        mock.decorated(a)
        return a

    def plain(b: str) -> str:
        """Plain function."""
        mock.plain(b)
        return b

    toolkit = Toolkit(decorated, plain)

    config = TestConfig(
        ToolCallEvent(name="plain", arguments=json.dumps({"b": "hi"})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    await agent.ask("Hi!")

    mock.decorated.assert_not_called()
    mock.plain.assert_called_once_with("hi")


@pytest.mark.asyncio()
async def test_toolkit_tool_decorator(mock: MagicMock) -> None:
    toolkit = Toolkit()

    @toolkit.tool
    def greet(name: str) -> str:
        """Greet someone."""
        mock(name)
        return f"hello {name}"

    config = TestConfig(
        ToolCallEvent(name="greet", arguments=json.dumps({"name": "world"})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    await agent.ask("Hi!")

    mock.assert_called_once_with("world")


@pytest.mark.asyncio()
async def test_toolkit_tool_decorator_with_options(mock: MagicMock) -> None:
    toolkit = Toolkit()

    @toolkit.tool(name="say_hi", description="Custom greeting.")
    def greet(name: str) -> str:
        """Greet someone."""
        mock(name)
        return f"hello {name}"

    config = TestConfig(
        ToolCallEvent(name="say_hi", arguments=json.dumps({"name": "world"})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    await agent.ask("Hi!")

    mock.assert_called_once_with("world")


@pytest.mark.asyncio()
async def test_toolkit_empty() -> None:
    toolkit = Toolkit()

    config = TestConfig("done")
    agent = Agent("", config=config, tools=[toolkit])
    result = await agent.ask("Hi!")

    assert result.body == "done"
