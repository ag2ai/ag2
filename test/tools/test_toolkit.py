# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ag2 import Agent, Context, tool
from ag2.events import ToolCallEvent
from ag2.exceptions import ToolConflictError
from ag2.middleware import ToolExecution, ToolResultType
from ag2.testing import TestConfig
from ag2.tools import Toolkit


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
    from ag2 import Context

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


@pytest.mark.asyncio()
async def test_toolkit_middleware_applied_to_all_tools(mock: MagicMock) -> None:
    """Toolkit middleware wraps every tool in the set."""

    async def logging_middleware(
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        mock.before(event.name)
        result = await call_next(event, context)
        mock.after(event.name)
        return result

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    toolkit = Toolkit(add, multiply, middleware=[logging_middleware])

    config = TestConfig(
        ToolCallEvent(name="add", arguments=json.dumps({"a": 1, "b": 2})),
        ToolCallEvent(name="multiply", arguments=json.dumps({"a": 3, "b": 4})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    await agent.ask("Hi!")

    assert mock.before.call_count == 2
    mock.before.assert_any_call("add")
    mock.before.assert_any_call("multiply")
    assert mock.after.call_count == 2


@pytest.mark.asyncio()
async def test_toolkit_middleware_wraps_subagent_tool(mock: MagicMock) -> None:
    async def logging_middleware(
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        mock(event.name)
        return await call_next(event, context)

    child = Agent("child", config=TestConfig("child answer"))
    toolkit = Toolkit(child.as_tool(description="Delegate work"), middleware=[logging_middleware])
    parent = Agent(
        "parent",
        config=TestConfig(
            ToolCallEvent(name="task_child", arguments=json.dumps({"objective": "work"})),
            "done",
        ),
        tools=[toolkit],
    )

    assert (await parent.ask("Hi!")).body == "done"
    mock.assert_called_once_with("task_child")


@pytest.mark.asyncio()
async def test_toolkit_middleware_applied_to_decorator_tools(mock: MagicMock) -> None:
    """Toolkit middleware also wraps tools added via the .tool() decorator."""

    async def logging_middleware(
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        mock.before(event.name)
        result = await call_next(event, context)
        mock.after(event.name)
        return result

    toolkit = Toolkit(middleware=[logging_middleware])

    @toolkit.tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"hello {name}"

    config = TestConfig(
        ToolCallEvent(name="greet", arguments=json.dumps({"name": "world"})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    await agent.ask("Hi!")

    mock.before.assert_called_once_with("greet")
    mock.after.assert_called_once_with("greet")


@pytest.mark.asyncio()
async def test_toolkit_middleware_ordering() -> None:
    """Per-tool middleware runs before toolkit middleware."""
    call_order: list[str] = []

    async def tool_mw(
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        call_order.append("tool_mw")
        return await call_next(event, context)

    async def toolkit_mw(
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        call_order.append("toolkit_mw")
        return await call_next(event, context)

    @tool(middleware=[tool_mw])
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        call_order.append("tool")
        return a + b

    toolkit = Toolkit(add, middleware=[toolkit_mw])

    config = TestConfig(
        ToolCallEvent(name="add", arguments=json.dumps({"a": 1, "b": 2})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    await agent.ask("Hi!")

    assert call_order == ["toolkit_mw", "tool_mw", "tool"]


def test_tool_name_conflict() -> None:
    def add(a: int, b: int) -> int:
        pass

    with pytest.raises(ToolConflictError, match="add"):
        Toolkit(add, add)

    toolkit = Toolkit(add)
    with pytest.raises(ToolConflictError, match="add"):
        toolkit.tool(add)


def test_unsafe_override() -> None:
    def add(a: int, b: int) -> int:
        pass

    toolkit = Toolkit(add)
    toolkit._add_tool(add, unsafe=True)

    assert len(toolkit.tools) == 1


class TestMerger:
    def test_merge_toolkits(self) -> None:
        def add1(a: int, b: int) -> int:
            pass

        def add2(a: int, b: int) -> int:
            pass

        toolkit = Toolkit(add1) | Toolkit(add2)

        assert [t.name for t in toolkit.tools] == ["add1", "add2"]

    def test_merge_toolkit_and_tool(self) -> None:
        def add1(a: int, b: int) -> int:
            pass

        def add2(a: int, b: int) -> int:
            pass

        toolkit = Toolkit(add1) | add2

        assert [t.name for t in toolkit.tools] == ["add1", "add2"]

    def test_merged_toolkit_overrides_tool(self) -> None:
        def add1(a: int, b: int) -> int:
            pass

        toolkit = Toolkit(add1) | add1

        assert [t.name for t in toolkit.tools] == ["add1"]


async def first_merge_tool(context: Context) -> str:
    context.dependencies["calls"].append("body:first_merge_tool")
    return "first"


async def second_merge_tool(context: Context) -> str:
    context.dependencies["calls"].append("body:second_merge_tool")
    return "second"


async def record_toolkit_middleware(call_next: ToolExecution, event: ToolCallEvent, context: Context) -> ToolResultType:
    context.dependencies["calls"].append(f"toolkit:{event.name}")
    return await call_next(event, context)


async def record_right_middleware(call_next: ToolExecution, event: ToolCallEvent, context: Context) -> ToolResultType:
    context.dependencies["calls"].append(f"right:{event.name}")
    return await call_next(event, context)


async def record_tool_middleware(call_next: ToolExecution, event: ToolCallEvent, context: Context) -> ToolResultType:
    context.dependencies["calls"].append(f"tool:{event.name}")
    return await call_next(event, context)


@pytest.mark.asyncio
@pytest.mark.parametrize("right_kind", ["function", "tool", "toolkit"])
@pytest.mark.parametrize("merge_count", [1, 3])
async def test_merge_applies_middleware_once(right_kind: str, merge_count: int) -> None:
    left_tool = tool(first_merge_tool, middleware=[record_tool_middleware])
    original = Toolkit(left_tool, middleware=[record_toolkit_middleware])
    right_tool = tool(second_merge_tool, middleware=[record_tool_middleware])
    right = (
        second_merge_tool
        if right_kind == "function"
        else right_tool
        if right_kind == "tool"
        else Toolkit(right_tool, middleware=[record_right_middleware])
    )
    merged = original
    for _ in range(merge_count):
        merged = merged | right

    calls: list[str] = []
    agent = Agent(
        "test",
        tools=[merged],
        dependencies={"calls": calls},
        config=TestConfig(
            ToolCallEvent(name="first_merge_tool"),
            ToolCallEvent(name="second_merge_tool"),
            "done",
        ),
    )

    assert (await agent.ask("Run both tools")).body == "done"
    expected = ["toolkit:first_merge_tool", "tool:first_merge_tool", "body:first_merge_tool"]
    expected.append("toolkit:second_merge_tool")
    if right_kind == "toolkit":
        expected.append("right:second_merge_tool")
    if right_kind != "function":
        expected.append("tool:second_merge_tool")
    expected.append("body:second_merge_tool")
    assert calls == expected


@pytest.mark.asyncio
async def test_merge_preserves_source_toolkits_and_middleware_for_later_tools() -> None:
    original = Toolkit(first_merge_tool, middleware=[record_toolkit_middleware])
    right = Toolkit(second_merge_tool, middleware=[record_right_middleware])
    merged = original | right
    merged.tool(first_merge_tool, name="later", middleware=[record_tool_middleware])
    calls: list[str] = []

    for toolkit, name in [(original, "first_merge_tool"), (right, "second_merge_tool"), (merged, "later")]:
        agent = Agent(
            "test",
            tools=[toolkit],
            dependencies={"calls": calls},
            config=TestConfig(ToolCallEvent(name=name), "done"),
        )
        assert (await agent.ask("Run the tool")).body == "done"

    assert calls == [
        "toolkit:first_merge_tool",
        "body:first_merge_tool",
        "right:second_merge_tool",
        "body:second_merge_tool",
        "toolkit:later",
        "tool:later",
        "body:first_merge_tool",
    ]
    assert [t.name for t in original.tools] == ["first_merge_tool"]
    assert [t.name for t in right.tools] == ["second_merge_tool"]
