# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import ExitStack
from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent, Context, MemoryStream, tool
from autogen.beta.events import ClientToolCallEvent, ToolCallEvent
from autogen.beta.middleware import ToolExecution, ToolResultType
from autogen.beta.testing import TestConfig
from autogen.beta.tools.final.client_tool import ClientTool


@pytest.fixture()
def client_tool() -> ClientTool:
    return ClientTool(schema={"function": {"name": "my_client_tool", "description": "desc", "parameters": {}}})


@pytest.mark.asyncio
async def test_client_tool_call_returns_client_tool_call(client_tool: ClientTool, mock: MagicMock) -> None:
    """ClientTool.__call__ must return a ClientToolCallEvent wrapping the original call."""
    call = ToolCallEvent(name="my_client_tool", arguments="{}")
    result = await client_tool(call, mock())

    assert isinstance(result, ClientToolCallEvent)
    assert result.name == "my_client_tool"
    assert result.parent_id == call.id


@pytest.mark.asyncio
async def test_client_tool_register_execute_sends_to_stream(client_tool: ClientTool) -> None:
    """The execute closure inside register() must send ClientToolCallEvent to the stream.

    Regression: the original code did `return await execution(...)` without
    `await context.send(result)`, so ToolExecutor.execute_tools() would block
    forever waiting for a ClientToolCallEvent that was never sent to the stream.
    """
    stream = MemoryStream()
    context = Context(stream=stream)

    with ExitStack() as stack:
        client_tool.register(stack, context)
        call = ToolCallEvent(name="my_client_tool", arguments="{}")
        await stream.send(call, context)

    events = await stream.history.get_events()

    assert len(events) == 2
    assert isinstance(events[-1], ClientToolCallEvent)
    assert events[1].parent_id == call.id
    assert events[1].name == call.name


@pytest.mark.asyncio
async def test_client_tool_register_with_middleware(client_tool: ClientTool) -> None:
    """execute closure must propagate through middleware before sending."""
    stream = MemoryStream()
    context = Context(stream=stream)

    class TagMiddleware:
        async def on_tool_execution(self, call_next: object, event: object, context: object) -> object:
            result = await call_next(event, context)  # type: ignore[misc]
            result._tag = "middleware_ran"  # type: ignore[attr-defined]
            return result

    with ExitStack() as stack:
        client_tool.register(stack, context, middleware=[TagMiddleware()])

        call = ToolCallEvent(name="my_client_tool", arguments="{}")
        await stream.send(call, context)

    events = await stream.history.get_events()

    assert len(events) == 2
    assert isinstance(events[-1], ClientToolCallEvent)
    assert getattr(events[-1], "_tag", None) == "middleware_ran"


@pytest.mark.asyncio()
async def test_function_tool_with_middleware_preserves_existing() -> None:
    """with_middleware appends to existing middleware without replacing it."""
    call_order: list[str] = []

    async def first_mw(
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        call_order.append("first")
        return await call_next(event, context)

    async def second_mw(
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        call_order.append("second")
        return await call_next(event, context)

    @tool(middleware=[first_mw])
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        call_order.append("tool")
        return a + b

    wrapped = add.with_middleware(second_mw)

    assert len(add._tool_middleware) == 1
    assert len(wrapped._tool_middleware) == 2

    config = TestConfig(
        ToolCallEvent(name="add", arguments=json.dumps({"a": 1, "b": 2})),
        "done",
    )
    agent = Agent("", config=config, tools=[wrapped])
    await agent.ask("Hi!")

    assert call_order == ["second", "first", "tool"]
