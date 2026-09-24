# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import ExitStack
from unittest.mock import MagicMock

import pytest

from ag2 import Agent, Context, MemoryStream, tool
from ag2.events import ClientToolCallEvent, ModelResponse, ToolCallEvent
from ag2.middleware import BaseMiddleware, ToolExecution, ToolResultType
from ag2.testing import TestConfig
from ag2.tools.final.client_tool import ClientTool


@pytest.fixture()
def client_tool() -> ClientTool:
    return ClientTool(schema={"function": {"name": "my_client_tool", "description": "desc", "parameters": {}}})


def test_client_tool_condition() -> None:
    condition = ClientToolCallEvent.id == "1"
    assert condition(ClientToolCallEvent(id="1", name="test"))
    assert not condition(ToolCallEvent(id="1", name="test"))


@pytest.mark.asyncio
async def test_client_tool_call_returns_client_tool_call(client_tool: ClientTool, mock: MagicMock) -> None:
    """ClientTool.__call__ must return a ClientToolCallEvent wrapping the original call."""
    call = ToolCallEvent(name="my_client_tool", arguments="{}")
    result = await client_tool(call, mock())

    assert isinstance(result, ClientToolCallEvent)
    assert result.name == "my_client_tool"
    assert result.id == call.id


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
        await context.send(call)

    events = list(await stream.history.get_events())

    assert isinstance(events[-1], ClientToolCallEvent)
    assert events[-1].id == call.id
    assert events[-1].name == call.name


@pytest.mark.asyncio
async def test_client_tool_register_with_middleware(client_tool: ClientTool) -> None:
    """execute closure must propagate through middleware before sending."""
    stream = MemoryStream()
    context = Context(stream=stream)

    class TagMiddleware(BaseMiddleware):
        async def on_tool_execution(
            self,
            call_next: ToolExecution,
            event: ToolCallEvent,
            context: Context,
        ) -> ToolResultType:
            result = await call_next(event, context)
            result._tag = "middleware_ran"  # type: ignore[union-attr]  # a marker the test reads back; no event declares it
            return result

    call = ToolCallEvent(name="my_client_tool", arguments="{}")
    with ExitStack() as stack:
        client_tool.register(stack, context, middleware=[TagMiddleware(call, context)])

        await context.send(call)

    events = list(await stream.history.get_events())

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

    assert len(add._middleware) == 1
    assert len(wrapped._middleware) == 2

    config = TestConfig(
        ToolCallEvent(name="add", arguments=json.dumps({"a": 1, "b": 2})),
        "done",
    )
    agent = Agent("", config=config, tools=[wrapped])
    await agent.ask("Hi!")

    assert call_order == ["second", "first", "tool"]


class TestClientCallsInHistory:
    """The turn the client must answer is recorded as the calls the model made.

    `ClientToolCallEvent` is an *outcome* — `ClientTool` subscribes to the
    model's `ToolCallEvent` and emits one as that call's result, which is why it
    is a sibling of `ToolResultEvent` rather than a `ToolCallEvent`. Putting
    outcomes into the `ToolCallsEvent` on the assistant turn made history claim
    the model had asked for them, and dropped the provider fields with it.
    """

    @pytest.mark.asyncio
    async def test_the_assistant_turn_carries_the_model_s_calls(self) -> None:
        client_tool = ClientTool(schema={"function": {"name": "pick", "description": "d", "parameters": {}}})
        stream = MemoryStream()
        call = ToolCallEvent(name="pick", arguments="{}", vendor_metadata={"caller": "ui"})
        agent = Agent("", config=TestConfig(call), tools=[client_tool])

        await agent.ask("go", stream=stream)

        events = await stream.history.get_events()
        pending = [e.tool_calls for e in events if isinstance(e, ModelResponse) and e.tool_calls.calls][-1]
        assert [type(c) for c in pending.calls] == [ToolCallEvent]
        assert pending.calls[0].vendor_metadata == {"caller": "ui"}

    @pytest.mark.asyncio
    async def test_that_history_maps_for_a_provider(self) -> None:
        """The mapper reads `vendor_metadata` off every call on the turn."""
        anthropic_mappers = pytest.importorskip("ag2.config.anthropic.mappers")
        serializer = pytest.importorskip("fast_depends.pydantic.serializer")

        client_tool = ClientTool(schema={"function": {"name": "pick", "description": "d", "parameters": {}}})
        stream = MemoryStream()
        agent = Agent("", config=TestConfig(ToolCallEvent(name="pick", arguments="{}")), tools=[client_tool])

        await agent.ask("go", stream=stream)

        events = list(await stream.history.get_events())
        anthropic_mappers.convert_messages(events, serializer.PydanticSerializer())
