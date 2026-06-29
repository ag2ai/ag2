# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock

import pytest
from fast_depends import Depends
from pydantic import BaseModel

from ag2 import (
    Agent,
    Context,
    DataInput,
    ImageInput,
    MemoryStream,
    TextInput,
    ToolResult,
    events,
    testing,
    tool,
)
from ag2.events import ToolResultsEvent
from ag2.exceptions import ToolNotFoundError
from ag2.middleware import ToolExecution
from ag2.tools.subagents import subagent_tool


@pytest.mark.asyncio
async def test_execute(async_mock: AsyncMock, mock: AsyncMock) -> None:
    @tool
    def my_func(a: str, b: int) -> str:
        mock(a=a, b=b)
        return "tool executed"

    result = await my_func(
        events.ToolCallEvent(
            arguments=json.dumps({"a": "1", "b": "1"}),
            name="my_func",
        ),
        context=Context(async_mock),
    )

    assert result.result.parts[0].content == "tool executed"
    mock.assert_called_once_with(a="1", b=1)


@pytest.mark.asyncio
async def test_execute_sync_without_thread(async_mock: AsyncMock, mock: MagicMock) -> None:
    @tool(sync_to_thread=False)
    def my_func(a: str, b: int) -> str:
        mock(a=a, b=b)
        return "tool executed"

    result = await my_func(
        events.ToolCallEvent(
            arguments=json.dumps({"a": "1", "b": "1"}),
            name="my_func",
        ),
        context=Context(async_mock),
    )

    assert result.result.parts[0].content == "tool executed"
    mock.assert_called_once_with(a="1", b=1)


@pytest.mark.asyncio
async def test_execute_async(async_mock: AsyncMock, mock: MagicMock) -> None:
    @tool
    async def my_func(a: str, b: int) -> str:
        mock(a=a, b=b)
        return "tool executed"

    result = await my_func(
        events.ToolCallEvent(
            arguments=json.dumps({"a": "1", "b": "1"}),
            name="my_func",
        ),
        context=Context(async_mock),
    )

    assert result.result.parts[0].content == "tool executed"
    mock.assert_called_once_with(a="1", b=1)


@pytest.mark.asyncio
async def test_return_model(async_mock: AsyncMock) -> None:
    class Result(BaseModel):
        a: str

    @tool
    def my_func(a: str, b: int) -> Result:
        return Result(a=a)

    result = await my_func(
        events.ToolCallEvent(
            arguments=json.dumps({"a": "1", "b": "1"}),
            name="my_func",
        ),
        context=Context(async_mock),
    )

    assert isinstance(result.result.parts[0], DataInput)
    assert result.result.parts[0].data == Result(a="1")


@pytest.mark.asyncio
async def test_return_result(async_mock: AsyncMock) -> None:
    @tool
    def my_func() -> ToolResult:
        return ToolResult("Hi!")

    result = await my_func(
        events.ToolCallEvent(name="my_func"),
        context=Context(async_mock),
    )

    assert result.result.parts[0].content == "Hi!"


@pytest.mark.asyncio
async def test_tool_with_depends(async_mock: AsyncMock) -> None:
    def dep(a: str) -> str:
        return a * 2

    @tool
    def my_func(a: str, b: Annotated[str, Depends(dep)]) -> str:
        return a + b

    result = await my_func(
        events.ToolCallEvent(
            arguments=json.dumps({"a": "1"}),
            name="my_func",
        ),
        context=Context(async_mock),
    )

    assert result.result.parts[0].content == "111"


@pytest.mark.asyncio
async def test_tool_get_context(async_mock: AsyncMock) -> None:
    @tool
    def my_func(a: str, context: Context) -> str:
        return "".join(context.prompt)

    result = await my_func(
        events.ToolCallEvent(
            arguments=json.dumps({"a": "1"}),
            name="my_func",
        ),
        context=Context(async_mock, prompt=["1"]),
    )

    assert result.result.parts[0].content == "1"


@pytest.mark.asyncio
async def test_tool_get_context_by_random_name(async_mock: AsyncMock) -> None:
    @tool
    def my_func(a: str, c: Context) -> str:
        return "".join(c.prompt)

    result = await my_func(
        events.ToolCallEvent(
            arguments=json.dumps({"a": "1"}),
            name="my_func",
        ),
        context=Context(async_mock, prompt=["1"]),
    )

    assert result.result.parts[0].content == "1"


@pytest.mark.asyncio
class TestReturnInput:
    @pytest.fixture
    def config(self) -> testing.TrackingConfig:
        return testing.TrackingConfig(
            testing.TestConfig(
                events.ToolCallEvent(name="my_func"),
                "done",
            )
        )

    async def test_tool_return_input(self, config: testing.TrackingConfig) -> None:
        @tool
        def my_func() -> DataInput:
            return DataInput({"a": "1"})

        agent = Agent("", config=config, tools=[my_func])
        await agent.ask("Call my func")

        tool_result_msg: events.ToolResultEvent = config.mock.call_args_list[1][0][0].results[0]
        assert tool_result_msg.result.parts[0] == DataInput({"a": "1"})

    async def test_return_multiple_parts(self, config: testing.TrackingConfig) -> None:
        @tool
        def my_func() -> ToolResult:
            return ToolResult(
                TextInput("Hi!"),
                DataInput({"b": "2"}),
            )

        agent = Agent("", config=config, tools=[my_func])
        await agent.ask("Call my func")

        tool_result_msg: events.ToolResultEvent = config.mock.call_args_list[1][0][0].results[0]
        assert tool_result_msg.result.parts == [
            TextInput("Hi!"),
            DataInput({"b": "2"}),
        ]

    async def test_return_mixed_parts(self, config: testing.TrackingConfig) -> None:
        @tool
        def my_func() -> ToolResult:
            return ToolResult(
                TextInput("Hi!"),
                {"b": "2"},
            )

        agent = Agent("", config=config, tools=[my_func])
        await agent.ask("Call my func")

        tool_result_msg: events.ToolResultEvent = config.mock.call_args_list[1][0][0].results[0]
        assert tool_result_msg.result.parts == [
            TextInput("Hi!"),
            DataInput({"b": "2"}),
        ]

    async def test_text_input(self, config: testing.TrackingConfig) -> None:
        @tool
        def my_func() -> ToolResult:
            return ToolResult(TextInput("hello"), final=True)

        agent = Agent("", config=config, tools=[my_func])
        reply = await agent.ask("Call my func")

        assert reply.body == "hello"

    async def test_data_input(self, config: testing.TrackingConfig) -> None:
        @tool
        def my_func() -> ToolResult:
            return ToolResult(DataInput({"a": "1"}), final=True)

        agent = Agent("", config=config, tools=[my_func])
        reply = await agent.ask("Call my func")

        assert json.loads(reply.body) == {"a": "1"}

    async def test_unsupported_input_type(self, config: testing.TrackingConfig) -> None:
        @tool
        def my_func() -> ToolResult:
            return ToolResult(ImageInput("https://example.com/img.png"), final=True)

        agent = Agent("", config=config, tools=[my_func])

        with pytest.raises(ValueError, match="Unsupported part type"):
            await agent.ask("Call my func")

    async def test_multiple_parts_raises(self, config: testing.TrackingConfig) -> None:
        @tool
        def my_func() -> ToolResult:
            return ToolResult(TextInput("a"), TextInput("b"), final=True)

        agent = Agent("", config=config, tools=[my_func])

        with pytest.raises(ValueError, match="must have exactly one part"):
            await agent.ask("Call my func")

    async def test_llm_not_called_again(self, config: testing.TrackingConfig) -> None:
        @tool
        def my_func() -> ToolResult:
            return ToolResult(TextInput("result"), final=True)

        agent = Agent("", config=config, tools=[my_func])
        await agent.ask("Call my func")

        assert config.mock.call_count == 1


@pytest.mark.asyncio
async def test_unknown_tool_result_is_populated() -> None:
    stream = MemoryStream()
    agent = Agent("", config=testing.TestConfig(events.ToolCallEvent(name="missing_tool"), "done"))

    with pytest.raises(ToolNotFoundError):
        await agent.ask("Call a tool that does not exist", stream=stream)

    [not_found] = [e for e in await stream.history.get_events() if isinstance(e, events.ToolNotFoundEvent)]
    assert not_found.result is not None
    assert "missing_tool" in not_found.result.parts[0].content


@pytest.mark.asyncio
async def test_subagent_tool_surfaces_failure_to_caller() -> None:
    """A failed sub-task must return a non-empty error string to the parent LLM, never an empty success.

    Driven through the public seam: the worker is a real ``Agent`` whose tool raises (scripted with
    ``TestConfig``); the parent delegates to it via ``subagent_tool``. ``TrackingConfig`` lets us read
    the exact tool result the framework feeds back to the parent LLM — no patching, no agent doubles.
    """

    @tool
    async def explode() -> str:
        raise RuntimeError("boom")

    worker = Agent(
        "worker",
        config=testing.TestConfig(events.ToolCallEvent(name="explode")),
        tools=[explode],
    )
    delegate = subagent_tool(worker, description="do work", name="delegate")

    parent_config = testing.TrackingConfig(
        testing.TestConfig(
            events.ToolCallEvent(name="delegate", arguments=json.dumps({"objective": "solve it"})),
            "done",
        )
    )
    parent = Agent("parent", config=parent_config, tools=[delegate])

    await parent.ask("Delegate the task")

    # The framework's second LLM turn carries the delegate tool's result back to the parent LLM.
    tool_result: events.ToolResultEvent = parent_config.mock.call_args_list[1][0][0].results[0]
    out = tool_result.result.parts[0].content
    assert out != "", "sub-task failure must not return empty string to parent LLM"
    assert "boom" in out


@pytest.mark.asyncio
async def test_execute_tools_isolates_a_failing_tool() -> None:
    """A tool that raises must not suppress another tool's result — both come back.

    Driven through the public ``Agent`` + ``TestConfig`` seam: one turn issues two tool calls; the
    failing one surfaces as a ``ToolErrorEvent`` while the other still returns. The next LLM turn
    then re-raises the surfaced error (that is how ``TestConfig`` feeds a tool failure back to the
    model), so the run raises — but the batch's results are already on the stream, undropped.
    """

    @tool(name="good_tool")
    def good_tool() -> str:
        return "all good"

    @tool(name="bad_tool")
    def bad_tool() -> str:
        raise RuntimeError("tool exploded")

    stream = MemoryStream()
    agent = Agent(
        "",
        config=testing.TestConfig([
            events.ToolCallEvent(name="good_tool"),
            events.ToolCallEvent(name="bad_tool"),
        ]),
        tools=[good_tool, bad_tool],
    )

    with pytest.raises(RuntimeError, match="tool exploded"):
        await agent.ask("Call both tools", stream=stream)

    history = await stream.history.get_events()
    [results_event] = [e for e in history if isinstance(e, ToolResultsEvent)]
    # Both calls represented: one success, one error — nothing silently dropped.
    assert len(results_event.results) == 2
    errors = [r for r in results_event.results if isinstance(r, events.ToolErrorEvent)]
    assert any(e.result is not None and "tool exploded" in e.result.parts[0].content for e in errors)


@pytest.mark.asyncio
async def test_execute_tools_isolates_a_call_a_middleware_guard_rejects() -> None:
    """A middleware guard that rejects a call by raising must not discard the batch — the rejection
    comes back as that call's error while the other call still returns.

    Models a real authorization guard (a refund above the auto-approval limit), driven through the
    public ``Agent`` + ``TestConfig`` seam — the same shape a user would write against a live provider.
    """

    class RefundNotAuthorizedError(Exception):
        """Raised by the guard when a refund exceeds the auto-approval limit."""

    auto_approve_limit_usd = 100.0

    async def refund_authority(
        call_next: ToolExecution, event: events.ToolCallEvent, context: Context
    ) -> events.ToolResultEvent:
        amount = float(event.serialized_arguments.get("amount_usd", 0))
        if amount > auto_approve_limit_usd:
            raise RefundNotAuthorizedError(
                f"refund of ${amount:.2f} exceeds the ${auto_approve_limit_usd:.0f} auto-approval limit"
            )
        return await call_next(event, context)

    @tool(name="lookup_order")
    def lookup_order(order_id: str) -> str:
        return f"Order {order_id}: total $250.00"

    @tool(name="issue_refund", middleware=[refund_authority])
    def issue_refund(order_id: str, amount_usd: float) -> str:
        return f"Refund of ${amount_usd:.2f} issued for order {order_id}."

    stream = MemoryStream()
    agent = Agent(
        "",
        config=testing.TestConfig(
            [
                events.ToolCallEvent(name="lookup_order", arguments=json.dumps({"order_id": "A-4471"})),
                events.ToolCallEvent(
                    name="issue_refund", arguments=json.dumps({"order_id": "A-4471", "amount_usd": 250.0})
                ),
            ],
            "I could not issue the refund — it exceeds the auto-approval limit and needs human sign-off.",
        ),
        tools=[lookup_order, issue_refund],
    )

    reply = await agent.ask("Refund order A-4471 for $250", stream=stream)

    # The guard's rejection did not abort the turn — the agent still produced a final reply.
    assert reply.body is not None

    history = await stream.history.get_events()
    [results_event] = [e for e in history if isinstance(e, ToolResultsEvent)]
    # Both calls represented: the lookup succeeded, the rejected refund came back as an error.
    assert len(results_event.results) == 2
    errors = [r for r in results_event.results if isinstance(r, events.ToolErrorEvent)]
    assert any(
        e.result is not None and "exceeds the $100 auto-approval limit" in e.result.parts[0].content for e in errors
    )


@pytest.mark.asyncio
async def test_execute_tools_propagates_cancellation() -> None:
    """A cancelled tool call must propagate, not be swallowed into a results event."""

    @tool(name="cancel_tool")
    async def cancel_tool() -> str:
        raise asyncio.CancelledError

    stream = MemoryStream()
    agent = Agent(
        "",
        config=testing.TestConfig(events.ToolCallEvent(name="cancel_tool")),
        tools=[cancel_tool],
    )
    with pytest.raises(asyncio.CancelledError):
        await agent.ask("Trigger cancellation", stream=stream)

    history = await stream.history.get_events()
    assert not any(isinstance(e, ToolResultsEvent) for e in history)
