# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from contextlib import ExitStack
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock

import pytest
from fast_depends import Depends
from pydantic import BaseModel

import autogen.beta.tools.executor as executor_mod
from autogen.beta import (
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
from autogen.beta.events import ToolCallsEvent, ToolResultEvent, ToolResultsEvent
from autogen.beta.exceptions import ToolNotFoundError
from autogen.beta.tools.subagents import subagent_tool as subagent_tool_factory


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
    """A failed sub-task must return an error string, never an empty success."""

    class FailingWorker:
        name = "worker"
        _hitl_hook = None

        async def ask(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("boom")

    tool_obj = subagent_tool_factory(FailingWorker(), description="do work")

    ctx = Context(stream=MemoryStream())
    result = await tool_obj(
        events.ToolCallEvent(
            name=tool_obj.name,
            arguments=json.dumps({"objective": "solve it"}),
        ),
        context=ctx,
    )

    out = result.result.parts[0].content
    assert out != "", "sub-task failure must not return empty string to parent LLM"
    assert "boom" in out or "fail" in out.lower()


@pytest.mark.asyncio
async def test_execute_tools_isolates_a_failing_tool() -> None:
    """One failing tool must not suppress the other tool's result (public-API end-to-end).

    Note: this exercises FunctionTool's Exception→ToolErrorEvent conversion path.
    It does NOT cover the gather-level return_exceptions guard (executor.py:67-70),
    because a normal Exception is caught inside FunctionTool.__call__ before it
    reaches the gather. See test_execute_tools_isolates_a_raising_call for that guard.
    """

    @tool(name="good_tool")
    def good_tool() -> str:
        return "all good"

    @tool(name="bad_tool")
    def bad_tool() -> str:
        raise RuntimeError("tool exploded")

    stream = MemoryStream()
    ctx = Context(stream=stream)
    ex = executor_mod.ToolExecutor(serializer=MagicMock())

    with ExitStack() as stack:
        ex.register(stack, ctx, tools=[good_tool, bad_tool])
        await ctx.send(
            ToolCallsEvent([
                events.ToolCallEvent(name="good_tool"),
                events.ToolCallEvent(name="bad_tool"),
            ])
        )

    history = await stream.history.get_events()
    results_events = [e for e in history if isinstance(e, ToolResultsEvent)]
    assert results_events, "expected a ToolResultsEvent in history"
    [results_event] = results_events
    # Both calls represented: one success, one error — nothing silently dropped.
    assert len(results_event.results) == 2


@pytest.mark.asyncio
async def test_execute_tools_isolates_a_raising_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """White-box guard: the gather return_exceptions path isolates infra-level raises.

    The gather-level ``return_exceptions`` guard (executor.py:67-70) is only reachable
    when ``_execute_call`` itself raises at the infrastructure level. A normal raising
    tool is converted to a ToolErrorEvent inside FunctionTool.__call__ before reaching
    this guard, so this specific path requires injection via monkeypatch.
    """
    good = events.ToolCallEvent(id="ok", name="good")
    bad = events.ToolCallEvent(id="boom", name="bad")

    async def fake_execute_call(context: object, call: events.ToolCallEvent) -> ToolResultEvent:
        if call.id == "boom":
            raise RuntimeError("infra exploded")
        return ToolResultEvent(parent_id=call.id, result=ToolResult(TextInput("done")))

    monkeypatch.setattr(executor_mod, "_execute_call", fake_execute_call)

    sent: list[object] = []
    ctx = MagicMock()

    async def _send(ev: object, *a: object, **k: object) -> None:
        sent.append(ev)

    ctx.send = _send

    ex = executor_mod.ToolExecutor(serializer=MagicMock())
    await ex.execute_tools(ToolCallsEvent([good, bad]), ctx)

    results_event = next(e for e in sent if isinstance(e, ToolResultsEvent))
    # Both calls represented: one success, one error — nothing silently dropped.
    assert len(results_event.results) == 2


@pytest.mark.asyncio
async def test_execute_tools_propagates_cancellation() -> None:
    """A cancelled tool call must propagate, not be masked as a ToolResultsEvent.

    FunctionTool.__call__ only catches Exception (not BaseException), so
    asyncio.CancelledError propagates through _execute_call to the gather,
    where the guard at executor.py:69 re-raises it unconditionally.
    """

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
