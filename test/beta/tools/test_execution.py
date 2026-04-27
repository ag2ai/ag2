# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock

import pytest
from fast_depends import Depends
from pydantic import BaseModel

from autogen.beta import Agent, Context, DataInput, ImageInput, TextInput, ToolResult, events, testing, tool


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
