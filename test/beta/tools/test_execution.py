# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock

import pytest
from fast_depends import Depends
from pydantic import BaseModel

from autogen.beta import Agent, Context, TextInput, ToolResult, events, testing, tool


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

    assert result.content == '"tool executed"'
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

    assert result.content == '"tool executed"'
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

    assert result.content == '"tool executed"'
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

    assert result.content == '{"a":"1"}'


@pytest.mark.asyncio
async def test_return_result(async_mock: AsyncMock) -> None:
    @tool
    def my_func() -> ToolResult:
        return ToolResult("Hi!")

    result = await my_func(
        events.ToolCallEvent(name="my_func"),
        context=Context(async_mock),
    )

    assert result.content == '"Hi!"'


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

    assert result.content == '"111"'


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

    assert result.content == '"1"'


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

    assert result.content == '"1"'


@pytest.mark.asyncio
@pytest.mark.xfail(reason="TODO: Refactor ToolResultEvent")
async def test_tool_return_input() -> None:
    @tool
    def my_func() -> TextInput:
        return TextInput("Hi!")

    tracking = testing.TrackingConfig(
        testing.TestConfig(
            events.ToolCallEvent(name="my_func"),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[my_func])
    await agent.ask("Call my func")

    tool_result_msg: events.ToolResultEvent = tracking.mock.call_args_list[1][0][0].results[0]
    print(tool_result_msg)
    assert tool_result_msg.content == '"Hi!"'
