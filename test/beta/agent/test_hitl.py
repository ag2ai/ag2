# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent, Context
from autogen.beta.events import HumanInputRequest, HumanMessage, ToolCall
from autogen.beta.exceptions import HumanInputNotProvidedError
from autogen.beta.testing import TestConfig


@pytest.fixture()
def test_config() -> TestConfig:
    return TestConfig(
        ToolCall(name="my_tool"),
        "result",
    )


@pytest.mark.asyncio()
async def test_sync_hitl(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    async def my_tool(ctx: Context) -> str:
        mock(await ctx.input("Say smth", timeout=1.0))
        return ""

    def hitl_hook(event: HumanInputRequest) -> HumanMessage:
        mock.hitl(event.content)
        return HumanMessage(content="answer")

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        hitl_hook=hitl_hook,
    )

    await agent.ask("Hi!")

    mock.assert_called_once_with("answer")
    mock.hitl.assert_called_once_with("Say smth")


@pytest.mark.asyncio()
async def test_async_hitl(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    async def my_tool(ctx: Context) -> str:
        mock(await ctx.input("Say smth", timeout=1.0))
        return ""

    async def hitl_hook(event: HumanInputRequest) -> HumanMessage:
        return HumanMessage(content="answer")

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        hitl_hook=hitl_hook,
    )

    await agent.ask("Hi!")

    mock.assert_called_once_with("answer")


@pytest.mark.asyncio()
async def test_hitl_decorator(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    async def my_tool(ctx: Context) -> str:
        mock(await ctx.input("Say smth", timeout=1.0))
        return ""

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
    )

    @agent.hitl_hook
    def hitl_hook(event: HumanInputRequest) -> HumanMessage:
        return HumanMessage(content="answer")

    await agent.ask("Hi!")

    mock.assert_called_once_with("answer")


@pytest.mark.asyncio()
async def test_hitl_decorator_override(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    async def my_tool(ctx: Context) -> str:
        mock(await ctx.input("Say smth", timeout=1.0))
        return ""

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
    )

    @agent.hitl_hook
    def overrided_hook(event: HumanInputRequest) -> HumanMessage:
        return HumanMessage(content="wrong")

    with pytest.warns(RuntimeWarning):

        @agent.hitl_hook
        def hitl_hook(event: HumanInputRequest) -> HumanMessage:
            return HumanMessage(content="answer")

    await agent.ask("Hi!")

    mock.assert_called_once_with("answer")


@pytest.mark.asyncio()
async def test_hitl_not_set(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    async def my_tool(ctx: Context) -> str:
        try:
            await ctx.input("Say smth", timeout=1.0)
        except HumanInputNotProvidedError:
            mock()
        return ""

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
    )

    await agent.ask("Hi!")

    mock.assert_called_once()
