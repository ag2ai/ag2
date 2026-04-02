# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent, Context
from autogen.beta.events import BaseEvent, HumanInputRequest, HumanMessage, ToolCallEvent
from autogen.beta.middleware import BaseMiddleware, Middleware
from autogen.beta.middleware.base import HumanInputHook
from autogen.beta.testing import TestConfig


@pytest.fixture()
def test_config() -> TestConfig:
    return TestConfig(
        ToolCallEvent(name="my_tool"),
        "result",
    )


class MockHumanInputMiddleware(BaseMiddleware):
    def __init__(
        self,
        event: BaseEvent,
        ctx: Context,
        mock: MagicMock,
    ) -> None:
        super().__init__(event, ctx)
        self.mock = mock

    async def on_human_input(
        self,
        call_next: HumanInputHook,
        event: HumanInputRequest,
        ctx: Context,
    ) -> HumanMessage:
        self.mock.enter(event.content)
        result = await call_next(event, ctx)
        self.mock.exit(result.content)
        return result


@pytest.mark.asyncio()
async def test_human_input_middleware(mock: MagicMock, test_config: TestConfig) -> None:
    async def my_tool(ctx: Context) -> str:
        await ctx.input("Say smth", timeout=1.0)
        return ""

    def hitl_hook(event: HumanInputRequest) -> HumanMessage:
        return HumanMessage(content="answer")

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        hitl_hook=hitl_hook,
        middleware=[Middleware(MockHumanInputMiddleware, mock=mock)],
    )

    await agent.ask("Hi!")

    mock.enter.assert_called_once_with("Say smth")
    mock.exit.assert_called_once_with("answer")


class OrderingHumanInputMiddleware(BaseMiddleware):
    def __init__(
        self,
        event: BaseEvent,
        ctx: Context,
        mock: MagicMock,
        position: int,
    ) -> None:
        super().__init__(event, ctx)
        self.mock = mock
        self.position = position

    async def on_human_input(
        self,
        call_next: HumanInputHook,
        event: HumanInputRequest,
        ctx: Context,
    ) -> HumanMessage:
        self.mock.enter(self.position)
        result = await call_next(event, ctx)
        self.mock.exit(self.position)
        return result


@pytest.mark.asyncio()
async def test_human_input_middleware_call_sequence(mock: MagicMock, test_config: TestConfig) -> None:
    async def my_tool(ctx: Context) -> str:
        await ctx.input("Say smth", timeout=1.0)
        return ""

    def hitl_hook(event: HumanInputRequest) -> HumanMessage:
        return HumanMessage(content="answer")

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        hitl_hook=hitl_hook,
        middleware=[Middleware(OrderingHumanInputMiddleware, mock=mock, position=i) for i in range(1, 4)],
    )

    await agent.ask("Hi!")

    assert [c.args[0] for c in mock.enter.call_args_list] == [1, 2, 3]
    assert [c.args[0] for c in mock.exit.call_args_list] == [3, 2, 1]


@pytest.mark.asyncio()
async def test_human_input_middleware_mutates_request(mock: MagicMock, test_config: TestConfig) -> None:
    class MutatingMiddleware(BaseMiddleware):
        async def on_human_input(
            self,
            call_next: HumanInputHook,
            event: HumanInputRequest,
            ctx: Context,
        ) -> HumanMessage:
            event = HumanInputRequest(content=event.content + "!")
            return await call_next(event, ctx)

    async def my_tool(ctx: Context) -> str:
        await ctx.input("Say smth", timeout=1.0)
        return ""

    def hitl_hook(event: HumanInputRequest) -> HumanMessage:
        mock.hitl(event.content)
        return HumanMessage(content="answer")

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        hitl_hook=hitl_hook,
        middleware=[MutatingMiddleware, MutatingMiddleware, MutatingMiddleware],
    )

    await agent.ask("Hi!")

    mock.hitl.assert_called_once_with("Say smth!!!")


@pytest.mark.asyncio()
async def test_human_input_middleware_mutates_response(mock: MagicMock, test_config: TestConfig) -> None:
    class MutatingMiddleware(BaseMiddleware):
        async def on_human_input(
            self,
            call_next: HumanInputHook,
            event: HumanInputRequest,
            ctx: Context,
        ) -> HumanMessage:
            result = await call_next(event, ctx)
            return HumanMessage(content=result.content + "!")

    async def my_tool(ctx: Context) -> str:
        mock(await ctx.input("Say smth", timeout=1.0))
        return ""

    def hitl_hook(event: HumanInputRequest) -> HumanMessage:
        return HumanMessage(content="answer")

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        hitl_hook=hitl_hook,
        middleware=[MutatingMiddleware, MutatingMiddleware, MutatingMiddleware],
    )

    await agent.ask("Hi!")

    mock.assert_called_once_with("answer!!!")


@pytest.mark.asyncio()
async def test_human_input_middleware_short_circuits(mock: MagicMock, test_config: TestConfig) -> None:
    class ShortCircuitMiddleware(BaseMiddleware):
        async def on_human_input(
            self,
            call_next: HumanInputHook,
            event: HumanInputRequest,
            ctx: Context,
        ) -> HumanMessage:
            mock.intercepted(event.content)
            return HumanMessage(content="intercepted")

    async def my_tool(ctx: Context) -> str:
        mock(await ctx.input("Say smth", timeout=1.0))
        return ""

    def hitl_hook(event: HumanInputRequest) -> HumanMessage:
        mock.hitl()
        return HumanMessage(content="answer")

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        hitl_hook=hitl_hook,
        middleware=[Middleware(ShortCircuitMiddleware)],
    )

    await agent.ask("Hi!")

    mock.intercepted.assert_called_once_with("Say smth")
    mock.hitl.assert_not_called()
    mock.assert_called_once_with("intercepted")


@pytest.mark.asyncio()
async def test_human_input_hook_returns_raw_string(mock: MagicMock, test_config: TestConfig) -> None:
    async def my_tool(ctx: Context) -> str:
        mock(await ctx.input("Say smth", timeout=1.0))
        return ""

    def hitl_hook(event: HumanInputRequest) -> str:
        return "raw answer"

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        hitl_hook=hitl_hook,
    )

    await agent.ask("Hi!")

    mock.assert_called_once_with("raw answer")


@pytest.mark.asyncio()
async def test_human_input_hook_returns_raw_string_async(mock: MagicMock, test_config: TestConfig) -> None:
    async def my_tool(ctx: Context) -> str:
        mock(await ctx.input("Say smth", timeout=1.0))
        return ""

    async def hitl_hook(event: HumanInputRequest) -> str:
        return "async raw answer"

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        hitl_hook=hitl_hook,
    )

    await agent.ask("Hi!")

    mock.assert_called_once_with("async raw answer")


@pytest.mark.asyncio()
async def test_human_input_hook_passed_at_ask(mock: MagicMock, test_config: TestConfig) -> None:
    async def my_tool(ctx: Context) -> str:
        mock(await ctx.input("Say smth", timeout=1.0))
        return ""

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
    )

    await agent.ask("Hi!", hitl_hook=lambda event: "ask-level answer")

    mock.assert_called_once_with("ask-level answer")


@pytest.mark.asyncio()
async def test_human_input_hook_at_ask_overrides_agent(mock: MagicMock, test_config: TestConfig) -> None:
    async def my_tool(ctx: Context) -> str:
        mock(await ctx.input("Say smth", timeout=1.0))
        return ""

    def agent_hook(event: HumanInputRequest) -> str:
        return "agent-level"

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        hitl_hook=agent_hook,
    )

    await agent.ask("Hi!", hitl_hook=lambda event: "ask-level")

    mock.assert_called_once_with("ask-level")
