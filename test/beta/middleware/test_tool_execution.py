# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent, Context
from autogen.beta.events import BaseEvent, ToolCallEvent, ToolResultEvent, ToolResultsEvent
from autogen.beta.middleware import BaseMiddleware, Middleware, ToolExecution, ToolMiddleware
from autogen.beta.testing import TestConfig, TrackingConfig
from autogen.beta.tools import Toolkit, tool


class OrderingMiddleware(BaseMiddleware):
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

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: ToolCallEvent,
        ctx: Context,
    ) -> ToolResultEvent:
        self.mock.enter(self.position)
        result = await call_next(event, ctx)
        self.mock.exit(self.position)
        return result


class TestToolExecutionMiddleware:
    @pytest.mark.asyncio()
    async def test_basic(self, mock: MagicMock) -> None:
        class MockMiddleware(BaseMiddleware):
            def __init__(
                self,
                event: BaseEvent,
                ctx: Context,
                mock: MagicMock,
            ) -> None:
                super().__init__(event, ctx)
                self.mock = mock

            async def on_tool_execution(
                self,
                call_next: ToolExecution,
                event: ToolCallEvent,
                ctx: Context,
            ) -> ToolResultEvent:
                self.mock.enter(event.name)
                r = await call_next(event, ctx)
                self.mock.exit(r.result.parts[0].content)
                return r

        def my_tool() -> str:
            return "tool executed"

        agent = Agent(
            "",
            config=TestConfig(
                ToolCallEvent(name="my_tool"),
                "result",
            ),
            tools=[my_tool],
            middleware=[Middleware(MockMiddleware, mock=mock)],
        )

        await agent.ask("Hi!")

        mock.enter.assert_called_once_with("my_tool")
        mock.exit.assert_called_once_with("tool executed")

    @pytest.mark.asyncio()
    async def test_call_sequence(self, mock: MagicMock) -> None:
        def my_tool() -> str:
            return "ok"

        agent = Agent(
            "",
            config=TestConfig(
                ToolCallEvent(name="my_tool"),
                "result",
            ),
            tools=[my_tool],
            middleware=[Middleware(OrderingMiddleware, mock=mock, position=i) for i in range(1, 4)],
        )

        await agent.ask("Hi!")

        assert [c.args[0] for c in mock.enter.call_args_list] == [1, 2, 3]
        assert [c.args[0] for c in mock.exit.call_args_list] == [3, 2, 1]

    @pytest.mark.asyncio()
    async def test_capture_error(self, mock: MagicMock) -> None:
        class MockMiddleware(BaseMiddleware):
            def __init__(
                self,
                event: BaseEvent,
                ctx: Context,
                mock: MagicMock,
            ) -> None:
                super().__init__(event, ctx)
                self.mock = mock

            async def on_tool_execution(
                self,
                call_next: ToolExecution,
                event: ToolCallEvent,
                ctx: Context,
            ) -> ToolResultEvent:
                r = await call_next(event, ctx)
                self.mock.exit(repr(r.error))
                # suppress the error
                return ToolResultEvent.from_call(event, result="tool executed")

        def my_tool() -> str:
            raise ValueError("tool execution error")

        tracking_config = TrackingConfig(TestConfig(ToolCallEvent(name="my_tool"), "result"))

        agent = Agent(
            "",
            config=tracking_config,
            tools=[my_tool],
            middleware=[Middleware(MockMiddleware, mock=mock)],
        )

        await agent.ask("Hi!")

        mock.exit.assert_called_once_with("ValueError('tool execution error')")

        tool_results_event: ToolResultsEvent = tracking_config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0].content == "tool executed"

    @pytest.mark.asyncio()
    async def test_mutates_arguments_and_result(self) -> None:
        class MutatingToolMiddleware(BaseMiddleware):
            async def on_tool_execution(
                self,
                call_next: ToolExecution,
                event: ToolCallEvent,
                ctx: Context,
            ) -> ToolResultEvent:
                event.serialized_arguments["x"] += 1
                result = await call_next(event, ctx)
                result.result.parts[0].content += "!"
                return result

        recorded_args = MagicMock()

        @tool
        def my_tool(x: int) -> str:
            recorded_args(x)
            return f"{x}"

        tracking_config = TrackingConfig(
            TestConfig(
                ToolCallEvent(name="my_tool", arguments='{"x": 1}'),
                "done",
            ),
        )

        agent = Agent(
            "",
            config=tracking_config,
            tools=[my_tool],
            middleware=[MutatingToolMiddleware, MutatingToolMiddleware, MutatingToolMiddleware],
        )

        await agent.ask("Hi!")

        recorded_args.assert_called_once_with(4)

        tool_results_event = tracking_config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0].content == "4!!!"

    @pytest.mark.asyncio()
    async def test_tool_local_then_agent_middleware_order(self, mock: MagicMock) -> None:
        def hook_factory(pos: int) -> ToolMiddleware:
            async def _hook(
                call_next: ToolExecution,
                event: ToolCallEvent,
                ctx: Context,
            ) -> ToolResultEvent:
                mock.enter(pos)
                r = await call_next(event, ctx)
                mock.exit(pos)
                return r

            return _hook

        @tool(middleware=[hook_factory(2), hook_factory(3)])
        def my_tool() -> str:
            return "ok"

        agent = Agent(
            "",
            config=TestConfig(
                ToolCallEvent(name="my_tool"),
                "result",
            ),
            tools=[my_tool],
            middleware=[Middleware(OrderingMiddleware, mock=mock, position=1)],
        )

        await agent.ask("Hi!")

        assert [c.args[0] for c in mock.enter.call_args_list] == [1, 2, 3]
        assert [c.args[0] for c in mock.exit.call_args_list] == [3, 2, 1]


class TestToolMiddlewareRegistration:
    @pytest.mark.asyncio()
    async def test_agent_tool_consumes_middleware(self, mock: MagicMock) -> None:
        async def tool_hook(
            call_next: ToolExecution,
            event: ToolCallEvent,
            ctx: Context,
        ) -> ToolResultEvent:
            mock.tool_middleware()
            return await call_next(event, ctx)

        agent = Agent(
            "",
            config=TestConfig(
                ToolCallEvent(name="my_tool"),
                "result",
            ),
        )

        @agent.tool(middleware=[tool_hook])
        def my_tool() -> str:
            return "from agent.tool"

        await agent.ask("Hi!")

        mock.tool_middleware.assert_called_once()

    @pytest.mark.asyncio()
    async def test_toolkit_tool_consumes_middleware(self, mock: MagicMock) -> None:
        async def tool_hook(
            call_next: ToolExecution,
            event: ToolCallEvent,
            ctx: Context,
        ) -> ToolResultEvent:
            mock.tool_middleware()
            return await call_next(event, ctx)

        tk = Toolkit()

        @tk.tool(middleware=[tool_hook])
        def my_tool() -> str:
            return "from toolkit.tool"

        agent = Agent(
            "",
            config=TestConfig(
                ToolCallEvent(name="my_tool"),
                "result",
            ),
            tools=[tk],
        )

        await agent.ask("Hi!")

        mock.tool_middleware.assert_called_once()
