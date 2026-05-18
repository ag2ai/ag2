# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent, Context
from autogen.beta.events import BaseEvent, ToolCallEvent, ToolResultEvent
from autogen.beta.events.conditions import TypeCondition
from autogen.beta.middleware import BaseMiddleware, ConditionalMiddleware, Middleware, ToolExecution
from autogen.beta.testing import TestConfig


class TrackingMiddleware(BaseMiddleware):
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
        self.mock.on_tool(event.name)
        return await call_next(event, ctx)


@pytest.mark.asyncio()
class TestConditionalMiddleware:
    async def test_activates_when_condition_matches(self, mock: MagicMock) -> None:
        condition = TypeCondition(BaseEvent)

        def my_tool() -> str:
            return "done"

        agent = Agent(
            "",
            config=TestConfig(
                ToolCallEvent(name="my_tool"),
                "result",
            ),
            tools=[my_tool],
            middleware=[
                ConditionalMiddleware(
                    Middleware(TrackingMiddleware, mock=mock),
                    condition=condition,
                ),
            ],
        )

        await agent.ask("Hi!")

        mock.on_tool.assert_called_once_with("my_tool")

    async def test_skips_when_condition_does_not_match(self, mock: MagicMock) -> None:
        condition = TypeCondition(ToolCallEvent)

        def my_tool() -> str:
            return "done"

        agent = Agent(
            "",
            config=TestConfig(
                ToolCallEvent(name="my_tool"),
                "result",
            ),
            tools=[my_tool],
            middleware=[
                ConditionalMiddleware(
                    Middleware(TrackingMiddleware, mock=mock),
                    condition=condition,
                ),
            ],
        )

        await agent.ask("Hi!")

        mock.on_tool.assert_not_called()

    async def test_inverted_condition(self, mock: MagicMock) -> None:
        condition = ~TypeCondition(ToolCallEvent)

        def my_tool() -> str:
            return "done"

        agent = Agent(
            "",
            config=TestConfig(
                ToolCallEvent(name="my_tool"),
                "result",
            ),
            tools=[my_tool],
            middleware=[
                ConditionalMiddleware(
                    Middleware(TrackingMiddleware, mock=mock),
                    condition=condition,
                ),
            ],
        )

        await agent.ask("Hi!")

        mock.on_tool.assert_called_once_with("my_tool")

    async def test_multiple_conditional_middleware(self, mock: MagicMock) -> None:
        always_match = TypeCondition(BaseEvent)
        never_match = TypeCondition(ToolCallEvent)

        def my_tool() -> str:
            return "done"

        mock_active = MagicMock()
        mock_skipped = MagicMock()

        agent = Agent(
            "",
            config=TestConfig(
                ToolCallEvent(name="my_tool"),
                "result",
            ),
            tools=[my_tool],
            middleware=[
                ConditionalMiddleware(
                    Middleware(TrackingMiddleware, mock=mock_active),
                    condition=always_match,
                ),
                ConditionalMiddleware(
                    Middleware(TrackingMiddleware, mock=mock_skipped),
                    condition=never_match,
                ),
            ],
        )

        await agent.ask("Hi!")

        mock_active.on_tool.assert_called_once_with("my_tool")
        mock_skipped.on_tool.assert_not_called()
