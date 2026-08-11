# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

from ag2 import Agent, Context
from ag2.events import (
    BaseEvent,
    HumanInputRequest,
    HumanMessage,
    ModelResponse,
    ToolCallEvent,
    ToolResultEvent,
)
from ag2.middleware import (
    AgentTurn,
    BaseMiddleware,
    ConditionalMiddleware,
    HumanInputHook,
    LLMCall,
    Middleware,
    ToolExecution,
)
from ag2.testing import TestConfig


class TrackingMiddleware(BaseMiddleware):
    def __init__(
        self,
        event: BaseEvent,
        ctx: Context,
        mock: MagicMock,
    ) -> None:
        super().__init__(event, ctx)
        self.mock = mock

    async def on_turn(
        self,
        call_next: AgentTurn,
        event: BaseEvent,
        ctx: Context,
    ) -> ModelResponse:
        self.mock.on_turn(event.__class__.__name__)
        return await call_next(event, ctx)

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: ToolCallEvent,
        ctx: Context,
    ) -> ToolResultEvent:
        self.mock.on_tool(event.name)
        return await call_next(event, ctx)

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        ctx: Context,
    ) -> ModelResponse:
        self.mock.on_llm(len(events))
        return await call_next(events, ctx)

    async def on_human_input(
        self,
        call_next: HumanInputHook,
        event: HumanInputRequest,
        ctx: Context,
    ) -> HumanMessage:
        self.mock.on_human(event.content)
        return await call_next(event, ctx)


@pytest.mark.asyncio()
class TestConditionalMiddleware:
    async def test_tool_condition_activates_on_matching_tool(self, mock: MagicMock) -> None:
        """The motivating example from #2781: condition on ToolCallEvent.name."""

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
                    condition=ToolCallEvent.name == "my_tool",
                ),
            ],
        )

        await agent.ask("Hi!")

        mock.on_tool.assert_called_once_with("my_tool")

    async def test_tool_condition_skips_non_matching_tool(self, mock: MagicMock) -> None:
        """Condition targets a different tool name — middleware should not fire."""

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
                    condition=ToolCallEvent.name == "other_tool",
                ),
            ],
        )

        await agent.ask("Hi!")

        mock.on_tool.assert_not_called()

    async def test_type_condition_activates_per_hook(self, mock: MagicMock) -> None:
        """Bare event type as condition: activates on_tool_execution but not on_turn."""

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
                    condition=ToolCallEvent,
                ),
            ],
        )

        await agent.ask("Hi!")

        mock.on_tool.assert_called_once_with("my_tool")
        mock.on_turn.assert_not_called()

    async def test_llm_call_skipped_when_condition_does_not_match(self, mock: MagicMock) -> None:
        """on_llm_call checks condition against initial event — skipped when it doesn't match."""
        agent = Agent(
            "",
            config=TestConfig("result"),
            middleware=[
                ConditionalMiddleware(
                    Middleware(TrackingMiddleware, mock=mock),
                    condition=ToolCallEvent,
                ),
            ],
        )

        await agent.ask("Hi!")

        mock.on_turn.assert_not_called()
        mock.on_llm.assert_not_called()

    async def test_llm_call_delegates_when_condition_matches(self, mock: MagicMock) -> None:
        """on_llm_call delegates when condition matches the initial event."""
        agent = Agent(
            "",
            config=TestConfig("result"),
            middleware=[
                ConditionalMiddleware(
                    Middleware(TrackingMiddleware, mock=mock),
                    condition=BaseEvent,
                ),
            ],
        )

        await agent.ask("Hi!")

        mock.on_turn.assert_called()
        mock.on_llm.assert_called()

    async def test_inverted_condition(self, mock: MagicMock) -> None:
        """~ToolCallEvent activates on_turn but not on_tool_execution."""

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
                    condition=~(ToolCallEvent.name == "my_tool"),
                ),
            ],
        )

        await agent.ask("Hi!")

        mock.on_turn.assert_called()
        mock.on_tool.assert_not_called()

    async def test_composed_or_condition(self, mock: MagicMock) -> None:
        """Composed condition from issue: (name == 'a') | (name == 'b')."""
        condition = (ToolCallEvent.name == "tool_a") | (ToolCallEvent.name == "tool_b")

        def tool_a() -> str:
            return "a"

        def tool_b() -> str:
            return "b"

        def tool_c() -> str:
            return "c"

        agent = Agent(
            "",
            config=TestConfig(
                [
                    ToolCallEvent(name="tool_a"),
                    ToolCallEvent(name="tool_b"),
                    ToolCallEvent(name="tool_c"),
                ],
                "result",
            ),
            tools=[tool_a, tool_b, tool_c],
            middleware=[
                ConditionalMiddleware(
                    Middleware(TrackingMiddleware, mock=mock),
                    condition=condition,
                ),
            ],
        )

        await agent.ask("Hi!")

        assert mock.on_tool.call_count == 2
        mock.on_tool.assert_any_call("tool_a")
        mock.on_tool.assert_any_call("tool_b")

    async def test_human_input_activates_when_condition_matches(self, mock: MagicMock) -> None:
        """on_human_input fires when condition matches HumanInputRequest."""

        async def my_tool(ctx: Context) -> str:
            await ctx.input("Approve?", timeout=1.0)
            return "done"

        def hitl_hook(event: HumanInputRequest) -> HumanMessage:
            return HumanMessage("yes")

        agent = Agent(
            "",
            config=TestConfig(
                ToolCallEvent(name="my_tool"),
                "result",
            ),
            tools=[my_tool],
            hitl_hook=hitl_hook,
            middleware=[
                ConditionalMiddleware(
                    Middleware(TrackingMiddleware, mock=mock),
                    condition=HumanInputRequest,
                ),
            ],
        )

        await agent.ask("Hi!")

        mock.on_human.assert_called_once_with("Approve?")
        mock.on_turn.assert_not_called()
        mock.on_tool.assert_not_called()

    async def test_human_input_skipped_when_condition_does_not_match(self, mock: MagicMock) -> None:
        """on_human_input skipped when condition targets a different event type."""

        async def my_tool(ctx: Context) -> str:
            await ctx.input("Approve?", timeout=1.0)
            return "done"

        def hitl_hook(event: HumanInputRequest) -> HumanMessage:
            return HumanMessage("yes")

        agent = Agent(
            "",
            config=TestConfig(
                ToolCallEvent(name="my_tool"),
                "result",
            ),
            tools=[my_tool],
            hitl_hook=hitl_hook,
            middleware=[
                ConditionalMiddleware(
                    Middleware(TrackingMiddleware, mock=mock),
                    condition=ToolCallEvent.name == "other_tool",
                ),
            ],
        )

        await agent.ask("Hi!")

        mock.on_human.assert_not_called()
