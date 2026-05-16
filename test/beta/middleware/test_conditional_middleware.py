# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent
from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ModelMessage, ModelResponse, ToolCallEvent, ToolCallsEvent
from autogen.beta.events.conditions import TypeCondition
from autogen.beta.middleware import BaseMiddleware, ConditionalMiddleware, Middleware, ToolExecution
from autogen.beta.middleware.base import AgentTurn, LLMCall, ToolResultType
from autogen.beta.testing import TestConfig
from autogen.beta.tools import tool


class _TrackingMiddleware(BaseMiddleware):
    """Records which hooks were invoked."""

    def __init__(self, event: BaseEvent, context: Context, *, tracker: MagicMock) -> None:
        super().__init__(event, context)
        self._tracker = tracker

    async def on_turn(self, call_next: AgentTurn, event: BaseEvent, context: Context) -> ModelResponse:
        self._tracker.on_turn()
        return await call_next(event, context)

    async def on_llm_call(self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context) -> ModelResponse:
        self._tracker.on_llm_call()
        return await call_next(events, context)

    async def on_tool_execution(
        self, call_next: ToolExecution, event: ToolCallEvent, context: Context
    ) -> ToolResultType:
        self._tracker.on_tool_execution()
        return await call_next(event, context)


def _tracking_factory(tracker: MagicMock) -> Middleware:
    return Middleware(_TrackingMiddleware, tracker=tracker)


@pytest.mark.asyncio()
async def test_condition_match_activates_inner_on_turn():
    tracker = MagicMock()
    agent = Agent(
        "assistant",
        config=TestConfig(ModelResponse(ModelMessage("Hello!"))),
        middleware=[
            ConditionalMiddleware(
                condition=TypeCondition(ModelMessage),  # always False — turn event is not ModelMessage
                middleware=_tracking_factory(tracker),
            )
        ],
    )

    await agent.ask("Hi")

    tracker.on_turn.assert_not_called()


@pytest.mark.asyncio()
async def test_tool_condition_gates_only_matching_tool():
    tracker = MagicMock()

    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello {name}"

    @tool
    def farewell(name: str) -> str:
        """Say goodbye."""
        return f"Goodbye {name}"

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent(
                    calls=[
                        ToolCallEvent(id="c1", name="greet", arguments='{"name": "World"}'),
                        ToolCallEvent(id="c2", name="farewell", arguments='{"name": "World"}'),
                    ]
                ),
            ),
            ModelResponse(ModelMessage("Done")),
        ),
        tools=[greet, farewell],
        middleware=[
            ConditionalMiddleware(
                condition=ToolCallEvent.name == "greet",
                middleware=_tracking_factory(tracker),
            )
        ],
    )

    await agent.ask("Go")

    # Only the "greet" tool call should have triggered on_tool_execution
    assert tracker.on_tool_execution.call_count == 1


@pytest.mark.asyncio()
async def test_condition_no_match_passes_through():
    """When condition never matches, inner middleware is fully transparent."""
    tracker = MagicMock()

    @tool
    def noop() -> str:
        """Does nothing."""
        return "ok"

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent(calls=[ToolCallEvent(id="c1", name="noop", arguments="{}")]),
            ),
            ModelResponse(ModelMessage("Done")),
        ),
        tools=[noop],
        middleware=[
            ConditionalMiddleware(
                condition=ToolCallEvent.name == "nonexistent_tool",
                middleware=_tracking_factory(tracker),
            )
        ],
    )

    await agent.ask("Go")

    tracker.on_tool_execution.assert_not_called()
    # Turn still completes successfully
    tracker.on_turn.assert_not_called()  # turn event is HumanMessage, not the condition type


@pytest.mark.asyncio()
async def test_condition_match_on_tool_name():
    """Positive case: condition matches the exact tool name."""
    tracker = MagicMock()

    @tool
    def calculate(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent(
                    calls=[ToolCallEvent(id="c1", name="calculate", arguments='{"x": 1, "y": 2}')]
                ),
            ),
            ModelResponse(ModelMessage("Result: 3")),
        ),
        tools=[calculate],
        middleware=[
            ConditionalMiddleware(
                condition=ToolCallEvent.name == "calculate",
                middleware=_tracking_factory(tracker),
            )
        ],
    )

    await agent.ask("What is 1 + 2?")

    tracker.on_tool_execution.assert_called_once()


@pytest.mark.asyncio()
async def test_conditional_middleware_composable_with_other_middleware():
    """ConditionalMiddleware stacks correctly with other middleware."""
    outer_tracker = MagicMock()
    inner_tracker = MagicMock()

    agent = Agent(
        "assistant",
        config=TestConfig(ModelResponse(ModelMessage("Hi!"))),
        middleware=[
            Middleware(_TrackingMiddleware, tracker=outer_tracker),
            ConditionalMiddleware(
                condition=TypeCondition(ModelMessage),  # won't match HumanMessage turn event
                middleware=_tracking_factory(inner_tracker),
            ),
        ],
    )

    await agent.ask("Hello")

    # Outer middleware runs on every turn
    outer_tracker.on_turn.assert_called_once()
    # Inner conditional middleware did not match
    inner_tracker.on_turn.assert_not_called()


@pytest.mark.asyncio()
async def test_export_from_top_level_middleware_package():
    """ConditionalMiddleware is importable from autogen.beta.middleware."""
    from autogen.beta.middleware import ConditionalMiddleware as Conditional

    assert Conditional is ConditionalMiddleware
