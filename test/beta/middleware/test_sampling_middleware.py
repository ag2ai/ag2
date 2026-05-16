# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import pytest

from autogen.beta import Agent
from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ModelMessage, ModelResponse, ToolCallEvent, ToolCallsEvent
from autogen.beta.middleware import BaseMiddleware, Middleware, SamplingMiddleware, ToolExecution
from autogen.beta.middleware.base import AgentTurn, LLMCall, ToolResultType
from autogen.beta.testing import TestConfig
from autogen.beta.tools import tool


class _TrackingMiddleware(BaseMiddleware):
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
async def test_rate_zero_never_activates():
    """rate=0.0 — inner middleware is never invoked."""
    tracker = MagicMock()
    agent = Agent(
        "assistant",
        config=TestConfig(ModelResponse(ModelMessage("Hello!"))),
        middleware=[SamplingMiddleware(rate=0.0, middleware=_tracking_factory(tracker))],
    )

    await agent.ask("Hi")

    tracker.on_turn.assert_not_called()
    tracker.on_llm_call.assert_not_called()


@pytest.mark.asyncio()
async def test_rate_one_always_activates():
    """rate=1.0 — inner middleware runs on every turn."""
    tracker = MagicMock()
    agent = Agent(
        "assistant",
        config=TestConfig(ModelResponse(ModelMessage("Hello!"))),
        middleware=[SamplingMiddleware(rate=1.0, middleware=_tracking_factory(tracker))],
    )

    await agent.ask("Hi")

    tracker.on_turn.assert_called_once()


@pytest.mark.asyncio()
async def test_rate_one_activates_tool_execution():
    """rate=1.0 — on_tool_execution is called for every tool call."""
    tracker = MagicMock()

    @tool
    def greet(name: str) -> str:
        """Say hello."""
        return f"Hello {name}"

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent(calls=[ToolCallEvent(id="c1", name="greet", arguments='{"name": "World"}')]),
            ),
            ModelResponse(ModelMessage("Done")),
        ),
        tools=[greet],
        middleware=[SamplingMiddleware(rate=1.0, middleware=_tracking_factory(tracker))],
    )

    await agent.ask("Go")

    tracker.on_tool_execution.assert_called_once()


@pytest.mark.asyncio()
async def test_sampling_decision_consistent_within_turn():
    """All hooks in a sampled turn share the same decision (coherent tracing)."""
    tracker = MagicMock()

    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent(calls=[ToolCallEvent(id="c1", name="ping", arguments="{}")]),
            ),
            ModelResponse(ModelMessage("Done")),
        ),
        tools=[ping],
        middleware=[SamplingMiddleware(rate=1.0, middleware=_tracking_factory(tracker))],
    )

    await agent.ask("Ping")

    # Both on_turn and on_tool_execution must run (or both skip) — here rate=1.0 so both run.
    assert tracker.on_turn.call_count == 1
    assert tracker.on_tool_execution.call_count == 1


@pytest.mark.asyncio()
async def test_sampling_uses_random():
    """SamplingMiddleware draws a random float to decide per-turn activation."""
    tracker = MagicMock()
    agent = Agent(
        "assistant",
        config=TestConfig(ModelResponse(ModelMessage("Hello!"))),
        middleware=[SamplingMiddleware(rate=0.5, middleware=_tracking_factory(tracker))],
    )

    # Force random to return 0.3 < 0.5 → should activate
    with patch("autogen.beta.middleware.builtin.sampling.random.random", return_value=0.3):
        await agent.ask("Hi")
    tracker.on_turn.assert_called_once()

    tracker.reset_mock()

    # Force random to return 0.7 >= 0.5 → should NOT activate
    with patch("autogen.beta.middleware.builtin.sampling.random.random", return_value=0.7):
        await agent.ask("Hi")
    tracker.on_turn.assert_not_called()


def test_invalid_rate_raises():
    """rate outside [0, 1] raises ValueError at construction time."""
    with pytest.raises(ValueError, match="rate"):
        SamplingMiddleware(rate=1.5, middleware=_tracking_factory(MagicMock()))
    with pytest.raises(ValueError, match="rate"):
        SamplingMiddleware(rate=-0.1, middleware=_tracking_factory(MagicMock()))


@pytest.mark.asyncio()
async def test_export_from_top_level_middleware_package():
    """SamplingMiddleware is importable from autogen.beta.middleware."""
    from autogen.beta.middleware import SamplingMiddleware as Sampling

    assert Sampling is SamplingMiddleware
