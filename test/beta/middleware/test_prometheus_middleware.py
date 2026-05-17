# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for PrometheusMiddleware.

Each test gets a fresh CollectorRegistry() to avoid duplicate-registration errors
between tests and the global default registry.
"""

from collections.abc import Iterable, Sequence
from typing import Any

import pytest

pytest.importorskip("prometheus_client")

from prometheus_client import CollectorRegistry

from autogen.beta import Agent
from autogen.beta.annotations import ConversationContext
from autogen.beta.events import ModelMessage, ModelResponse, ToolCallEvent, ToolCallsEvent, Usage
from autogen.beta.middleware.builtin.prometheus import PrometheusMiddleware
from autogen.beta.testing import TestConfig
from autogen.beta.tools import tool
from autogen.beta.tools.schemas import ToolSchema


def _get_sample(registry: CollectorRegistry, metric_name: str, labels: dict[str, str]) -> float | None:
    """Return the current sample value for a metric+label combination, or None if not found."""
    for metric in registry.collect():
        if metric.name == metric_name:
            for sample in metric.samples:
                if all(sample.labels.get(k) == v for k, v in labels.items()):
                    return sample.value
    return None


def _get_count(registry: CollectorRegistry, metric_name: str, labels: dict[str, str]) -> float:
    """Return total counter value (sum across _total suffix variants)."""
    for metric in registry.collect():
        if metric.name in (metric_name, f"{metric_name}_total"):
            for sample in metric.samples:
                if sample.name.endswith("_total") and all(sample.labels.get(k) == v for k, v in labels.items()):
                    return sample.value
    return 0.0


@pytest.mark.asyncio
async def test_turn_duration_observed() -> None:
    registry = CollectorRegistry()
    agent = Agent(
        "assistant",
        config=TestConfig(ModelResponse(ModelMessage("Hi!"))),
        middleware=[PrometheusMiddleware(agent_name="assistant", registry=registry)],
    )

    await agent.ask("Hello")

    # turn_duration histogram should have count=1
    count = _get_sample(registry, "ag2_turn_duration_seconds", {"agent_name": "assistant", "le": "+Inf"})
    assert count == 1.0


@pytest.mark.asyncio
async def test_turn_error_increments_counter() -> None:
    registry = CollectorRegistry()

    class _ErrorClient:
        async def __call__(
            self,
            messages: Sequence[Any],
            context: ConversationContext,
            *,
            tools: Iterable[ToolSchema],
            response_schema: Any,
            serializer: Any,
        ) -> ModelResponse:
            raise RuntimeError("simulated LLM failure")

    class _ErrorConfig:
        def create(self) -> _ErrorClient:
            return _ErrorClient()

    agent = Agent(
        "assistant",
        config=_ErrorConfig(),
        middleware=[PrometheusMiddleware(agent_name="assistant", registry=registry)],
    )

    with pytest.raises(Exception):
        await agent.ask("Hello")

    count = _get_count(registry, "ag2_turn_errors", {"agent_name": "assistant"})
    assert count == 1.0


@pytest.mark.asyncio
async def test_llm_call_duration_and_tokens() -> None:
    registry = CollectorRegistry()
    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                message=ModelMessage("Hi!"),
                usage=Usage(prompt_tokens=10, completion_tokens=5),
            )
        ),
        middleware=[PrometheusMiddleware(agent_name="assistant", registry=registry)],
    )

    await agent.ask("Hello")

    # LLM call duration histogram should have count=1
    llm_count = _get_sample(registry, "ag2_llm_call_duration_seconds", {"agent_name": "assistant", "le": "+Inf"})
    assert llm_count == 1.0

    # Input tokens
    input_tokens = _get_count(registry, "ag2_llm_tokens", {"agent_name": "assistant", "token_type": "input"})
    assert input_tokens == 10.0

    # Output tokens
    output_tokens = _get_count(registry, "ag2_llm_tokens", {"agent_name": "assistant", "token_type": "output"})
    assert output_tokens == 5.0


@pytest.mark.asyncio
async def test_tool_call_counted_on_success() -> None:
    registry = CollectorRegistry()

    @tool
    async def echo(msg: str) -> str:
        """Return msg."""
        return msg

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(tool_calls=ToolCallsEvent(calls=[ToolCallEvent(name="echo", arguments='{"msg": "hello"}')])),
            ModelResponse(message=ModelMessage("done")),
        ),
        tools=[echo],
        middleware=[PrometheusMiddleware(agent_name="assistant", registry=registry)],
    )

    await agent.ask("echo hello")

    success_count = _get_count(
        registry, "ag2_tool_calls", {"agent_name": "assistant", "tool_name": "echo", "status": "success"}
    )
    assert success_count == 1.0

    # tool duration histogram count
    tool_dur_count = _get_sample(
        registry, "ag2_tool_execution_duration_seconds", {"agent_name": "assistant", "tool_name": "echo", "le": "+Inf"}
    )
    assert tool_dur_count == 1.0


@pytest.mark.asyncio
async def test_registry_isolation_between_agents() -> None:
    """Two agents with separate registries do not share metric state."""
    registry_a = CollectorRegistry()
    registry_b = CollectorRegistry()

    agent_a = Agent(
        "agent-a",
        config=TestConfig(ModelResponse(ModelMessage("A reply"))),
        middleware=[PrometheusMiddleware(agent_name="agent-a", registry=registry_a)],
    )
    Agent(
        "agent-b",
        config=TestConfig(ModelResponse(ModelMessage("B reply"))),
        middleware=[PrometheusMiddleware(agent_name="agent-b", registry=registry_b)],
    )

    await agent_a.ask("hello")
    await agent_a.ask("hello again")

    a_count = _get_sample(registry_a, "ag2_turn_duration_seconds", {"agent_name": "agent-a", "le": "+Inf"})
    b_count = _get_sample(registry_b, "ag2_turn_duration_seconds", {"agent_name": "agent-b", "le": "+Inf"})

    assert a_count == 2.0
    assert b_count is None  # agent-b was never run


@pytest.mark.asyncio
async def test_same_registry_reused_across_instances() -> None:
    """Two PrometheusMiddleware instances sharing a registry don't double-register metrics."""
    registry = CollectorRegistry()

    # Creating two instances with the same registry should not raise
    mw1 = PrometheusMiddleware(agent_name="agent-1", registry=registry)
    mw2 = PrometheusMiddleware(agent_name="agent-2", registry=registry)

    # They share the same bundle (same registry key)
    assert mw1._bundle is mw2._bundle

    agent1 = Agent(
        "agent-1",
        config=TestConfig(ModelResponse(ModelMessage("hi"))),
        middleware=[mw1],
    )
    agent2 = Agent(
        "agent-2",
        config=TestConfig(ModelResponse(ModelMessage("hi"))),
        middleware=[mw2],
    )

    await agent1.ask("hello")
    await agent2.ask("hello")

    # Both have turn_duration observations, via different agent_name labels
    a1_count = _get_sample(registry, "ag2_turn_duration_seconds", {"agent_name": "agent-1", "le": "+Inf"})
    a2_count = _get_sample(registry, "ag2_turn_duration_seconds", {"agent_name": "agent-2", "le": "+Inf"})
    assert a1_count == 1.0
    assert a2_count == 1.0


@pytest.mark.asyncio
async def test_import_accessible_from_public_api() -> None:
    from autogen.beta.middleware import PrometheusMiddleware as PrometheusMiddlewareAlias

    assert PrometheusMiddlewareAlias is PrometheusMiddleware
