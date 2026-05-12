# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from dirty_equals import IsFloat
from prometheus_client import CollectorRegistry
from typing_extensions import Self

from autogen.beta import Agent
from autogen.beta.annotations import Context
from autogen.beta.config import LLMClient, ModelConfig, ModelProvider
from autogen.beta.events import (
    BaseEvent,
    HumanInputRequest,
    HumanMessage,
    ModelMessage,
    ModelResponse,
    ToolCallEvent,
    Usage,
)
from autogen.beta.middleware import MetricsMiddleware, RetryMiddleware
from autogen.beta.testing import TestConfig


@pytest.fixture
def registry() -> CollectorRegistry:
    return CollectorRegistry()


def test_rejects_multiple_metrics_middlewares_for_same_registry(registry: CollectorRegistry) -> None:
    MetricsMiddleware(registry=registry)

    with pytest.raises(ValueError, match="same CollectorRegistry"):
        MetricsMiddleware(registry=registry)


@pytest.mark.asyncio()
@pytest.mark.parametrize(
    ("model_provider", "expected_provider_label"),
    (
        pytest.param(ModelProvider.ANTHROPIC, "anthropic", id="anthropic"),
        pytest.param(ModelProvider.DASHSCOPE, "dashscope", id="dashscope"),
        pytest.param(ModelProvider.GEMINI, "gemini", id="gemini"),
        pytest.param(ModelProvider.OLLAMA, "ollama", id="ollama"),
        pytest.param(ModelProvider.OPENAI, "openai", id="openai"),
        pytest.param(ModelProvider.VERTEXAI, "vertexai", id="vertexai"),
    ),
)
async def test_records_llm_call_and_token_success_metrics(
    registry: CollectorRegistry,
    model_provider: ModelProvider,
    expected_provider_label: str,
) -> None:
    agent = Agent(
        "llm-metrics-agent",
        config=TestConfig(
            ModelResponse(
                ModelMessage("Hello!"),
                usage=Usage(
                    prompt_tokens=10,
                    completion_tokens=5,
                    cache_read_input_tokens=3,
                    cache_creation_input_tokens=2,
                ),
                provider="response-provider",
                model="response-model",
                finish_reason="stop",
            ),
            provider=model_provider,
            model="metrics-model",
        ),
        middleware=[MetricsMiddleware(registry=registry)],
    )

    await agent.ask("Hi")

    llm_success_labels = {
        "agent": "llm-metrics-agent",
        "provider": expected_provider_label,
        "model": "metrics-model",
        "outcome": "success",
        "error_type": "",
    }
    token_labels = {"agent": "llm-metrics-agent", "provider": expected_provider_label, "model": "metrics-model"}

    assert registry.get_sample_value("ag2_llm_tokens_total", {**token_labels, "token_type": "input"}) == 10.0
    assert registry.get_sample_value("ag2_llm_tokens_total", {**token_labels, "token_type": "output"}) == 5.0
    assert registry.get_sample_value("ag2_llm_tokens_total", {**token_labels, "token_type": "cache_read_input"}) == 3.0
    assert (
        registry.get_sample_value("ag2_llm_tokens_total", {**token_labels, "token_type": "cache_creation_input"}) == 2.0
    )

    assert (
        registry.get_sample_value(
            "ag2_llm_calls_total",
            {
                **llm_success_labels,
                "finish_reason": "stop",
            },
        )
        == 1.0
    )
    assert registry.get_sample_value("ag2_llm_call_duration_seconds_count", llm_success_labels) == 1.0
    assert registry.get_sample_value("ag2_llm_call_duration_seconds_sum", llm_success_labels) == IsFloat()


@pytest.mark.asyncio()
async def test_records_llm_call_error_metrics(registry: CollectorRegistry) -> None:
    agent = Agent(
        "llm-error-agent",
        config=_FailOnceConfig(ValueError),
        middleware=[MetricsMiddleware(registry=registry)],
    )

    with pytest.raises(ValueError):
        await agent.ask("Hi")

    llm_error_labels = {
        "agent": "llm-error-agent",
        "provider": "openai",
        "model": "test-model",
        "outcome": "error",
        "error_type": "ValueError",
    }

    assert (
        registry.get_sample_value(
            "ag2_llm_calls_total",
            {
                **llm_error_labels,
                "finish_reason": "unknown",
            },
        )
        == 1.0
    )
    assert registry.get_sample_value("ag2_llm_call_duration_seconds_count", llm_error_labels) == 1.0
    assert registry.get_sample_value("ag2_llm_call_duration_seconds_sum", llm_error_labels) == IsFloat()


@pytest.mark.asyncio()
async def test_retries_record_each_llm_attempt(registry: CollectorRegistry) -> None:
    config = _FailOnceConfig(model="retry-model", provider=ModelProvider.OPENAI, exception_type=_TransientError)
    agent = Agent(
        "retry-agent",
        config=config,
        middleware=[
            RetryMiddleware(max_retries=1, retry_on=(_TransientError,)),
            MetricsMiddleware(registry=registry),
        ],
    )

    await agent.ask("Hi")

    assert config.client.call_count == 2
    assert (
        registry.get_sample_value(
            "ag2_llm_calls_total",
            {
                "agent": "retry-agent",
                "provider": "openai",
                "model": "retry-model",
                "outcome": "error",
                "finish_reason": "unknown",
                "error_type": "_TransientError",
            },
        )
        == 1.0
    )
    assert (
        registry.get_sample_value(
            "ag2_llm_calls_total",
            {
                "agent": "retry-agent",
                "provider": "openai",
                "model": "retry-model",
                "outcome": "success",
                "finish_reason": "stop",
                "error_type": "",
            },
        )
        == 1.0
    )


@pytest.mark.asyncio()
async def test_no_retry_record_each_llm_attempt_if_retry_middleware_after_metrics(registry: CollectorRegistry) -> None:
    config = _FailOnceConfig(model="retry-model", provider=ModelProvider.OPENAI, exception_type=_TransientError)
    agent = Agent(
        "retry-agent",
        config=config,
        middleware=[
            MetricsMiddleware(registry=registry),
            RetryMiddleware(max_retries=1, retry_on=(_TransientError,)),
        ],
    )

    await agent.ask("Hi")

    assert config.client.call_count == 2
    assert (
        registry.get_sample_value(
            "ag2_llm_calls_total",
            {
                "agent": "retry-agent",
                "provider": "openai",
                "model": "retry-model",
                "outcome": "error",
                "finish_reason": "unknown",
                "error_type": "_TransientError",
            },
        )
        is None
    )
    assert (
        registry.get_sample_value(
            "ag2_llm_calls_total",
            {
                "agent": "retry-agent",
                "provider": "openai",
                "model": "retry-model",
                "outcome": "success",
                "finish_reason": "stop",
                "error_type": "",
            },
        )
        == 1.0
    )


@pytest.mark.asyncio()
async def test_normalizes_missing_llm_labels_and_omits_zero_tokens(registry: CollectorRegistry) -> None:
    agent = Agent(
        "normalize-agent",
        config=TestConfig(
            ModelResponse(
                ModelMessage("Hello!"),
                usage=Usage(prompt_tokens=0, completion_tokens=5, cache_read_input_tokens=0),
                finish_reason=None,
            ),
            provider=ModelProvider.OPENAI,
            model="test-model",
        ),
        middleware=[MetricsMiddleware(registry=registry)],
    )

    await agent.ask("Hi")

    llm_success_labels = {
        "agent": "normalize-agent",
        "provider": "openai",
        "model": "test-model",
        "outcome": "success",
        "finish_reason": "unknown",
        "error_type": "",
    }
    token_labels = {"agent": "normalize-agent", "provider": "openai", "model": "test-model"}

    assert registry.get_sample_value("ag2_llm_calls_total", llm_success_labels) == 1.0
    assert registry.get_sample_value("ag2_llm_tokens_total", {**token_labels, "token_type": "input"}) is None
    assert registry.get_sample_value("ag2_llm_tokens_total", {**token_labels, "token_type": "cache_read_input"}) is None
    assert registry.get_sample_value("ag2_llm_tokens_total", {**token_labels, "token_type": "output"}) == 5.0


@pytest.mark.asyncio()
async def test_records_agent_turn_success_metrics(registry: CollectorRegistry) -> None:
    agent = Agent(
        "agent-turn-success-agent",
        config=TestConfig(
            ModelResponse(ModelMessage("Hello!")),
            provider=ModelProvider.OPENAI,
            model="test-model",
        ),
        middleware=[MetricsMiddleware(registry=registry)],
    )

    await agent.ask("Hi")

    agent_success_labels = {"agent": "agent-turn-success-agent", "outcome": "success", "error_type": ""}

    assert registry.get_sample_value("ag2_agent_turns_total", agent_success_labels) == 1.0
    assert registry.get_sample_value("ag2_agent_turn_duration_seconds_count", agent_success_labels) == 1.0
    assert registry.get_sample_value("ag2_agent_turn_duration_seconds_sum", agent_success_labels) == IsFloat()


@pytest.mark.asyncio()
async def test_records_agent_turn_error_metrics(registry: CollectorRegistry) -> None:
    agent = Agent(
        "agent-turn-error-agent",
        config=_FailOnceConfig(ValueError),
        middleware=[MetricsMiddleware(registry=registry)],
    )

    with pytest.raises(ValueError):
        await agent.ask("Hi")

    agent_error_labels = {"agent": "agent-turn-error-agent", "outcome": "error", "error_type": "ValueError"}

    assert registry.get_sample_value("ag2_agent_turns_total", agent_error_labels) == 1.0
    assert registry.get_sample_value("ag2_agent_turn_duration_seconds_count", agent_error_labels) == 1.0
    assert registry.get_sample_value("ag2_agent_turn_duration_seconds_sum", agent_error_labels) == IsFloat()


@pytest.mark.asyncio()
async def test_records_human_input_success_metrics(registry: CollectorRegistry) -> None:
    agent = Agent(
        "human-input-success-agent",
        config=TestConfig(
            ToolCallEvent(name="_collect_human_input", arguments="{}"),
            ModelResponse(ModelMessage("Done")),
            provider=ModelProvider.ANTHROPIC,
            model="test-model",
        ),
        tools=[_collect_human_input],
        hitl_hook=_return_human_input,
        middleware=[MetricsMiddleware(registry=registry)],
    )

    await agent.ask("Ask for input")

    human_input_success_labels = {"agent": "human-input-success-agent", "outcome": "success", "error_type": ""}

    assert registry.get_sample_value("ag2_human_input_requests_total", human_input_success_labels) == 1.0
    assert registry.get_sample_value("ag2_human_input_duration_seconds_count", human_input_success_labels) == 1.0
    assert registry.get_sample_value("ag2_human_input_duration_seconds_sum", human_input_success_labels) == IsFloat()


@pytest.mark.asyncio()
async def test_records_human_input_error_metrics(registry: CollectorRegistry) -> None:
    agent = Agent(
        "human-input-error-agent",
        config=TestConfig(
            ToolCallEvent(name="_collect_human_input", arguments="{}"),
            ModelResponse(ModelMessage("Done")),
            provider=ModelProvider.OPENAI,
            model="test-model",
        ),
        tools=[_collect_human_input],
        hitl_hook=_raise_timeout,
        middleware=[MetricsMiddleware(registry=registry)],
    )

    with pytest.raises(TimeoutError):
        await agent.ask("Ask for input")

    human_input_error_labels = {
        "agent": "human-input-error-agent",
        "outcome": "error",
        "error_type": "TimeoutError",
    }

    assert registry.get_sample_value("ag2_human_input_requests_total", human_input_error_labels) == 1.0
    assert registry.get_sample_value("ag2_human_input_duration_seconds_count", human_input_error_labels) == 1.0
    assert registry.get_sample_value("ag2_human_input_duration_seconds_sum", human_input_error_labels) == IsFloat()


@pytest.mark.asyncio()
async def test_records_cancelled_human_input_as_error(registry: CollectorRegistry) -> None:
    agent = Agent(
        "cancelled-human-input-agent",
        config=TestConfig(
            ToolCallEvent(name="_collect_human_input", arguments="{}"),
            ModelResponse(ModelMessage("Done")),
            provider=ModelProvider.OPENAI,
            model="test-model",
        ),
        tools=[_collect_human_input],
        hitl_hook=_raise_cancelled,
        middleware=[MetricsMiddleware(registry=registry)],
    )

    with pytest.raises(asyncio.CancelledError):
        await agent.ask("Ask for input")

    cancelled_labels = {
        "agent": "cancelled-human-input-agent",
        "outcome": "error",
        "error_type": "CancelledError",
    }
    assert registry.get_sample_value("ag2_human_input_requests_total", cancelled_labels) == 1.0
    assert registry.get_sample_value("ag2_human_input_duration_seconds_count", cancelled_labels) == 1.0
    assert registry.get_sample_value("ag2_human_input_duration_seconds_sum", cancelled_labels) == IsFloat()


@pytest.mark.asyncio()
async def test_records_tool_success_metrics(
    registry: CollectorRegistry,
) -> None:
    agent = Agent(
        "tool-success-agent",
        config=TestConfig(
            ToolCallEvent(name="successful_tool", arguments="{}"),
            ModelResponse(ModelMessage("Done")),
            provider=ModelProvider.OPENAI,
            model="test-model",
        ),
        middleware=[MetricsMiddleware(registry=registry)],
    )

    @agent.tool
    def successful_tool() -> str:
        return "result"

    await agent.ask("Call successful_tool")

    tool_success_labels = {
        "agent": "tool-success-agent",
        "tool": "successful_tool",
        "outcome": "success",
        "error_type": "",
    }

    assert registry.get_sample_value("ag2_tool_calls_total", tool_success_labels) == 1.0
    assert registry.get_sample_value("ag2_tool_duration_seconds_count", tool_success_labels) == 1.0
    assert registry.get_sample_value("ag2_tool_duration_seconds_sum", tool_success_labels) == IsFloat()


@pytest.mark.asyncio()
async def test_records_tool_error_metrics(registry: CollectorRegistry) -> None:
    agent = Agent(
        "tool-error-agent",
        config=TestConfig(
            ToolCallEvent(name="failing_tool", arguments="{}"),
            ModelResponse(ModelMessage("Done")),
            provider=ModelProvider.OPENAI,
            model="test-model",
        ),
        middleware=[MetricsMiddleware(registry=registry)],
    )

    @agent.tool
    def failing_tool() -> str:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await agent.ask("Call failing_tool")

    tool_error_labels = {
        "agent": "tool-error-agent",
        "tool": "failing_tool",
        "outcome": "error",
        "error_type": "ValueError",
    }

    assert registry.get_sample_value("ag2_tool_calls_total", tool_error_labels) == 1.0
    assert registry.get_sample_value("ag2_tool_duration_seconds_count", tool_error_labels) == 1.0
    assert registry.get_sample_value("ag2_tool_duration_seconds_sum", tool_error_labels) == IsFloat()


class _TransientError(Exception):
    pass


class _FailOnceClient(LLMClient):
    def __init__(self, exception_type: type[Exception]) -> None:
        self.call_count = 0
        self._exception_type = exception_type

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        **kwargs: Any,
    ) -> ModelResponse:
        self.call_count += 1
        if self.call_count == 1:
            raise self._exception_type("Fail")
        return ModelResponse(ModelMessage("Hello!"), finish_reason="stop")


class _FailOnceConfig(ModelConfig):
    def __init__(
        self,
        exception_type: type[Exception],
        provider: ModelProvider = ModelProvider.OPENAI,
        model: str = "test-model",
    ) -> None:
        self.client = _FailOnceClient(exception_type)
        self._provider = provider
        self._model = model

    @property
    def provider(self) -> ModelProvider:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    def copy(self) -> Self:
        return self

    def create(self) -> _FailOnceClient:
        return self.client


async def _collect_human_input(context: Context) -> str:
    return await context.input("Enter input", timeout=1.0)


def _return_human_input(event: HumanInputRequest) -> HumanMessage:
    return HumanMessage(content=f"User input for {event.content}")


def _raise_timeout(_: HumanInputRequest) -> HumanMessage:
    raise TimeoutError("input timed out")


def _raise_cancelled(_: HumanInputRequest) -> HumanMessage:
    raise asyncio.CancelledError
