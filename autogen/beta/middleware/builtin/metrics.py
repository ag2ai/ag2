# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import typing
from collections.abc import Sequence
from enum import Enum
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autogen.beta import Agent
    from autogen.beta.annotations import Context
    from autogen.beta.config import ModelConfig
    from autogen.beta.events import (
        BaseEvent,
        HumanInputRequest,
        HumanMessage,
        ModelResponse,
        ToolCallEvent,
    )
    from autogen.beta.middleware.base import (
        AgentTurn,
        HumanInputHook,
        LLMCall,
    )

from autogen.beta.events import (
    ToolErrorEvent,
)
from autogen.beta.middleware import BaseMiddleware
from autogen.beta.middleware.base import (
    MiddlewareFactory,
    ToolExecution,
    ToolResultType,
)
from autogen.beta.utils import AGENT_CONTEXT_DEPENDENCY_KEY, MODEL_CONFIG_CONTEXT_DEPENDENCY_KEY

try:
    from prometheus_client import CollectorRegistry, Counter, Histogram
except ImportError as _err:
    raise ImportError(
        "prometheus_client is required for MetricsMiddleware. Install with: pip install ag2[metrics]"
    ) from _err


class Outcome(Enum):
    SUCCESS = "success"
    ERROR = "error"


class _MetricContainer:
    def __init__(self, registry: CollectorRegistry) -> None:
        self.llm_calls_total = Counter(
            name="ag2_llm_calls_total",
            documentation="Total number of LLM calls",
            labelnames=["agent", "provider", "model", "outcome", "finish_reason", "error_type"],
            registry=registry,
        )

        self.llm_call_duration_seconds = Histogram(
            name="ag2_llm_call_duration_seconds",
            documentation="Duration of LLM calls in seconds",
            labelnames=["agent", "provider", "model", "outcome", "error_type"],
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, float("inf")],
            registry=registry,
        )

        self.llm_tokens_total = Counter(
            name="ag2_llm_tokens_total",
            documentation="Total number of LLM tokens used",
            labelnames=["agent", "provider", "model", "token_type"],
            registry=registry,
        )

        self.tool_calls_total = Counter(
            name="ag2_tool_calls_total",
            documentation="Total number of tool calls",
            labelnames=["agent", "tool", "outcome", "error_type"],
            registry=registry,
        )

        self.tool_duration_seconds = Histogram(
            name="ag2_tool_duration_seconds",
            documentation="Duration of tool execution in seconds",
            labelnames=["agent", "tool", "outcome", "error_type"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")],
            registry=registry,
        )

        self.agent_turns_total = Counter(
            name="ag2_agent_turns_total",
            documentation="Total number of agent turns",
            labelnames=["agent", "outcome", "error_type"],
            registry=registry,
        )

        self.agent_turn_duration_seconds = Histogram(
            name="ag2_agent_turn_duration_seconds",
            documentation="Duration of agent turns in seconds",
            labelnames=["agent", "outcome", "error_type"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0, float("inf")],
            registry=registry,
        )

        self.human_input_requests_total = Counter(
            name="ag2_human_input_requests_total",
            documentation="Total number of human input requests",
            labelnames=["agent", "outcome", "error_type"],
            registry=registry,
        )

        self.human_input_duration_seconds = Histogram(
            name="ag2_human_input_duration_seconds",
            documentation="Duration of human input requests in seconds",
            labelnames=["agent", "outcome", "error_type"],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0, 1800.0, 3600.0, float("inf")],
            registry=registry,
        )


class MetricsMiddleware(MiddlewareFactory):
    def __init__(self, registry: CollectorRegistry) -> None:
        try:
            self._metrics = _MetricContainer(registry)
        except ValueError as exc:
            if "Duplicated timeseries" not in str(exc):
                raise
            raise ValueError(
                "MetricsMiddleware cannot be created more than once for the same CollectorRegistry. "
                "Create one MetricsMiddleware per CollectorRegistry and reuse that middleware instance across agents."
            ) from exc

    def __call__(self, event: "BaseEvent", context: "Context") -> BaseMiddleware:
        return _MetricsMiddleware(event, context, self._metrics)


class _MetricsMiddleware(BaseMiddleware):
    def __init__(
        self,
        event: "BaseEvent",
        context: "Context",
        metrics: _MetricContainer,
    ) -> None:
        super().__init__(event, context)
        self._metrics = metrics
        self._agent_name = self._normalize_label(self._get_agent_name(context))

    async def on_turn(
        self,
        call_next: "AgentTurn",
        event: "BaseEvent",
        context: "Context",
    ) -> "ModelResponse":
        start_time = perf_counter()

        try:
            response = await call_next(event, context)
        except BaseException as e:
            error_type = self._get_error_type(e)
            self._metrics.agent_turns_total.labels(
                agent=self._agent_name, outcome=Outcome.ERROR.value, error_type=error_type
            ).inc()
            self._metrics.agent_turn_duration_seconds.labels(
                agent=self._agent_name, outcome=Outcome.ERROR.value, error_type=error_type
            ).observe(perf_counter() - start_time)
            raise

        self._metrics.agent_turns_total.labels(
            agent=self._agent_name, outcome=Outcome.SUCCESS.value, error_type=""
        ).inc()
        self._metrics.agent_turn_duration_seconds.labels(
            agent=self._agent_name, outcome=Outcome.SUCCESS.value, error_type=""
        ).observe(perf_counter() - start_time)
        return response

    async def on_llm_call(
        self,
        call_next: "LLMCall",
        events: Sequence["BaseEvent"],
        context: "Context",
    ) -> "ModelResponse":
        provider, model = self._get_model_labels(context)
        start_time = perf_counter()

        try:
            response = await call_next(events, context)
        except BaseException as exc:
            error_type = self._get_error_type(exc)

            self._metrics.llm_call_duration_seconds.labels(
                agent=self._agent_name,
                provider=provider,
                model=model,
                outcome=Outcome.ERROR.value,
                error_type=error_type,
            ).observe(perf_counter() - start_time)

            self._metrics.llm_calls_total.labels(
                agent=self._agent_name,
                provider=provider,
                model=model,
                outcome=Outcome.ERROR.value,
                finish_reason="unknown",
                error_type=error_type,
            ).inc()
            raise

        self._metrics.llm_call_duration_seconds.labels(
            agent=self._agent_name, provider=provider, model=model, outcome=Outcome.SUCCESS.value, error_type=""
        ).observe(perf_counter() - start_time)

        self._metrics.llm_calls_total.labels(
            agent=self._agent_name,
            provider=provider,
            model=model,
            outcome=Outcome.SUCCESS.value,
            finish_reason=self._normalize_label(response.finish_reason),
            error_type="",
        ).inc()

        usage = response.usage
        if usage.prompt_tokens:
            self._metrics.llm_tokens_total.labels(
                agent=self._agent_name,
                provider=provider,
                model=model,
                token_type="input",
            ).inc(usage.prompt_tokens)

        if usage.completion_tokens:
            self._metrics.llm_tokens_total.labels(
                agent=self._agent_name,
                provider=provider,
                model=model,
                token_type="output",
            ).inc(usage.completion_tokens)

        if usage.cache_read_input_tokens:
            self._metrics.llm_tokens_total.labels(
                agent=self._agent_name,
                provider=provider,
                model=model,
                token_type="cache_read_input",
            ).inc(usage.cache_read_input_tokens)

        if usage.cache_creation_input_tokens:
            self._metrics.llm_tokens_total.labels(
                agent=self._agent_name,
                provider=provider,
                model=model,
                token_type="cache_creation_input",
            ).inc(usage.cache_creation_input_tokens)

        if usage.thinking_tokens:
            self._metrics.llm_tokens_total.labels(
                agent=self._agent_name,
                provider=provider,
                model=model,
                token_type="thinking",
            ).inc(usage.thinking_tokens)

        return response

    async def on_tool_execution(
        self,
        call_next: "ToolExecution",
        event: "ToolCallEvent",
        context: "Context",
    ) -> "ToolResultType":
        tool_name = self._normalize_label(event.name)
        start_time = perf_counter()

        try:
            result = await call_next(event, context)
        except BaseException as exc:
            error_type = self._get_error_type(exc)
            self._metrics.tool_duration_seconds.labels(
                agent=self._agent_name, tool=tool_name, outcome=Outcome.ERROR.value, error_type=error_type
            ).observe(perf_counter() - start_time)
            self._metrics.tool_calls_total.labels(
                agent=self._agent_name, tool=tool_name, outcome=Outcome.ERROR.value, error_type=error_type
            ).inc()
            raise

        outcome = Outcome.ERROR if isinstance(result, ToolErrorEvent) else Outcome.SUCCESS
        error_type = self._get_error_type(result.error) if isinstance(result, ToolErrorEvent) else ""

        self._metrics.tool_duration_seconds.labels(
            agent=self._agent_name, tool=tool_name, outcome=outcome.value, error_type=error_type
        ).observe(perf_counter() - start_time)
        self._metrics.tool_calls_total.labels(
            agent=self._agent_name, tool=tool_name, outcome=outcome.value, error_type=error_type
        ).inc()

        return result

    async def on_human_input(
        self,
        call_next: "HumanInputHook",
        event: "HumanInputRequest",
        context: "Context",
    ) -> "HumanMessage":
        start_time = perf_counter()

        try:
            response = await call_next(event, context)
        except BaseException as exc:
            error_type = self._get_error_type(exc)
            self._metrics.human_input_duration_seconds.labels(
                agent=self._agent_name, outcome=Outcome.ERROR.value, error_type=error_type
            ).observe(perf_counter() - start_time)
            self._metrics.human_input_requests_total.labels(
                agent=self._agent_name, outcome=Outcome.ERROR.value, error_type=error_type
            ).inc()
            raise

        self._metrics.human_input_duration_seconds.labels(
            agent=self._agent_name, outcome=Outcome.SUCCESS.value, error_type=""
        ).observe(perf_counter() - start_time)
        self._metrics.human_input_requests_total.labels(
            agent=self._agent_name, outcome=Outcome.SUCCESS.value, error_type=""
        ).inc()

        return response

    def _get_agent_name(self, context: "Context") -> str | None:
        agent: Agent | None = context.dependencies.get(AGENT_CONTEXT_DEPENDENCY_KEY)
        if agent is not None:
            return agent.name
        return None

    def _get_model_labels(self, context: "Context") -> tuple[str, str]:
        model_config = typing.cast("ModelConfig | None", context.dependencies.get(MODEL_CONFIG_CONTEXT_DEPENDENCY_KEY))
        if model_config is None:
            return "unknown", "unknown"

        return (
            self._normalize_provider_label(self._get_model_provider(model_config)),
            self._normalize_label(self._get_model_name(model_config)),
        )

    def _get_model_provider(self, model_config: "ModelConfig") -> Any:
        try:
            return model_config.provider
        except NotImplementedError:
            return None

    def _get_model_name(self, model_config: "ModelConfig") -> Any:
        try:
            return model_config.model
        except NotImplementedError:
            return None

    def _normalize_provider_label(self, provider: Any) -> str:
        if isinstance(provider, Enum):
            return self._normalize_label(provider.value)
        return self._normalize_label(provider)

    def _normalize_label(self, value: Any) -> str:
        if not value:
            return "unknown"
        return str(value)

    def _get_error_type(self, exception: BaseException) -> str:
        return type(exception).__name__
