# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""PrometheusMiddleware — Prometheus metrics for AG2 Beta agent turns, LLM calls, tool executions."""

import time
import weakref
from collections.abc import Sequence
from typing import ClassVar

from autogen.beta.annotations import Context
from autogen.beta.events import (
    BaseEvent,
    HumanInputRequest,
    HumanMessage,
    ModelResponse,
    ToolCallEvent,
    ToolErrorEvent,
)
from autogen.beta.middleware.base import (
    AgentTurn,
    BaseMiddleware,
    HumanInputHook,
    LLMCall,
    MiddlewareFactory,
    ToolExecution,
    ToolResultType,
)

try:
    from prometheus_client import REGISTRY as _DEFAULT_REGISTRY
    from prometheus_client import CollectorRegistry, Counter, Histogram
except ImportError as _err:
    raise ImportError(
        "prometheus-client is required for PrometheusMiddleware. Install it with: pip install ag2[metrics]"
    ) from _err

__all__ = ("PrometheusMiddleware",)

_DEFAULT_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


class _MetricsBundle:
    """All Prometheus metric objects for one registry."""

    __slots__ = (
        "human_input",
        "llm_call_duration",
        "llm_tokens",
        "tool_calls",
        "tool_duration",
        "turn_duration",
        "turn_errors",
    )

    def __init__(self, registry: "CollectorRegistry") -> None:
        self.turn_duration = Histogram(
            "ag2_turn_duration_seconds",
            "Wall-clock duration of one agent turn",
            ["agent_name"],
            buckets=_DEFAULT_DURATION_BUCKETS,
            registry=registry,
        )
        self.turn_errors = Counter(
            "ag2_turn_errors_total",
            "Number of agent turns that raised an exception",
            ["agent_name"],
            registry=registry,
        )
        self.llm_call_duration = Histogram(
            "ag2_llm_call_duration_seconds",
            "Wall-clock duration of one LLM API call",
            ["agent_name", "provider", "model"],
            buckets=_DEFAULT_DURATION_BUCKETS,
            registry=registry,
        )
        self.llm_tokens = Counter(
            "ag2_llm_tokens_total",
            "Tokens consumed in LLM calls",
            ["agent_name", "provider", "model", "token_type"],
            registry=registry,
        )
        self.tool_calls = Counter(
            "ag2_tool_calls_total",
            "Number of tool invocations",
            ["agent_name", "tool_name", "status"],
            registry=registry,
        )
        self.tool_duration = Histogram(
            "ag2_tool_execution_duration_seconds",
            "Wall-clock duration of one tool execution",
            ["agent_name", "tool_name"],
            buckets=_DEFAULT_DURATION_BUCKETS,
            registry=registry,
        )
        self.human_input = Counter(
            "ag2_human_input_requests_total",
            "Number of human-in-the-loop input requests",
            ["agent_name"],
            registry=registry,
        )


class PrometheusMiddleware(MiddlewareFactory):
    """Middleware that records Prometheus metrics for agent turns, LLM calls, tool executions, and human input.

    Metrics emitted:

    - ``ag2_turn_duration_seconds`` — histogram of full agent turn latency
    - ``ag2_turn_errors_total`` — counter of turns that raised an exception
    - ``ag2_llm_call_duration_seconds`` — histogram of LLM API call latency
    - ``ag2_llm_tokens_total`` — counter of tokens, by ``token_type`` (``input`` / ``output`` / ``cache_creation`` / ``cache_read``)
    - ``ag2_tool_calls_total`` — counter of tool invocations, by ``status`` (``success`` / ``error``)
    - ``ag2_tool_execution_duration_seconds`` — histogram of tool execution latency
    - ``ag2_human_input_requests_total`` — counter of human-in-the-loop requests

    All metrics carry an ``agent_name`` label.  LLM metrics also carry ``provider`` and ``model`` labels.
    Tool metrics carry a ``tool_name`` label.

    Args:
        agent_name: Value for the ``agent_name`` label on all metrics.
        registry: Prometheus ``CollectorRegistry``.  Defaults to the global default registry.
            Pass a fresh ``CollectorRegistry()`` in tests to avoid registration conflicts.

    Example::

        from prometheus_client import start_http_server
        from autogen.beta.middleware.builtin import PrometheusMiddleware

        start_http_server(8000)  # exposes /metrics on port 8000

        agent = Agent(
            "assistant",
            ...,
            middleware=[PrometheusMiddleware(agent_name="assistant")],
        )
    """

    _bundles: ClassVar[weakref.WeakKeyDictionary["CollectorRegistry", _MetricsBundle]] = weakref.WeakKeyDictionary()

    def __init__(
        self,
        *,
        agent_name: str = "unknown",
        registry: "CollectorRegistry | None" = None,
    ) -> None:
        resolved_registry = registry if registry is not None else _DEFAULT_REGISTRY
        if resolved_registry not in PrometheusMiddleware._bundles:
            PrometheusMiddleware._bundles[resolved_registry] = _MetricsBundle(resolved_registry)
        self._bundle = PrometheusMiddleware._bundles[resolved_registry]
        self._agent_name = agent_name

    def __call__(self, event: BaseEvent, context: Context) -> BaseMiddleware:
        return _PrometheusMiddlewareInstance(
            event,
            context,
            bundle=self._bundle,
            agent_name=self._agent_name,
        )


class _PrometheusMiddlewareInstance(BaseMiddleware):
    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        *,
        bundle: _MetricsBundle,
        agent_name: str,
    ) -> None:
        super().__init__(event, context)
        self._bundle = bundle
        self._agent_name = agent_name

    async def on_turn(
        self,
        call_next: AgentTurn,
        event: BaseEvent,
        context: Context,
    ) -> ModelResponse:
        start = time.perf_counter()
        try:
            return await call_next(event, context)
        except Exception:
            self._bundle.turn_errors.labels(agent_name=self._agent_name).inc()
            raise
        finally:
            self._bundle.turn_duration.labels(agent_name=self._agent_name).observe(time.perf_counter() - start)

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        start = time.perf_counter()
        response = await call_next(events, context)
        duration = time.perf_counter() - start

        provider = response.provider or ""
        model = response.model or ""

        self._bundle.llm_call_duration.labels(
            agent_name=self._agent_name,
            provider=provider,
            model=model,
        ).observe(duration)

        usage = response.usage
        token_map = {
            "input": int(usage.prompt_tokens or 0),
            "output": int(usage.completion_tokens or 0),
            "cache_creation": int(usage.cache_creation_input_tokens or 0),
            "cache_read": int(usage.cache_read_input_tokens or 0),
        }
        for token_type, count in token_map.items():
            if count:
                self._bundle.llm_tokens.labels(
                    agent_name=self._agent_name,
                    provider=provider,
                    model=model,
                    token_type=token_type,
                ).inc(count)

        return response

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        start = time.perf_counter()
        result = await call_next(event, context)
        duration = time.perf_counter() - start

        status = "error" if isinstance(result, ToolErrorEvent) else "success"
        self._bundle.tool_calls.labels(
            agent_name=self._agent_name,
            tool_name=event.name,
            status=status,
        ).inc()
        self._bundle.tool_duration.labels(
            agent_name=self._agent_name,
            tool_name=event.name,
        ).observe(duration)

        return result

    async def on_human_input(
        self,
        call_next: HumanInputHook,
        event: HumanInputRequest,
        context: Context,
    ) -> HumanMessage:
        self._bundle.human_input.labels(agent_name=self._agent_name).inc()
        return await call_next(event, context)
