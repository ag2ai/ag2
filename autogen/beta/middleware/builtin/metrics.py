# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from prometheus_client import CollectorRegistry, Counter

from autogen.beta.annotations import Context
from autogen.beta.events import (
    BaseEvent,
    ModelResponse,
)
from autogen.beta.middleware import BaseMiddleware
from autogen.beta.middleware.base import LLMCall, MiddlewareFactory


class MetricsMiddleware(MiddlewareFactory):
    def __init__(self, registry: CollectorRegistry) -> None:
        self._registry = registry
        self._llm_calls_total_metric = Counter(
            name="ag2_llm_calls_total",
            documentation="",
            labelnames=[],
            registry=registry,
        )

    def __call__(self, event: "BaseEvent", context: "Context") -> BaseMiddleware:
        return _MetricsMiddleware(event, context, self._llm_calls_total_metric)


class _MetricsMiddleware(BaseMiddleware):
    def __init__(self, event: "BaseEvent", context: Context, llm_calls_total_metric: Counter) -> None:
        super().__init__(event, context)
        self._llm_calls_total_metric = llm_calls_total_metric

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: "Sequence[BaseEvent]",
        context: "Context",
    ) -> "ModelResponse":
        self._llm_calls_total_metric.inc()
        return await call_next(events, context)
