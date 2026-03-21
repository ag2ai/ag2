# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Sequence

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, HumanInputRequest, HumanMessage, ModelResponse, ToolCallEvent
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
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import SpanKind, StatusCode
except ImportError as _err:
    raise ImportError(
        "OpenTelemetry packages are required for TelemetryMiddleware. Install them with: pip install ag2[tracing]"
    ) from _err

from autogen.beta.events import ToolErrorEvent

_SCHEMA_URL = "https://opentelemetry.io/schemas/1.11.0"
_INSTRUMENTING_MODULE = "opentelemetry.instrumentation.ag2.beta"


def _get_tracer(tracer_provider: TracerProvider | None = None) -> trace.Tracer:
    provider = tracer_provider or trace.get_tracer_provider()
    return provider.get_tracer(_INSTRUMENTING_MODULE, schema_url=_SCHEMA_URL)


class TelemetryMiddleware(MiddlewareFactory):
    """Middleware that emits OpenTelemetry spans for agent turns, LLM calls, tool executions, and human input.

    Follows the OpenTelemetry GenAI Semantic Conventions.

    Args:
        tracer_provider: Optional TracerProvider. Defaults to the global provider.
        capture_content: Whether to include message content, tool arguments/results in spans. Defaults to True.
        agent_name: Agent name for span attributes. If not set, defaults to "unknown".
        provider_name: LLM provider name (e.g. "openai", "anthropic").
        model_name: Model name (e.g. "gpt-4o-mini").
    """

    def __init__(
        self,
        *,
        tracer_provider: TracerProvider | None = None,
        capture_content: bool = True,
        agent_name: str | None = None,
        provider_name: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self._tracer = _get_tracer(tracer_provider)
        self._capture_content = capture_content
        self._agent_name = agent_name or "unknown"
        self._provider_name = provider_name
        self._model_name = model_name

    def __call__(self, event: BaseEvent, context: Context) -> BaseMiddleware:
        return _TelemetryMiddlewareInstance(
            event,
            context,
            tracer=self._tracer,
            capture_content=self._capture_content,
            agent_name=self._agent_name,
            provider_name=self._provider_name,
            model_name=self._model_name,
        )


class _TelemetryMiddlewareInstance(BaseMiddleware):
    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        *,
        tracer: trace.Tracer,
        capture_content: bool,
        agent_name: str,
        provider_name: str | None,
        model_name: str | None,
    ) -> None:
        super().__init__(event, context)
        self._tracer = tracer
        self._capture_content = capture_content
        self._agent_name = agent_name
        self._provider_name = provider_name
        self._model_name = model_name

    async def on_turn(
        self,
        call_next: AgentTurn,
        event: BaseEvent,
        context: Context,
    ) -> ModelResponse:
        with self._tracer.start_as_current_span(
            f"invoke_agent {self._agent_name}",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("ag2.span.type", "agent")
            span.set_attribute("gen_ai.operation.name", "invoke_agent")
            span.set_attribute("gen_ai.agent.name", self._agent_name)
            if self._provider_name:
                span.set_attribute("gen_ai.provider.name", self._provider_name)
            if self._model_name:
                span.set_attribute("gen_ai.request.model", self._model_name)

            try:
                response = await call_next(event, context)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                raise

            return response

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        span_name = f"chat {self._model_name}" if self._model_name else "chat"

        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
        ) as span:
            span.set_attribute("ag2.span.type", "llm")
            span.set_attribute("gen_ai.operation.name", "chat")
            if self._provider_name:
                span.set_attribute("gen_ai.provider.name", self._provider_name)
            if self._model_name:
                span.set_attribute("gen_ai.request.model", self._model_name)

            if self._capture_content:
                input_messages = json.dumps([e.to_api() for e in events if hasattr(e, "to_api")])
                span.set_attribute("gen_ai.input.messages", input_messages)

            try:
                response = await call_next(events, context)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                raise

            # Auto-detect provider/model from response, falling back to constructor params
            provider = response.provider or self._provider_name
            model = response.model or self._model_name

            if provider and not self._provider_name:
                span.set_attribute("gen_ai.provider.name", provider)
            if model and not self._model_name:
                span.set_attribute("gen_ai.request.model", model)
                span.update_name(f"chat {model}")
            if model:
                span.set_attribute("gen_ai.response.model", model)
            if response.finish_reason:
                span.set_attribute("gen_ai.response.finish_reasons", [response.finish_reason])

            usage = response.usage
            if usage.get("prompt_tokens"):
                span.set_attribute("gen_ai.usage.input_tokens", int(usage["prompt_tokens"]))
            if usage.get("completion_tokens"):
                span.set_attribute("gen_ai.usage.output_tokens", int(usage["completion_tokens"]))

            if self._capture_content and response.message:
                span.set_attribute("gen_ai.output.messages", json.dumps([response.to_api()]))

            return response

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        with self._tracer.start_as_current_span(
            f"execute_tool {event.name}",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("ag2.span.type", "tool")
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            span.set_attribute("gen_ai.tool.name", event.name)
            span.set_attribute("gen_ai.tool.call.id", event.id)
            span.set_attribute("gen_ai.tool.type", "function")

            if self._capture_content:
                span.set_attribute("gen_ai.tool.call.arguments", event.arguments)

            try:
                result = await call_next(event, context)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                raise

            if isinstance(result, ToolErrorEvent):
                span.record_exception(result.error)
                span.set_status(StatusCode.ERROR, str(result.error))
            elif self._capture_content and hasattr(result, "content"):
                span.set_attribute("gen_ai.tool.call.result", result.content)

            return result

    async def on_human_input(
        self,
        call_next: HumanInputHook,
        event: HumanInputRequest,
        context: Context,
    ) -> HumanMessage:
        with self._tracer.start_as_current_span(
            f"await_human_input {self._agent_name}",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("ag2.span.type", "human_input")
            span.set_attribute("gen_ai.operation.name", "await_human_input")
            span.set_attribute("gen_ai.agent.name", self._agent_name)

            if self._capture_content:
                span.set_attribute("ag2.human_input.prompt", event.content)

            try:
                response = await call_next(event, context)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                raise

            if self._capture_content:
                span.set_attribute("ag2.human_input.response", response.content)

            return response
