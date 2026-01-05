# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from a2a.utils.telemetry import SpanKind
from opentelemetry.trace import Tracer

from autogen.doc_utils import export_module
from autogen.oai import client as oai_client_module
from autogen.oai.client import OpenAIWrapper
from autogen.opentelemetry.consts import SpanType
from autogen.opentelemetry.utils import (
    get_model_from_config_list,
    get_provider_from_config_list,
    message_to_otel,
    messages_to_otel,
    set_llm_request_params,
)


@export_module("autogen.opentelemetry")
def instrument_llm_wrapper(tracer: Tracer, capture_messages: bool = False) -> None:
    """Instrument OpenAIWrapper.create() to emit LLM spans.

    This creates 'chat' spans for each LLM API call, capturing:
    - Provider name (openai, anthropic, etc.)
    - Model name (gpt-4, claude-3, etc.)
    - Token usage (input/output)
    - Response metadata (finish reasons, cost)

    LLM spans automatically become children of agent invoke spans via
    OpenTelemetry's context propagation.

    Args:
        tracer: The OpenTelemetry tracer
        capture_messages: If True, capture input/output messages in span attributes.
            Default is False since messages may contain sensitive data.

    Usage:
        from autogen.instrumentation import setup_instrumentation, instrument_llm_wrapper

        tracer = setup_instrumentation("my-service")
        instrument_llm_wrapper(tracer)

        # Or with message capture enabled (for debugging)
        instrument_llm_wrapper(tracer, capture_messages=True)
    """
    original_create = OpenAIWrapper.create

    def traced_create(self: OpenAIWrapper, **config: Any) -> Any:
        # Get model from config or wrapper's config_list
        model = config.get("model") or get_model_from_config_list(self._config_list)
        span_name = f"chat {model}" if model else "chat"

        with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
            # Required attributes
            span.set_attribute("ag2.span.type", SpanType.LLM.value)
            span.set_attribute("gen_ai.operation.name", "chat")

            # Provider and model
            provider = get_provider_from_config_list(self._config_list)
            if provider:
                span.set_attribute("gen_ai.provider.name", provider)
            if model:
                span.set_attribute("gen_ai.request.model", model)

            # Agent name (from extra_kwargs passed by ConversableAgent)
            agent = config.get("agent")
            if agent and hasattr(agent, "name"):
                span.set_attribute("gen_ai.agent.name", agent.name)

            # Request parameters
            set_llm_request_params(span, config)

            # Input messages (opt-in)
            if capture_messages and "messages" in config:
                otel_msgs = messages_to_otel(config["messages"])
                span.set_attribute("gen_ai.input.messages", json.dumps(otel_msgs))

            try:
                response = original_create(self, **config)
            except Exception as e:
                span.set_attribute("error.type", type(e).__name__)
                raise

            # Response attributes
            _set_llm_response_attributes(span, response, capture_messages)

            return response

    # Apply the patch to OpenAIWrapper.create
    OpenAIWrapper.create = traced_create
    oai_client_module.OpenAIWrapper.create = traced_create


def _set_llm_response_attributes(span: Any, response: Any, capture_messages: bool = False) -> None:
    # Response model (may differ from request)
    if hasattr(response, "model") and response.model:
        span.set_attribute("gen_ai.response.model", response.model)

    # Token usage
    if hasattr(response, "usage") and response.usage:
        if hasattr(response.usage, "prompt_tokens") and response.usage.prompt_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
        if hasattr(response.usage, "completion_tokens"):
            span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens or 0)

    # Finish reasons
    if hasattr(response, "choices") and response.choices:
        reasons = [str(c.finish_reason) for c in response.choices if hasattr(c, "finish_reason") and c.finish_reason]
        if reasons:
            span.set_attribute("gen_ai.response.finish_reasons", json.dumps(reasons))

    # Cost (AG2-specific)
    if hasattr(response, "cost") and response.cost is not None:
        span.set_attribute("gen_ai.usage.cost", response.cost)

    # Output messages (opt-in)
    if capture_messages and hasattr(response, "choices") and response.choices:
        output_msgs = []
        for choice in response.choices:
            if hasattr(choice, "message") and choice.message:
                msg: dict[str, Any] = {"role": "assistant", "content": getattr(choice.message, "content", "") or ""}
                if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in choice.message.tool_calls
                    ]
                output_msgs.append(message_to_otel(msg))
        if output_msgs:
            span.set_attribute("gen_ai.output.messages", json.dumps(output_msgs))
