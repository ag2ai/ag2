# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import suppress
from typing import Any

from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

TRACE_PROPAGATOR = TraceContextTextMapPropagator()


def message_to_otel(message: dict[str, Any]) -> dict[str, Any]:
    """Convert an AG2/OpenAI message to OTEL GenAI semantic convention format.

    AG2 format:
        {"role": "user", "content": "Hello", "name": "user_proxy"}
        {"role": "assistant", "tool_calls": [{"id": "...", "function": {"name": "fn", "arguments": "{}"}}]}
        {"role": "tool", "tool_call_id": "...", "content": "result"}

    OTEL format:
        {"role": "user", "parts": [{"type": "text", "content": "Hello"}]}
        {"role": "assistant", "parts": [{"type": "tool_call", "id": "...", "name": "fn", "arguments": {...}}]}
        {"role": "tool", "parts": [{"type": "tool_call_response", "id": "...", "response": "result"}]}
    """
    role = message.get("role", "user")
    parts: list[dict[str, Any]] = []

    # Handle tool_calls (assistant requesting tool execution)
    if "tool_calls" in message and message["tool_calls"]:
        for tool_call in message["tool_calls"]:
            func = tool_call.get("function", {})
            arguments = func.get("arguments", "{}")
            # Parse arguments if it's a JSON string
            if isinstance(arguments, str):
                with suppress(json.JSONDecodeError):
                    arguments = json.loads(arguments)

            parts.append({
                "type": "tool_call",
                "id": tool_call.get("id", ""),
                "name": func.get("name", ""),
                "arguments": arguments,
            })

    # Handle tool response
    elif role == "tool" and "tool_call_id" in message:
        parts.append({
            "type": "tool_call_response",
            "id": message.get("tool_call_id", ""),
            "response": message.get("content", ""),
        })

    # Handle regular text content
    elif "content" in message and message["content"]:
        content = message["content"]
        if isinstance(content, str):
            parts.append({"type": "text", "content": content})
        elif isinstance(content, list):
            # Handle multimodal content (list of content parts)
            for item in content:
                if isinstance(item, str):
                    parts.append({"type": "text", "content": item})
                elif isinstance(item, dict):
                    parts.append(item)

    result: dict[str, Any] = {"role": role, "parts": parts}

    return result


def messages_to_otel(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert a list of AG2 messages to OTEL format."""
    return [message_to_otel(msg) for msg in messages]


def reply_to_otel_message(reply: str | dict[str, Any] | None) -> list[dict[str, Any]]:
    """Convert an agent reply to OTEL output messages format.

    The reply can be:
    - A string (simple text response)
    - A dict with content and/or tool_calls
    - None (no response)
    """
    if reply is None:
        return []

    if isinstance(reply, str):
        return [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": reply}],
                "finish_reason": "stop",
            }
        ]

    if isinstance(reply, dict):
        return [message_to_otel({"role": "assistant", **reply})]

    return []


def aggregate_usage(usage_by_model: dict[str, dict[str, Any]]) -> tuple[str, int, int] | None:
    """Aggregate token usage across multiple models.

    Args:
        usage_by_model: Dict mapping model names to their usage data
            (prompt_tokens, completion_tokens, etc.)

    Returns:
        Tuple of (model_name, input_tokens, output_tokens) or None if empty.
        For multiple models, model_name is comma-separated.
    """
    if not usage_by_model:
        return None

    models = list(usage_by_model.keys())
    model_str = models[0] if len(models) == 1 else ", ".join(models)
    input_tokens = sum(d.get("prompt_tokens", 0) for d in usage_by_model.values())
    output_tokens = sum(d.get("completion_tokens", 0) for d in usage_by_model.values())

    return model_str, input_tokens, output_tokens


# Mapping from AG2 api_type to OTEL gen_ai.provider.name
# See: https://opentelemetry.io/docs/specs/semconv/attributes-registry/gen-ai/
API_TYPE_TO_PROVIDER = {
    "openai": "openai",
    "azure": "azure.ai.openai",
    "anthropic": "anthropic",
    "bedrock": "aws.bedrock",
    "mistral": "mistral_ai",
    "groq": "groq",
    "cohere": "cohere",
    "deepseek": "deepseek",
    "together": "together",  # Not in OTEL spec, but common
    "ollama": "ollama",  # Not in OTEL spec, but common
    "cerebras": "cerebras",  # Not in OTEL spec, but common
}


def get_provider_name(agent: Any) -> str | None:
    """Extract the provider name from an agent's LLM config.

    Returns the OTEL-standard provider name based on the agent's api_type,
    or None if no LLM config is present.
    """
    if not hasattr(agent, "llm_config") or not agent.llm_config:
        return None

    config_list = getattr(agent.llm_config, "config_list", None)
    if not config_list:
        return None

    # Get api_type from first config entry
    first_config = config_list[0]
    api_type = getattr(first_config, "api_type", None) or first_config.get("api_type")
    if not api_type:
        return None

    return API_TYPE_TO_PROVIDER.get(api_type, api_type)


def get_model_name(agent: Any) -> str | None:
    """Extract the model name from an agent's LLM config.

    Returns the configured model name, or None if no LLM config is present.
    """
    if not hasattr(agent, "llm_config") or not agent.llm_config:
        return None

    config_list = getattr(agent.llm_config, "config_list", None)
    if not config_list:
        return None

    first_config = config_list[0]
    return getattr(first_config, "model", None) or first_config.get("model")


def get_provider_from_config_list(config_list: list[dict[str, Any]]) -> str | None:
    """Extract provider name from a wrapper's config list.

    Returns the OTEL-standard provider name based on the first config's api_type,
    or "openai" as the default if no api_type is specified.
    """
    if not config_list:
        return "openai"  # Default provider
    first = config_list[0]
    api_type = first.get("api_type") if isinstance(first, dict) else getattr(first, "api_type", None)
    if not api_type:
        return "openai"
    return API_TYPE_TO_PROVIDER.get(api_type, api_type)


def get_model_from_config_list(config_list: list[dict[str, Any]]) -> str | None:
    """Extract model name from a wrapper's config list.

    Returns the model name from the first config entry, or None if not found.
    """
    if not config_list:
        return None
    first = config_list[0]
    return first.get("model") if isinstance(first, dict) else getattr(first, "model", None)


def set_llm_request_params(span: Any, config: dict[str, Any]) -> None:
    """Set optional LLM request parameters on a span.

    Captures temperature, max_tokens, top_p, frequency_penalty, and presence_penalty
    if they are present in the config.
    """
    params = [
        ("temperature", "gen_ai.request.temperature"),
        ("max_tokens", "gen_ai.request.max_tokens"),
        ("top_p", "gen_ai.request.top_p"),
        ("frequency_penalty", "gen_ai.request.frequency_penalty"),
        ("presence_penalty", "gen_ai.request.presence_penalty"),
    ]
    for key, attr in params:
        if key in config and config[key] is not None:
            span.set_attribute(attr, config[key])
