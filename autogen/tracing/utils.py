# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any


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
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass  # Keep as string if not valid JSON

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
