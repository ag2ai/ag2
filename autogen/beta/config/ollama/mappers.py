# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable
from typing import Any

from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, ToolResultsEvent
from autogen.beta.exceptions import UnsupportedToolError
from autogen.beta.tools.final import FunctionToolSchema
from autogen.beta.tools.schemas import ToolSchema


def _ensure_object_schema(params: dict[str, Any]) -> dict[str, Any]:
    """Ollama SDK requires tool parameters to be type: object."""
    schema = dict(params)
    schema["type"] = "object"
    schema.setdefault("properties", {})
    return schema


def tool_to_api(t: ToolSchema) -> dict[str, Any]:
    if isinstance(t, FunctionToolSchema):
        return {
            "type": "function",
            "function": {
                "name": t.function.name,
                "description": t.function.description,
                "parameters": _ensure_object_schema(t.function.parameters),
            },
        }

    raise UnsupportedToolError(t.type, "ollama")


def convert_messages(
    system_prompt: Iterable[str],
    messages: Iterable[BaseEvent],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = [{"content": p, "role": "system"} for p in system_prompt]

    for message in messages:
        if isinstance(message, ModelRequest):
            result.append({"role": "user", "content": message.content})
        elif isinstance(message, ModelResponse):
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
            }
            tool_calls = [
                {
                    "function": {
                        "name": c.name,
                        "arguments": json.loads(c.arguments) if c.arguments else {},
                    },
                }
                for c in message.tool_calls.calls
            ]
            if tool_calls:
                msg["tool_calls"] = tool_calls
            result.append(msg)
        elif isinstance(message, ToolResultsEvent):
            for r in message.results:
                result.append({
                    "role": "tool",
                    "content": r.content,
                })

    return result
