# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Any

from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, ToolResultsEvent
from autogen.beta.exceptions import UnsupportedToolError
from autogen.beta.response import ResponseProto
from autogen.beta.tools.final import FunctionToolSchema
from autogen.beta.tools.schemas import ToolSchema


def response_proto_to_format(response: ResponseProto | None) -> dict[str, Any] | None:
    """Convert a ResponseProto to DashScope response_format (OpenAI-compatible)."""
    if not response or not response.json_schema:
        return None

    schema: dict[str, Any] = {
        "schema": response.json_schema,
        "name": response.name,
    }
    if response.description:
        schema["description"] = response.description

    return {"type": "json_schema", "json_schema": schema}


def tool_to_api(t: ToolSchema) -> dict[str, Any]:
    if isinstance(t, FunctionToolSchema):
        return {
            "type": "function",
            "function": {
                "name": t.function.name,
                "description": t.function.description,
                "parameters": t.function.parameters,
            },
        }

    raise UnsupportedToolError(t.type, "dashscope")


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
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
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
                    "tool_call_id": r.parent_id,
                    "content": r.content,
                })

    return result
