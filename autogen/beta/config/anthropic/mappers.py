# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable
from typing import Any

from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, ToolResultsEvent
from autogen.beta.exceptions import UnsupportedToolError
from autogen.beta.tools.builtin.code_execution import CodeExecutionToolSchema
from autogen.beta.tools.builtin.web_search import WebSearchToolSchema
from autogen.beta.tools.final import FunctionToolSchema
from autogen.beta.tools.schemas import ToolSchema


def _ensure_object_schema(params: dict[str, Any]) -> dict[str, Any]:
    """Anthropic requires input_schema to be type: object."""
    schema = dict(params)
    schema["type"] = "object"
    schema.setdefault("properties", {})
    return schema


def tool_to_api(t: ToolSchema) -> dict[str, Any]:
    if isinstance(t, FunctionToolSchema):
        return {
            "name": t.function.name,
            "description": t.function.description,
            "input_schema": _ensure_object_schema(t.function.parameters),
        }

    elif isinstance(t, WebSearchToolSchema):
        result: dict[str, Any] = {"type": "web_search_20250305", "name": "web_search"}
        if t.max_uses is not None:
            result["max_uses"] = t.max_uses
        if t.user_location is not None:
            loc: dict[str, str] = {"type": "approximate"}
            if t.user_location.city is not None:
                loc["city"] = t.user_location.city
            if t.user_location.region is not None:
                loc["region"] = t.user_location.region
            if t.user_location.country is not None:
                loc["country"] = t.user_location.country
            if t.user_location.timezone is not None:
                loc["timezone"] = t.user_location.timezone
            result["user_location"] = loc
        return result

    elif isinstance(t, CodeExecutionToolSchema):
        # https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
        return {"type": "code_execution_20250825", "name": "code_execution"}

    raise UnsupportedToolError(t.type, "anthropic")


def convert_messages(
    messages: Iterable[BaseEvent],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for message in messages:
        if isinstance(message, ModelRequest):
            result.append({
                "role": "user",
                "content": message.content,
            })
        elif isinstance(message, ModelResponse):
            content: list[dict[str, Any]] = []
            if message.message:
                content.append({"type": "text", "text": message.message.content})
            for call in message.tool_calls.calls:
                content.append({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": json.loads(call.arguments),
                })
            if content:
                result.append({"role": "assistant", "content": content})
        elif isinstance(message, ToolResultsEvent):
            tool_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": r.parent_id,
                    "content": r.content,
                }
                for r in message.results
            ]
            result.append({"role": "user", "content": tool_results})

    return result
