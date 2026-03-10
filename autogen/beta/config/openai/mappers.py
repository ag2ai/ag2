# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Any

from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, ToolResults
from autogen.beta.tools import Tool


def convert_messages(
    system_prompt: Iterable[str],
    messages: tuple[BaseEvent, ...],
) -> list[dict[str, str]]:
    # legacy prompt message format
    result: list[dict[str, str]] = [{"content": "\n".join(system_prompt), "role": "system"}]

    for message in messages:
        if isinstance(message, (ModelRequest, ModelResponse)):
            result.append(message.to_api())
        elif isinstance(message, ToolResults):
            for r in message.results:
                result.append(r.to_api())

    return result


def _ensure_object_schema(params: dict[str, Any]) -> dict[str, Any]:
    """OpenAI requires tool parameters to be type: object with properties."""
    schema = dict(params)
    schema["type"] = "object"
    schema.setdefault("properties", {})
    schema.setdefault("additionalProperties", False)
    return schema


def tool_to_api(t: Tool) -> dict[str, Any]:
    """Chat Completions API tool format."""
    return {
        "type": "function",
        "function": {
            "name": t.schema.function.name,
            "description": t.schema.function.description,
            "parameters": _ensure_object_schema(t.schema.function.parameters),
        },
    }


def tool_to_responses_api(t: Tool) -> dict[str, Any]:
    """Responses API tool format — name/description at top level."""
    return {
        "type": "function",
        "name": t.schema.function.name,
        "description": t.schema.function.description,
        "parameters": _ensure_object_schema(t.schema.function.parameters),
    }
