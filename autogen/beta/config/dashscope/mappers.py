# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Any

from fast_depends.library.serializer import SerializerProto

from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, TextInput, ToolResultsEvent
from autogen.beta.events.input_events import DataInput
from autogen.beta.exceptions import UnsupportedInputError, UnsupportedToolError
from autogen.beta.response import ResponseProto
from autogen.beta.tools.builtin.skills import SkillsToolSchema
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

    elif isinstance(t, SkillsToolSchema):
        raise UnsupportedToolError(t.type, "dashscope")

    raise UnsupportedToolError(t.type, "dashscope")


def convert_messages(
    system_prompt: Iterable[str],
    messages: Iterable[BaseEvent],
    serializer: SerializerProto,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = [{"content": p, "role": "system"} for p in system_prompt]

    for message in messages:
        if isinstance(message, ModelRequest):
            for inp in message.parts:
                if isinstance(inp, TextInput):
                    result.append(inp.to_api())
                elif isinstance(inp, DataInput):
                    result.append({"role": "user", "content": serializer.encode(inp.data).decode()})
                else:
                    raise UnsupportedInputError(type(inp).__name__, "dashscope")

        elif isinstance(message, ModelResponse):
            result.append(message.to_api())

        elif isinstance(message, ToolResultsEvent):
            for r in message.results:
                parts: list[dict[str, Any]] = []
                for part in r.result.parts:
                    if isinstance(part, TextInput):
                        parts.append({"type": "text", "text": part.content})
                    elif isinstance(part, DataInput):
                        parts.append({"type": "text", "text": serializer.encode(part.data).decode()})
                    else:
                        raise UnsupportedInputError(type(part).__name__, "dashscope")
                content = parts[0]["text"] if len(parts) == 1 and parts[0]["type"] == "text" else parts
                result.append({"role": "tool", "tool_call_id": r.parent_id, "content": content})

    return result
