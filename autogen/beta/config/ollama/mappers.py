# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from collections.abc import Iterable
from typing import Any

from fast_depends.library.serializer import SerializerProto

from autogen.beta.events import (
    BaseEvent,
    BinaryInput,
    BinaryType,
    DataInput,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolResultsEvent,
)
from autogen.beta.exceptions import UnsupportedInputError, UnsupportedToolError
from autogen.beta.response import ResponseProto
from autogen.beta.tools.builtin.skills import SkillsToolSchema
from autogen.beta.tools.final import FunctionToolSchema
from autogen.beta.tools.schemas import ToolSchema


def response_proto_to_format(response: ResponseProto | None) -> dict[str, Any] | str | None:
    """Convert a ResponseProto to Ollama's format parameter."""
    if not response or not response.json_schema:
        return None

    return response.json_schema


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

    elif isinstance(t, SkillsToolSchema):
        raise UnsupportedToolError(t.type, "ollama")

    raise UnsupportedToolError(t.type, "ollama")


def convert_messages(
    system_prompt: Iterable[str],
    messages: Iterable[BaseEvent],
    serializer: SerializerProto,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = [{"content": p, "role": "system"} for p in system_prompt]

    for message in messages:
        if isinstance(message, ModelRequest):
            text_parts: list[str] = []
            images: list[str] = []
            for inp in message.parts:
                if isinstance(inp, TextInput):
                    text_parts.append(inp.content)
                elif isinstance(inp, DataInput):
                    text_parts.append(serializer.encode(inp.data).decode())
                elif isinstance(inp, BinaryInput) and inp.kind is BinaryType.IMAGE:
                    images.append(base64.b64encode(inp.data).decode())
                else:
                    raise UnsupportedInputError(type(inp).__name__, "ollama")

            # Ollama API only accepts plain-string `content`, so we emit one
            # user message per TextInput instead of joining them. Images are
            # attached to the last text message to keep them in the same turn.
            for text in text_parts:
                result.append({"role": "user", "content": text})
            if images:
                if text_parts:
                    result[-1]["images"] = images
                else:
                    result.append({"role": "user", "content": "", "images": images})

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
                parts: list[dict[str, Any]] = []
                for part in r.result.parts:
                    if isinstance(part, TextInput):
                        parts.append({"type": "text", "text": part.content})
                    elif isinstance(part, DataInput):
                        parts.append({"type": "text", "text": serializer.encode(part.data).decode()})
                    else:
                        raise UnsupportedInputError(type(part).__name__, "ollama")
                content = parts[0]["text"] if len(parts) == 1 and parts[0]["type"] == "text" else parts
                result.append({"role": "tool", "content": content})

    return result
