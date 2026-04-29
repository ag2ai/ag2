# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
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
    UrlInput,
)
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
            blocks: list[dict[str, str]] = []
            has_non_text = False
            for inp in message.parts:
                if isinstance(inp, TextInput):
                    blocks.append({"text": inp.content})
                elif isinstance(inp, DataInput):
                    blocks.append({"text": serializer.encode(inp.data).decode()})
                elif isinstance(inp, BinaryInput) and inp.kind is BinaryType.IMAGE:
                    b64 = base64.b64encode(inp.data).decode()
                    blocks.append({"image": f"data:{inp.media_type};base64,{b64}"})
                    has_non_text = True
                elif isinstance(inp, UrlInput) and inp.kind is BinaryType.IMAGE:
                    blocks.append({"image": inp.url})
                    has_non_text = True
                else:
                    raise UnsupportedInputError(type(inp).__name__, "dashscope")

            if not blocks:
                continue
            if not has_non_text and len(blocks) == 1:
                result.append({"role": "user", "content": blocks[0]["text"]})
            else:
                result.append({"role": "user", "content": blocks})

        elif isinstance(message, ModelResponse):
            result.append(message.to_api())

        elif isinstance(message, ToolResultsEvent):
            for r in message.results:
                blocks: list[dict[str, str]] = []
                has_non_text = False
                for part in r.result.parts:
                    if isinstance(part, TextInput):
                        blocks.append({"text": part.content})
                    elif isinstance(part, DataInput):
                        blocks.append({"text": serializer.encode(part.data).decode()})
                    elif isinstance(part, BinaryInput) and part.kind is BinaryType.IMAGE:
                        b64 = base64.b64encode(part.data).decode()
                        blocks.append({"image": f"data:{part.media_type};base64,{b64}"})
                        has_non_text = True
                    elif isinstance(part, UrlInput) and part.kind is BinaryType.IMAGE:
                        blocks.append({"image": part.url})
                        has_non_text = True
                    else:
                        raise UnsupportedInputError(type(part).__name__, "dashscope")

                if not has_non_text and len(blocks) == 1:
                    content: str | list[dict[str, str]] = blocks[0]["text"]
                else:
                    content = blocks
                result.append({"role": "tool", "tool_call_id": r.parent_id, "content": content})

    return result
