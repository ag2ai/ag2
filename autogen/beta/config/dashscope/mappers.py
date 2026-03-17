# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Any

from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, ToolResults
from autogen.beta.exceptions import UnsupportedToolError
from autogen.beta.tools.schemas import ToolSchema


def tool_to_api(t: ToolSchema) -> dict[str, Any]:
    if t.type == "function":
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
    messages: tuple[BaseEvent, ...],
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
        elif isinstance(message, ToolResults):
            for r in message.results:
                result.append({
                    "role": "tool",
                    "tool_call_id": r.parent_id,
                    "content": r.content,
                })

    return result
