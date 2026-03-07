# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable
from typing import Any

from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, ToolResults
from autogen.beta.tools import Tool


def tool_to_api(t: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": t.schema.function.name,
            "description": t.schema.function.description,
            "parameters": t.schema.function.parameters,
        },
    }


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
                "content": message.message.content if message.message else "",
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
        elif isinstance(message, ToolResults):
            for r in message.results:
                result.append({
                    "role": "tool",
                    "content": r.content,
                })

    return result
