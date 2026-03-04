# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, ToolResults
from autogen.beta.tools import Tool


def tool_to_api(t: Tool) -> dict[str, Any]:
    return {
        "name": t.schema.function.name,
        "description": t.schema.function.description,
        "input_schema": t.schema.function.parameters,
    }


def convert_messages(
    messages: tuple[BaseEvent, ...],
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
        elif isinstance(message, ToolResults):
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
