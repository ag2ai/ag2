# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from google.genai import types

from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, ToolResults
from autogen.beta.tools import Tool


def tool_to_api(t: Tool) -> dict[str, Any]:
    return {
        "name": t.schema.function.name,
        "description": t.schema.function.description,
        "parameters": t.schema.function.parameters,
    }


def convert_messages(
    messages: tuple[BaseEvent, ...],
) -> list[types.Content]:
    result: list[types.Content] = []

    for message in messages:
        if isinstance(message, ModelRequest):
            result.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=message.content)],
                )
            )
        elif isinstance(message, ModelResponse):
            parts: list[types.Part] = []
            if message.message:
                parts.append(types.Part.from_text(text=message.message.content))
            for call in message.tool_calls.calls:
                fc_part = types.Part.from_function_call(
                    name=call.name,
                    args=json.loads(call.arguments),
                )
                if "thought_signature" in call.provider_data:
                    fc_part.thought_signature = call.provider_data["thought_signature"]
                parts.append(fc_part)
            if parts:
                result.append(types.Content(role="model", parts=parts))
        elif isinstance(message, ToolResults):
            parts_list: list[types.Part] = []
            for r in message.results:
                parts_list.append(
                    types.Part.from_function_response(
                        name=r.name if hasattr(r, "name") else "",
                        response={"result": r.content},
                    )
                )
            result.append(types.Content(role="user", parts=parts_list))

    return result
