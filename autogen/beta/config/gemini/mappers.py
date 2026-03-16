# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json

from google.genai import types

from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, ToolResults
from autogen.beta.exceptions import UnsupportedToolError
from autogen.beta.tools.builtin.web_search import WebSearchToolSchema
from autogen.beta.tools.final import FunctionToolSchema
from autogen.beta.tools.schemas import ToolSchema


def build_tools(schemas: list[ToolSchema]) -> list[types.Tool] | None:
    """Build Gemini tool objects from a list of ToolSchemas."""
    function_declarations: list[types.FunctionDeclaration] = []
    extra_tools: list[types.Tool] = []

    for t in schemas:
        if isinstance(t, FunctionToolSchema):
            function_declarations.append(
                types.FunctionDeclaration(
                    name=t.function.name,
                    description=t.function.description,
                    parameters=t.function.parameters,
                )
            )

        elif isinstance(t, WebSearchToolSchema):
            extra_tools.append(types.Tool(google_search=types.GoogleSearch()))

        else:
            raise UnsupportedToolError(t.type, "gemini")

    result: list[types.Tool] = []
    if function_declarations:
        result.append(types.Tool(function_declarations=function_declarations))
    result.extend(extra_tools)

    return result or None


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
