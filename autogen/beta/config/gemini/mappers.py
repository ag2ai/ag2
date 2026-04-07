# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable
from typing import Any

from google.genai import types

from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, ToolResultsEvent
from autogen.beta.events.types import Usage
from autogen.beta.exceptions import UnsupportedToolError
from autogen.beta.response import ResponseProto
from autogen.beta.tools.builtin.code_execution import CodeExecutionToolSchema
from autogen.beta.tools.builtin.web_fetch import WebFetchToolSchema
from autogen.beta.tools.builtin.web_search import WebSearchToolSchema
from autogen.beta.tools.final import FunctionToolSchema
from autogen.beta.tools.schemas import ToolSchema


def response_proto_to_config(response: ResponseProto | None) -> dict[str, Any]:
    """Convert a ResponseProto to Gemini GenerateContentConfig kwargs."""
    if not response or not response.json_schema:
        return {}

    return {
        "response_mime_type": "application/json",
        "response_json_schema": response.json_schema,
    }


def build_system_instruction(
    system_prompt: Iterable[str],
) -> str | None:
    """Join system prompt parts into a single string for Gemini's system_instruction."""
    joined = "\n".join(system_prompt)
    return joined or None


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
            gs_kwargs: dict[str, Any] = {}
            if t.blocked_domains:
                gs_kwargs["exclude_domains"] = t.blocked_domains
            extra_tools.append(types.Tool(google_search=types.GoogleSearch(**gs_kwargs)))

        elif isinstance(t, WebFetchToolSchema):
            extra_tools.append(types.Tool(url_context=types.UrlContext()))

        elif isinstance(t, CodeExecutionToolSchema):
            extra_tools.append(types.Tool(code_execution=types.ToolCodeExecution()))

        else:
            raise UnsupportedToolError(t.type, "gemini")

    result: list[types.Tool] = []
    if function_declarations:
        result.append(types.Tool(function_declarations=function_declarations))
    result.extend(extra_tools)

    return result or None


def convert_messages(
    messages: Iterable[BaseEvent],
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
                    args=json.loads(call.arguments or "{}"),
                )
                if "thought_signature" in call.provider_data:
                    fc_part.thought_signature = call.provider_data["thought_signature"]
                parts.append(fc_part)
            if parts:
                result.append(types.Content(role="model", parts=parts))

        elif isinstance(message, ToolResultsEvent):
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


def normalize_usage(metadata: Any) -> Usage:
    """Build usage from Gemini UsageMetadata, normalizing to standard keys."""
    cache_read: float | None = None
    if metadata.cached_content_token_count:
        cache_read = float(metadata.cached_content_token_count)
    return Usage(
        prompt_tokens=float(metadata.prompt_token_count),
        completion_tokens=float(metadata.candidates_token_count),
        total_tokens=float(metadata.total_token_count),
        cache_read_input_tokens=cache_read,
    )
