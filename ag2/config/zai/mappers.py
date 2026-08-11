# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable
from typing import Any

from fast_depends.library.serializer import SerializerProto
from zai.types.chat.chat_completion import CompletionUsage

from ag2.compact import CompactionSummary
from ag2.events import (
    BaseEvent,
    DataInput,
    Input,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
    ToolResultsEvent,
    Usage,
)
from ag2.exceptions import UnsupportedInputError, UnsupportedToolError
from ag2.response import ResponseProto
from ag2.tools.builtin.retrieval import RetrievalToolSchema
from ag2.tools.builtin.web_search import WebSearchToolSchema
from ag2.tools.final import FunctionToolSchema
from ag2.tools.schemas import ToolSchema

PROVIDER = "zai"


_SCHEMA_INSTRUCTION = (
    "You must respond with valid JSON that conforms to the following JSON schema:\n"
    "```json\n{schema}\n```\n"
    "Respond with only the JSON object — no markdown code fences, no commentary."
)


def response_proto_to_format(response: ResponseProto | None) -> dict[str, Any] | None:
    """Map a response schema to Z.AI's ``response_format``.

    Z.AI only supports JSON mode via (``{"type": "json_object"}``); it does not
    support OpenAI-style ``{"type": "json_schema"}``, so the schema itself is
    conveyed to the model via injection into the system prompt (see ``schema_instruction``).
    """
    if not response or not response.json_schema:
        return None
    return {"type": "json_object"}


def schema_instruction(response: ResponseProto | None) -> str | None:
    """System-prompt text describing the JSON schema for Z.AI's JSON mode.

    Returns ``None`` when the schema already supplies its own prompt (e.g.
    ``PromptedSchema``) or when there is no native schema to describe.
    """
    if not response or response.system_prompt or not response.json_schema:
        return None
    schema = _ensure_additional_properties_false(response.json_schema)
    return _SCHEMA_INSTRUCTION.format(schema=json.dumps(schema, indent=2))


def _ensure_additional_properties_false(schema: dict[str, Any]) -> dict[str, Any]:
    schema = dict(schema)

    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)

    if "properties" in schema:
        schema["properties"] = {
            k: _ensure_additional_properties_false(v) if isinstance(v, dict) else v
            for k, v in schema["properties"].items()
        }

    if "$defs" in schema:
        schema["$defs"] = {
            k: _ensure_additional_properties_false(v) if isinstance(v, dict) else v for k, v in schema["$defs"].items()
        }

    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            schema[key] = [
                _ensure_additional_properties_false(item) if isinstance(item, dict) else item for item in schema[key]
            ]

    if "items" in schema and isinstance(schema["items"], dict):
        schema["items"] = _ensure_additional_properties_false(schema["items"])

    return schema


def _ensure_object_schema(params: dict[str, Any]) -> dict[str, Any]:
    schema = dict(params)
    schema["type"] = "object"
    schema.setdefault("properties", {})
    schema.setdefault("additionalProperties", False)
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

    if isinstance(t, WebSearchToolSchema):
        # Z.AI's web_search tool only understands `content_size`; max_uses,
        # user_location, allowed_domains and blocked_domains have no equivalent
        # and are silently dropped. `enable` is required: it defaults to false,
        # so the search never activates unless we set it explicitly.
        web_search: dict[str, Any] = {"enable": True, "search_engine": "search-prime"}
        if t.search_context_size is not None:
            web_search["content_size"] = t.search_context_size
        return {"type": "web_search", "web_search": web_search}

    if isinstance(t, RetrievalToolSchema):
        retrieval: dict[str, Any] = {"knowledge_id": t.knowledge_id}
        if t.prompt_template is not None:
            retrieval["prompt_template"] = t.prompt_template
        return {"type": "retrieval", "retrieval": retrieval}

    raise UnsupportedToolError(t.type, PROVIDER)


def _input_to_content(inp: Input, serializer: SerializerProto) -> str:
    if isinstance(inp, TextInput):
        return inp.content
    if isinstance(inp, DataInput):
        return serializer.encode(inp.data).decode()
    raise UnsupportedInputError(type(inp).__name__, PROVIDER)


def json_arguments(arguments: Any) -> str:
    if arguments is None or arguments == "":
        return "{}"
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments)


def tool_call_index(raw_index: Any, fallback: int) -> int:
    if isinstance(raw_index, int):
        return raw_index
    if isinstance(raw_index, str):
        try:
            return int(raw_index)
        except ValueError:
            return fallback
    return fallback


def tool_call_event(call_id: Any, name: Any, arguments: Any) -> ToolCallEvent | None:
    if not isinstance(call_id, str) or not call_id:
        return None
    if not isinstance(name, str) or not name:
        return None
    return ToolCallEvent(id=call_id, name=name, arguments=json_arguments(arguments))


def _result_content(parts: Iterable[Input], serializer: SerializerProto) -> str | list[dict[str, str]]:
    content = [{"type": "text", "text": _input_to_content(part, serializer)} for part in parts]
    if len(content) == 1:
        return content[0]["text"]
    return content


def convert_messages(
    prompt: Iterable[str],
    messages: Iterable[BaseEvent],
    serializer: SerializerProto,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    system_prompt = "\n".join(prompt)
    if system_prompt:
        result.append({"role": "system", "content": system_prompt})

    for message in messages:
        if isinstance(message, ModelResponse):
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": message.message.content if message.message else None,
            }
            if message.tool_calls.calls:
                assistant["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments or "{}"},
                    }
                    for call in message.tool_calls.calls
                ]
            result.append(assistant)

        elif isinstance(message, ToolResultsEvent):
            for r in message.results:
                result.append({
                    "role": "tool",
                    "tool_call_id": r.parent_id,
                    "content": _result_content(r.result.parts, serializer),
                })

        elif isinstance(message, ModelRequest):
            parts = [{"type": "text", "text": _input_to_content(inp, serializer)} for inp in message.parts]
            content: str | list[dict[str, str]] = parts[0]["text"] if len(parts) == 1 else parts
            result.append({"role": "user", "content": content})

        elif isinstance(message, CompactionSummary):
            result.append({"role": "user", "content": f"[Summary of earlier conversation]\n{message.summary}"})

        elif isinstance(message, (ToolResultEvent, ToolErrorEvent)):
            result.append({
                "role": "tool",
                "tool_call_id": message.parent_id,
                "content": _result_content(message.result.parts, serializer),
            })

    return result


def normalize_usage(raw: CompletionUsage | None) -> Usage:
    if raw is None:
        return Usage()

    prompt_value = float(raw.prompt_tokens) if raw.prompt_tokens is not None else 0
    completion_value = float(raw.completion_tokens) if raw.completion_tokens is not None else 0
    total_value = float(raw.total_tokens) if raw.total_tokens is not None else prompt_value + completion_value

    prompt_details = raw.prompt_tokens_details
    completion_details = raw.completion_tokens_details
    cache_read = prompt_details.cached_tokens if prompt_details else None
    thinking = completion_details.reasoning_tokens if completion_details else None

    return Usage(
        prompt_tokens=prompt_value,
        completion_tokens=completion_value,
        total_tokens=total_value,
        cache_read_input_tokens=float(cache_read) if cache_read is not None else None,
        thinking_tokens=float(thinking) if thinking is not None else None,
    )
