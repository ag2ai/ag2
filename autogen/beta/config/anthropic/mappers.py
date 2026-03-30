# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable
from typing import Any

from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, ToolResultsEvent
from autogen.beta.exceptions import UnsupportedToolError
from autogen.beta.response import ResponseProto
from autogen.beta.tools.builtin.code_execution import CodeExecutionToolSchema
from autogen.beta.tools.builtin.memory import MemoryToolSchema
from autogen.beta.tools.builtin.web_fetch import WebFetchToolSchema
from autogen.beta.tools.builtin.web_search import WebSearchToolSchema
from autogen.beta.tools.final import FunctionToolSchema
from autogen.beta.tools.schemas import ToolSchema


def _ensure_additional_properties_false(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively add additionalProperties: false to all object schemas.

    Anthropic requires this on every object node in output_config.format.schema.
    """
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


def response_proto_to_output_config(response: ResponseProto | None) -> dict[str, Any] | None:
    """Convert a ResponseProto to Anthropic output_config."""
    if not response or not response.json_schema:
        return None

    strict_schema = _ensure_additional_properties_false(response.json_schema)

    return {
        "format": {
            "type": "json_schema",
            "schema": strict_schema,
        },
    }


def _ensure_object_schema(params: dict[str, Any]) -> dict[str, Any]:
    """Anthropic requires input_schema to be type: object."""
    schema = dict(params)
    schema["type"] = "object"
    schema.setdefault("properties", {})
    return schema


def tool_to_api(
    t: ToolSchema,
    *,
    web_search_version: str = "web_search_20250305",
    web_fetch_version: str = "web_fetch_20250910",
) -> dict[str, Any]:
    if isinstance(t, FunctionToolSchema):
        return {
            "name": t.function.name,
            "description": t.function.description,
            "input_schema": _ensure_object_schema(t.function.parameters),
        }

    elif isinstance(t, WebSearchToolSchema):
        result: dict[str, Any] = {"type": web_search_version, "name": "web_search"}
        if t.max_uses is not None:
            result["max_uses"] = t.max_uses
        if t.user_location is not None:
            loc: dict[str, str] = {"type": "approximate"}
            if t.user_location.city is not None:
                loc["city"] = t.user_location.city
            if t.user_location.region is not None:
                loc["region"] = t.user_location.region
            if t.user_location.country is not None:
                loc["country"] = t.user_location.country
            if t.user_location.timezone is not None:
                loc["timezone"] = t.user_location.timezone
            result["user_location"] = loc
        if t.allowed_domains is not None:
            result["allowed_domains"] = t.allowed_domains
        if t.blocked_domains is not None:
            result["blocked_domains"] = t.blocked_domains
        return result

    elif isinstance(t, CodeExecutionToolSchema):
        # https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
        return {"type": "code_execution_20250825", "name": "code_execution"}

    elif isinstance(t, WebFetchToolSchema):
        result = {"type": web_fetch_version, "name": "web_fetch"}
        if t.max_uses is not None:
            result["max_uses"] = t.max_uses
        if t.allowed_domains is not None:
            result["allowed_domains"] = t.allowed_domains
        if t.blocked_domains is not None:
            result["blocked_domains"] = t.blocked_domains
        if t.citations is not None:
            result["citations"] = {"enabled": t.citations.enabled}
        if t.max_content_tokens is not None:
            result["max_content_tokens"] = t.max_content_tokens
        return result

    elif isinstance(t, MemoryToolSchema):
        # https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool
        return {"type": "memory_20250818", "name": "memory"}

    raise UnsupportedToolError(t.type, "anthropic")


def convert_messages(
    messages: Iterable[BaseEvent],
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
                    "input": json.loads(call.arguments or "{}"),
                })
            if content:
                result.append({"role": "assistant", "content": content})
        elif isinstance(message, ToolResultsEvent):
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


def normalize_usage(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize Anthropic's native usage keys to standard format."""
    usage: dict[str, Any] = {
        "prompt_tokens": raw.get("input_tokens", 0),
        "completion_tokens": raw.get("output_tokens", 0),
    }
    if raw.get("cache_creation_input_tokens"):
        usage["cache_creation_input_tokens"] = raw["cache_creation_input_tokens"]
    if raw.get("cache_read_input_tokens"):
        usage["cache_read_input_tokens"] = raw["cache_read_input_tokens"]
    return usage
