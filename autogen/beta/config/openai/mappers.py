# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
from collections.abc import Iterable, Sequence
from typing import Any

from openai.types import CompletionUsage
from openai.types.responses import ResponseUsage

from autogen.beta.events import (
    BaseEvent,
    BinaryInput,
    BinaryType,
    DocumentUrlInput,
    FileIdInput,
    ImageUrlInput,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolResultsEvent,
)
from autogen.beta.events.types import Usage
from autogen.beta.exceptions import UnsupportedInputError, UnsupportedToolError
from autogen.beta.response import ResponseProto
from autogen.beta.tools.builtin.code_execution import CodeExecutionToolSchema
from autogen.beta.tools.builtin.image_generation import ImageGenerationToolSchema
from autogen.beta.tools.builtin.mcp_server import MCPServerToolSchema
from autogen.beta.tools.builtin.shell import (
    ContainerAutoEnvironment,
    ContainerReferenceEnvironment,
    ShellToolSchema,
)
from autogen.beta.tools.builtin.web_search import WebSearchToolSchema
from autogen.beta.tools.final import FunctionToolSchema
from autogen.beta.tools.schemas import ToolSchema


def response_proto_to_schema(response: ResponseProto | None) -> dict[str, Any] | None:
    """Convert a ResponseProto to Chat Completions response_format."""
    if not response or not response.json_schema:
        return

    strict_schema = _ensure_additional_properties_false(response.json_schema)
    schema: dict[str, Any] = {
        "schema": strict_schema,
        "name": response.name,
        "strict": True,
    }
    if response.description:
        schema["description"] = response.description

    return {"type": "json_schema", "json_schema": schema}


def _ensure_additional_properties_false(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively add additionalProperties: false to all object schemas.

    The OpenAI Responses API requires this on every object node.
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


def response_proto_to_text_config(
    response: ResponseProto | None,
) -> dict[str, Any] | None:
    """Convert a ResponseProto to Responses API text config."""
    if not response or not response.json_schema:
        return

    strict_schema = _ensure_additional_properties_false(response.json_schema)

    fmt: dict[str, Any] = {
        "type": "json_schema",
        "name": response.name,
        "schema": strict_schema,
        "strict": True,
    }
    if response.description:
        fmt["description"] = response.description

    return {"format": fmt}


def events_to_responses_input(messages: Sequence[BaseEvent]) -> list[dict[str, Any]]:
    """Convert a sequence of events to Responses API input items."""
    result: list[dict[str, Any]] = []

    for message in messages:
        if isinstance(message, ModelRequest):
            for inp in message.inputs:
                if isinstance(inp, TextInput):
                    result.append({"role": "user", "content": [{"type": "input_text", "text": inp.content}]})

                elif isinstance(inp, ImageUrlInput):
                    result.append({"role": "user", "content": [{"type": "input_image", "image_url": inp.url}]})

                elif isinstance(inp, DocumentUrlInput):
                    result.append({"role": "user", "content": [{"type": "input_file", "file_url": inp.url}]})

                elif isinstance(inp, FileIdInput):
                    item: dict[str, Any] = {"type": "input_file", "file_id": inp.file_id}
                    if inp.filename is not None:
                        item["filename"] = inp.filename
                    result.append({
                        "role": "user",
                        "content": [item],
                    })

                elif isinstance(inp, BinaryInput):
                    b64 = base64.b64encode(inp.data).decode()
                    item: dict[str, Any] = {
                        "type": "input_file",
                        "file_data": f"data:{inp.media_type};base64,{b64}",
                        **inp.vendor_metadata,
                    }
                    result.append({"role": "user", "content": [item]})

                else:
                    raise UnsupportedInputError(type(inp).__name__, "openai-responses")

        elif isinstance(message, ModelResponse):
            # Reconstruct assistant message
            content: list[dict[str, Any]] = []
            if message.message:
                content.append({"type": "output_text", "text": message.message.content})
            if content:
                result.append({
                    "role": "assistant",
                    "content": content,
                })
            # Add function call items from the response
            for call in message.tool_calls.calls:
                result.append({
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                })

        elif isinstance(message, ToolResultsEvent):
            for r in message.results:
                result.append({
                    "type": "function_call_output",
                    "call_id": r.parent_id,
                    "output": r.content,
                })

    return result


def convert_messages(
    system_prompt: Iterable[str],
    messages: Iterable[BaseEvent],
) -> list[dict[str, str]]:
    # legacy prompt message format
    result: list[dict[str, str]] = [{"content": "\n".join(system_prompt), "role": "system"}]

    for message in messages:
        if isinstance(message, ModelRequest):
            for inp in message.inputs:
                if isinstance(inp, TextInput):
                    result.append(inp.to_api())
                elif isinstance(inp, ImageUrlInput):
                    result.append({
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": inp.url}}],
                    })
                elif isinstance(inp, BinaryInput):
                    if inp.kind == BinaryType.AUDIO:
                        b64 = base64.b64encode(inp.data).decode()
                        fmt = _MIME_TO_AUDIO_FORMAT.get(inp.media_type, inp.media_type.split("/", 1)[1])
                        item: dict[str, Any] = {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}
                    elif inp.kind == BinaryType.IMAGE:
                        b64 = base64.b64encode(inp.data).decode()
                        data_url = f"data:{inp.media_type};base64,{b64}"
                        item = {"type": "image_url", "image_url": {"url": data_url}, **inp.vendor_metadata}
                    else:
                        raise UnsupportedInputError(type(inp).__name__, "openai-completions")

                    result.append({"role": "user", "content": [item]})

                else:
                    raise UnsupportedInputError(type(inp).__name__, "openai-completions")

        elif isinstance(message, ModelResponse):
            result.append(message.to_api())

        elif isinstance(message, ToolResultsEvent):
            for r in message.results:
                result.append(r.to_api())

    return result


def _ensure_object_schema(params: dict[str, Any]) -> dict[str, Any]:
    """OpenAI requires tool parameters to be type: object with properties."""
    schema = dict(params)
    schema["type"] = "object"
    schema.setdefault("properties", {})
    schema.setdefault("additionalProperties", False)
    return schema


def tool_to_api(t: ToolSchema) -> dict[str, Any]:
    """Chat Completions API tool format."""
    if isinstance(t, FunctionToolSchema):
        return {
            "type": "function",
            "function": {
                "name": t.function.name,
                "description": t.function.description,
                "parameters": _ensure_object_schema(t.function.parameters),
            },
        }

    raise UnsupportedToolError(t.type, "openai-completions")


def tool_to_responses_api(t: ToolSchema) -> dict[str, Any]:
    """Responses API tool format — name/description at top level."""
    if isinstance(t, FunctionToolSchema):
        return {
            "type": "function",
            "name": t.function.name,
            "description": t.function.description,
            "parameters": _ensure_object_schema(t.function.parameters),
        }

    elif isinstance(t, WebSearchToolSchema):
        result: dict[str, Any] = {"type": "web_search"}
        if t.search_context_size is not None:
            result["search_context_size"] = t.search_context_size
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
            result["filters"] = {"allowed_domains": t.allowed_domains}
        return result

    elif isinstance(t, CodeExecutionToolSchema):
        # https://developers.openai.com/api/docs/guides/tools-code-interpreter
        return {"type": "code_interpreter", "container": {"type": "auto"}}

    elif isinstance(t, ShellToolSchema):
        # https://developers.openai.com/api/docs/guides/tools-shell
        result_shell: dict[str, Any] = {"type": "shell"}
        if t.environment is not None:
            env: dict[str, Any]
            if isinstance(t.environment, ContainerAutoEnvironment):
                env = {"type": "container_auto"}
                if t.environment.network_policy is not None:
                    env["network_policy"] = {
                        "type": "allowlist",
                        "allowed_domains": t.environment.network_policy.allowed_domains,
                    }
            elif isinstance(t.environment, ContainerReferenceEnvironment):
                env = {
                    "type": "container_reference",
                    "container_id": t.environment.container_id,
                }
            else:
                env = {"type": "local"}
            result_shell["environment"] = env
        return result_shell

    elif isinstance(t, ImageGenerationToolSchema):
        result: dict[str, Any] = {"type": "image_generation"}
        if t.quality is not None:
            result["quality"] = t.quality
        if t.size is not None:
            result["size"] = t.size
        if t.background is not None:
            result["background"] = t.background
        if t.output_format is not None:
            result["output_format"] = t.output_format
        if t.output_compression is not None:
            result["output_compression"] = t.output_compression
        if t.partial_images is not None:
            result["partial_images"] = t.partial_images
        return result

    elif isinstance(t, MCPServerToolSchema):
        # https://platform.openai.com/docs/guides/tools-remote-mcp
        result = {
            "type": "mcp",
            "server_label": t.server_label,
            "server_url": t.server_url,
            "require_approval": "never",
        }
        if t.description is not None:
            result["server_description"] = t.description
        if t.allowed_tools is not None:
            result["allowed_tools"] = t.allowed_tools
        if t.headers is not None:
            result["headers"] = t.headers
        elif t.authorization_token is not None:
            result["headers"] = {"Authorization": f"Bearer {t.authorization_token}"}
        return result

    raise UnsupportedToolError(t.type, "openai-responses")


def normalize_usage(usage: CompletionUsage) -> Usage:
    return Usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        cache_read_input_tokens=usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else None,
        cache_creation_input_tokens=usage.completion_tokens_details.reasoning_tokens
        if usage.completion_tokens_details
        else None,
    )


def normalize_responses_usage(usage: ResponseUsage) -> Usage:
    return Usage(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cache_read_input_tokens=usage.input_tokens_details.cached_tokens,
        cache_creation_input_tokens=usage.output_tokens_details.reasoning_tokens,
    )


_MIME_TO_AUDIO_FORMAT: dict[str, str] = {
    "audio/wav": "wav",
    "audio/mpeg": "mp3",
    "audio/ogg": "ogg",
    "audio/flac": "flac",
    "audio/aiff": "aiff",
    "audio/aac": "aac",
}
