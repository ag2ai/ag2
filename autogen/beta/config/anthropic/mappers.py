# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from collections.abc import Iterable
from typing import Any

from fast_depends.library.serializer import SerializerProto

from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, TextInput, ToolResultsEvent
from autogen.beta.events.input_events import (
    BinaryInput,
    BinaryType,
    DataInput,
    FileIdInput,
    UrlInput,
)
from autogen.beta.events.types import Usage
from autogen.beta.exceptions import UnsupportedInputError, UnsupportedToolError
from autogen.beta.files.types import FileProvider
from autogen.beta.response import ResponseProto
from autogen.beta.tools.builtin.code_execution import CodeExecutionToolSchema
from autogen.beta.tools.builtin.mcp_server import MCPServerToolSchema
from autogen.beta.tools.builtin.memory import MemoryToolSchema
from autogen.beta.tools.builtin.shell import ShellToolSchema
from autogen.beta.tools.builtin.skills import SkillsToolSchema
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


def tool_to_api(t: ToolSchema) -> dict[str, Any]:
    if isinstance(t, FunctionToolSchema):
        return {
            "name": t.function.name,
            "description": t.function.description,
            "input_schema": _ensure_object_schema(t.function.parameters),
        }

    elif isinstance(t, WebSearchToolSchema):
        result: dict[str, Any] = {"type": t.web_search_version, "name": "web_search"}
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
        return {"type": t.version, "name": "code_execution"}

    elif isinstance(t, WebFetchToolSchema):
        result = {"type": t.web_fetch_version, "name": "web_fetch"}
        if t.max_uses is not None:
            result["max_uses"] = t.max_uses
        if t.allowed_domains is not None:
            result["allowed_domains"] = t.allowed_domains
        if t.blocked_domains is not None:
            result["blocked_domains"] = t.blocked_domains
        if t.citations is not None:
            result["citations"] = {"enabled": t.citations}
        if t.max_content_tokens is not None:
            result["max_content_tokens"] = t.max_content_tokens
        return result

    elif isinstance(t, MemoryToolSchema):
        # https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool
        return {"type": t.version, "name": "memory"}

    elif isinstance(t, ShellToolSchema):
        # https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool
        return {"type": t.version, "name": "bash"}

    elif isinstance(t, SkillsToolSchema):
        # Skills are handled via the container parameter, not the tools[] array.
        # Use extract_skills_for_container() in the client instead.
        raise UnsupportedToolError(t.type, "anthropic")

    elif isinstance(t, MCPServerToolSchema):
        # https://platform.claude.com/docs/en/docs/agents-and-tools/mcp-connector
        result = {
            "type": "mcp_toolset",
            "mcp_server_name": t.server_label,
        }
        if t.allowed_tools is not None:
            result["default_config"] = {"enabled": False}
            result["configs"] = {name: {"enabled": True} for name in t.allowed_tools}
        if t.blocked_tools is not None:
            configs: dict[str, Any] = result.get("configs", {})
            configs.update({name: {"enabled": False} for name in t.blocked_tools})
            result["configs"] = configs
        return result

    raise UnsupportedToolError(t.type, "anthropic")


def extract_mcp_servers(tools: Iterable[ToolSchema]) -> list[dict[str, Any]]:
    """Extract Anthropic mcp_servers definitions from MCPServerToolSchema instances."""
    servers: list[dict[str, Any]] = []
    for t in tools:
        if isinstance(t, MCPServerToolSchema):
            server: dict[str, Any] = {
                "type": "url",
                "url": t.server_url,
                "name": t.server_label,
            }
            if t.authorization_token is not None:
                server["authorization_token"] = t.authorization_token
            servers.append(server)
    return servers


def extract_skills_for_container(tools: Iterable[ToolSchema]) -> list[dict[str, Any]]:
    """Extract Anthropic skills from SkillsToolSchema instances for the container parameter."""
    skills: list[dict[str, Any]] = []
    for t in tools:
        if isinstance(t, SkillsToolSchema):
            for s in t.skills:
                entry: dict[str, Any] = {
                    "type": "anthropic",
                    "skill_id": s.id,
                    "version": str(s.version) if s.version is not None else "latest",
                }
                skills.append(entry)
    return skills


_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})

# Anthropic content block keys that are safe to pass through from vendor_metadata.
_ANTHROPIC_VENDOR_KEYS = frozenset({"cache_control", "citations"})


def _file_id_block_type(filename: str | None) -> str:
    """Infer Anthropic content block type from filename extension."""
    if filename:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if f".{ext}" in _IMAGE_EXTENSIONS:
            return "image"
    return "document"


def has_file_id_references(messages: Iterable[BaseEvent]) -> bool:
    """True if any message (user turn or tool result) references a file_id.

    Used by the client to auto-inject the `files-api-2025-04-14` beta header.
    """
    for msg in messages:
        if isinstance(msg, ModelRequest):
            if any(isinstance(p, FileIdInput) for p in msg.parts):
                return True
        elif isinstance(msg, ToolResultsEvent):
            for r in msg.results:
                if any(isinstance(p, FileIdInput) for p in r.result.parts):
                    return True
    return False


def convert_messages(
    messages: Iterable[BaseEvent],
    serializer: SerializerProto,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for message in messages:
        if isinstance(message, ModelResponse):
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
            tool_results = []
            for r in message.results:
                parts: list[dict[str, Any]] = []
                for part in r.result.parts:
                    if isinstance(part, TextInput):
                        parts.append({"type": "text", "text": part.content})
                    elif isinstance(part, DataInput):
                        parts.append({"type": "text", "text": serializer.encode(part.data).decode()})
                    elif isinstance(part, BinaryInput):
                        if part.kind is BinaryType.IMAGE:
                            b64 = base64.b64encode(part.data).decode()
                            parts.append({
                                "type": "image",
                                "source": {"type": "base64", "media_type": part.media_type, "data": b64},
                            })
                        elif part.kind is BinaryType.DOCUMENT:
                            b64 = base64.b64encode(part.data).decode()
                            parts.append({
                                "type": "document",
                                "source": {"type": "base64", "media_type": part.media_type, "data": b64},
                            })
                        else:
                            raise UnsupportedInputError(f"BinaryInput({part.kind.value})", "anthropic")
                    elif isinstance(part, UrlInput):
                        if part.kind is BinaryType.IMAGE:
                            parts.append({"type": "image", "source": {"type": "url", "url": part.url}})
                        elif part.kind in (BinaryType.DOCUMENT, BinaryType.BINARY):
                            parts.append({"type": "document", "source": {"type": "url", "url": part.url}})
                        else:
                            raise UnsupportedInputError(f"UrlInput({part.kind.value})", "anthropic")
                    elif isinstance(part, FileIdInput):
                        block_type = _file_id_block_type(part.filename)
                        parts.append({
                            "type": block_type,
                            "source": {"type": "file", "file_id": part.file_id},
                        })
                    else:
                        raise UnsupportedInputError(type(part).__name__, "anthropic")

                if len(parts) == 1 and (part := parts[0])["type"] == "text":
                    tool_content: str | list[dict[str, Any]] = part["text"]
                else:
                    tool_content = parts
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": r.parent_id,
                    "content": tool_content,
                })
            result.append({"role": "user", "content": tool_results})

        elif isinstance(message, ModelRequest):
            content_parts: list[dict[str, Any]] = []
            for inp in message.parts:
                if isinstance(inp, TextInput):
                    content_parts.append({"type": "text", "text": inp.content})

                elif isinstance(inp, DataInput):
                    content_parts.append({"type": "text", "text": serializer.encode(inp.data).decode()})

                elif isinstance(inp, FileIdInput):
                    if (provider := getattr(inp, "provider", None)) and provider is not FileProvider.ANTHROPIC:
                        raise UnsupportedInputError(
                            f"file uploaded via '{provider.value}' cannot be used with '{FileProvider.ANTHROPIC.value}'",
                            "anthropic",
                        )

                    block_type = _file_id_block_type(inp.filename)
                    content_parts.append({"type": block_type, "source": {"type": "file", "file_id": inp.file_id}})

                elif isinstance(inp, UrlInput):
                    if inp.kind is BinaryType.IMAGE:
                        content_parts.append({"type": "image", "source": {"type": "url", "url": inp.url}})

                    elif inp.kind in (BinaryType.DOCUMENT, BinaryType.BINARY):
                        content_parts.append({"type": "document", "source": {"type": "url", "url": inp.url}})

                    else:
                        raise UnsupportedInputError(f"UrlInput({inp.kind.value})", "anthropic")

                elif isinstance(inp, BinaryInput):
                    extra = {k: v for k, v in inp.vendor_metadata.items() if k in _ANTHROPIC_VENDOR_KEYS}
                    if inp.kind is BinaryType.IMAGE:
                        b64 = base64.b64encode(inp.data).decode()
                        item: dict[str, Any] = {
                            "type": "image",
                            "source": {"type": "base64", "media_type": inp.media_type, "data": b64},
                            **extra,
                        }
                        content_parts.append(item)

                    elif inp.kind is BinaryType.DOCUMENT:
                        b64 = base64.b64encode(inp.data).decode()
                        item = {
                            "type": "document",
                            "source": {"type": "base64", "media_type": inp.media_type, "data": b64},
                            **extra,
                        }
                        content_parts.append(item)

                    else:
                        raise UnsupportedInputError(f"BinaryInput({inp.kind.value})", "anthropic")

                else:
                    raise UnsupportedInputError(type(inp).__name__, "anthropic")

            if content_parts:
                if len(content_parts) == 1 and (part := content_parts[0])["type"] == "text":
                    content: str | list[dict[str, Any]] = part["text"]
                else:
                    content = content_parts
                result.append({"role": "user", "content": content})

    return result


def normalize_usage(raw: dict[str, Any]) -> Usage:
    """Normalize Anthropic's native usage keys to standard format."""
    cc = raw.get("cache_creation_input_tokens")
    cr = raw.get("cache_read_input_tokens")
    prompt = float(raw.get("input_tokens", 0))
    completion = float(raw.get("output_tokens", 0))
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        cache_creation_input_tokens=float(cc) if cc else None,
        cache_read_input_tokens=float(cr) if cr else None,
    )
