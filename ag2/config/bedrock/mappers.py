# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

from fast_depends.library.serializer import SerializerProto

from ag2.compact import CompactionSummary
from ag2.events import (
    BaseEvent,
    BinaryInput,
    BinaryType,
    DataInput,
    Input,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolErrorEvent,
    ToolResultEvent,
    ToolResultsEvent,
    Usage,
)
from ag2.exceptions import UnsupportedInputError, UnsupportedToolError
from ag2.response import ResponseProto
from ag2.tools.final import FunctionToolSchema
from ag2.tools.schemas import ToolSchema

if TYPE_CHECKING:
    from types_boto3_bedrock_runtime.literals import (
        ConversationRoleType,
        DocumentFormatType,
        ImageFormatType,
        VideoFormatType,
    )
    from types_boto3_bedrock_runtime.type_defs import (
        ContentBlockTypeDef,
        DocumentBlockTypeDef,
        ImageBlockTypeDef,
        JsonSchemaDefinitionTypeDef,
        MessageTypeDef,
        OutputConfigTypeDef,
        ToolResultBlockTypeDef,
        ToolResultContentBlockTypeDef,
        ToolSpecificationTypeDef,
        ToolTypeDef,
        ToolUseBlockTypeDef,
        VideoBlockTypeDef,
    )

logger = logging.getLogger(__name__)

_IMAGE_MEDIA_TO_FORMAT: "dict[str, ImageFormatType]" = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}

_DOCUMENT_MEDIA_TO_FORMAT: "dict[str, DocumentFormatType]" = {
    "application/pdf": "pdf",
    "text/csv": "csv",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/html": "html",
    "text/plain": "txt",
    "text/markdown": "md",
}

_VIDEO_MEDIA_TO_FORMAT: "dict[str, VideoFormatType]" = {
    "video/x-matroska": "mkv",
    "video/quicktime": "mov",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "video/x-flv": "flv",
    "video/mpeg": "mpeg",
    "video/x-ms-wmv": "wmv",
    "video/3gpp": "three_gp",
}

# Converse document names allow only alphanumerics, single spaces, hyphens, parentheses, brackets
_DOC_NAME_INVALID = re.compile(r"[^A-Za-z0-9\-()\[\] ]+")

# User filler for assistant-first histories (Converse requires a user turn first)
_CONTINUATION_TEXT = "Please continue."


def _ensure_additional_properties_false(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively add additionalProperties: false — Converse rejects any other value."""
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
    """Convert a ResponseProto to Converse outputConfig (native structured output).

    Converse expects the JSON schema as a serialized string inside
    ``textFormat.structure.jsonSchema``.
    """
    if not response or not response.json_schema:
        return None

    strict_schema = _ensure_additional_properties_false(response.json_schema)
    json_schema: JsonSchemaDefinitionTypeDef = {
        "schema": json.dumps(strict_schema),
        "name": response.name,
    }
    if response.description:
        json_schema["description"] = response.description

    output_config: OutputConfigTypeDef = {
        "textFormat": {
            "type": "json_schema",
            "structure": {"jsonSchema": json_schema},
        },
    }
    return dict(output_config)


def _ensure_object_schema(params: dict[str, Any]) -> dict[str, Any]:
    """Converse requires toolSpec inputSchema.json to be type: object."""
    schema = dict(params)
    schema["type"] = "object"
    schema.setdefault("properties", {})
    return schema


def tool_to_api(t: ToolSchema) -> dict[str, Any]:
    if isinstance(t, FunctionToolSchema):
        spec: ToolSpecificationTypeDef = {
            "name": t.function.name,
            "inputSchema": {"json": _ensure_object_schema(t.function.parameters)},
        }
        # Converse rejects empty description strings
        if t.function.description:
            spec["description"] = t.function.description
        tool: ToolTypeDef = {"toolSpec": spec}
        return dict(tool)

    raise UnsupportedToolError(t.type, "bedrock")


def _sanitize_document_name(name: str | None) -> str:
    """Strip characters Converse rejects in document names."""
    if not name:
        return "document"
    cleaned = _DOC_NAME_INVALID.sub(" ", name)
    cleaned = " ".join(cleaned.split())
    return cleaned or "document"


def _image_block(part: BinaryInput) -> "ImageBlockTypeDef":
    image_format = _IMAGE_MEDIA_TO_FORMAT.get(part.media_type)
    if image_format is None:
        raise UnsupportedInputError(f"BinaryInput({part.media_type})", "bedrock")
    return {"format": image_format, "source": {"bytes": part.data}}


def _document_block(part: BinaryInput) -> "DocumentBlockTypeDef":
    document_format = _DOCUMENT_MEDIA_TO_FORMAT.get(part.media_type)
    if document_format is None:
        raise UnsupportedInputError(f"BinaryInput({part.media_type})", "bedrock")
    return {
        "format": document_format,
        "name": _sanitize_document_name(part.vendor_metadata.get("filename")),
        "source": {"bytes": part.data},
    }


def _video_block(part: BinaryInput) -> "VideoBlockTypeDef":
    video_format = _VIDEO_MEDIA_TO_FORMAT.get(part.media_type)
    if video_format is None:
        raise UnsupportedInputError(f"BinaryInput({part.media_type})", "bedrock")
    return {"format": video_format, "source": {"bytes": part.data}}


def _input_to_text(inp: Input, serializer: SerializerProto) -> str:
    """Render a non-binary Input as Converse text — the one conversion both positions share."""
    if isinstance(inp, TextInput):
        return inp.content
    elif isinstance(inp, DataInput):
        return serializer.encode(inp.data).decode()
    # UrlInput / FileIdInput: Converse sources accept bytes only; no Files API
    raise UnsupportedInputError(type(inp).__name__, "bedrock")


def _input_to_block(inp: Input, serializer: SerializerProto) -> "ContentBlockTypeDef":
    """Convert an Input to a block in *message* position.

    Converse accepts different block sets in its two content positions — a message block may also
    carry ``toolUse``/``toolResult``/``audio``, a toolResult block ``json``/``searchResult`` — so
    each position owns the list of keys it may emit while the conversions stay shared. The two
    ladders look alike only because today's inputs land in the overlap; an ``audio`` input would
    belong in this one alone.
    """
    if isinstance(inp, BinaryInput):
        if inp.kind is BinaryType.IMAGE:
            return {"image": _image_block(inp)}
        elif inp.kind is BinaryType.DOCUMENT:
            return {"document": _document_block(inp)}
        elif inp.kind is BinaryType.VIDEO:
            return {"video": _video_block(inp)}
        raise UnsupportedInputError(f"BinaryInput({inp.kind.value})", "bedrock")
    return {"text": _input_to_text(inp, serializer)}


def _input_to_tool_result_block(inp: Input, serializer: SerializerProto) -> "ToolResultContentBlockTypeDef":
    """Convert an Input to a block in *toolResult* position. See :func:`_input_to_block`."""
    if isinstance(inp, BinaryInput):
        if inp.kind is BinaryType.IMAGE:
            return {"image": _image_block(inp)}
        elif inp.kind is BinaryType.DOCUMENT:
            return {"document": _document_block(inp)}
        elif inp.kind is BinaryType.VIDEO:
            return {"video": _video_block(inp)}
        raise UnsupportedInputError(f"BinaryInput({inp.kind.value})", "bedrock")
    return {"text": _input_to_text(inp, serializer)}


def _append_blocks(
    result: list["MessageTypeDef"],
    role: "ConversationRoleType",
    blocks: list["ContentBlockTypeDef"],
) -> None:
    """Append blocks, merging same-role runs (Converse requires user-first, alternating roles)."""
    if not blocks:
        return

    if result and result[-1]["role"] == role:
        # MessageTypeDef declares content as a Sequence; every run here was built as a list
        cast("list[ContentBlockTypeDef]", result[-1]["content"]).extend(blocks)
        return

    if not result and role == "assistant":
        result.append({"role": "user", "content": [{"text": _CONTINUATION_TEXT}]})

    result.append({"role": role, "content": blocks})


def convert_messages(
    messages: Iterable[BaseEvent],
    serializer: SerializerProto,
) -> list[dict[str, Any]]:
    event_list = list(messages)

    # toolUse ids present in the conversation — toolResults without a match are dropped
    valid_tool_ids: set[str] = set()
    for message in event_list:
        if isinstance(message, ModelResponse):
            for call in message.tool_calls.calls:
                valid_tool_ids.add(call.id)

    # Mirror case: toolUse blocks whose result never persisted are dropped too.
    # Loose ToolResultEvent leaves count as resolving their parent.
    resolved_tool_ids: set[str] = set()
    for message in event_list:
        if isinstance(message, ToolResultsEvent):
            for r in message.results:
                if r.parent_id:
                    resolved_tool_ids.add(r.parent_id)
        elif isinstance(message, (ToolResultEvent, ToolErrorEvent)) and message.parent_id:
            resolved_tool_ids.add(message.parent_id)

    # Pre-populate from wrappers so the loose-leaf fallback below doesn't
    # double-emit a toolResult that a later ToolResultsEvent also carries.
    emitted_result_ids: set[str] = set()
    for message in event_list:
        if isinstance(message, ToolResultsEvent):
            for r in message.results:
                if r.parent_id in valid_tool_ids:
                    emitted_result_ids.add(r.parent_id)

    result: list[MessageTypeDef] = []

    for message in event_list:
        if isinstance(message, ModelResponse):
            content: list[ContentBlockTypeDef] = []
            if message.message:
                content.append({"text": message.message.content})
            # Orphan toolUse blocks are dropped; assistant text is kept.
            for call in message.tool_calls.calls:
                if call.id not in resolved_tool_ids:
                    logger.warning(
                        "Dropping orphan toolUse id=%s name=%s (no matching toolResult). "
                        "See mappers.py comment for context.",
                        call.id,
                        call.name,
                    )
                    continue
                tool_use: ToolUseBlockTypeDef = {
                    "toolUseId": call.id,
                    "name": call.name,
                    "input": json.loads(call.arguments or "{}"),
                }
                content.append({"toolUse": tool_use})
            _append_blocks(result, "assistant", content)

        elif isinstance(message, ToolResultsEvent):
            tool_results: list[ContentBlockTypeDef] = []
            for r in message.results:
                # Converse rejects a toolResult with no toolUse in the prior
                # turn — applies even when compaction removed every toolUse.
                # (A None parent_id never matches; spelling it out is what narrows toolUseId to str.)
                if r.parent_id is None or r.parent_id not in valid_tool_ids:
                    continue
                tool_result: ToolResultBlockTypeDef = {
                    "toolUseId": r.parent_id,
                    "content": [_input_to_tool_result_block(part, serializer) for part in r.result.parts],
                }
                if isinstance(r, ToolErrorEvent):
                    tool_result["status"] = "error"
                tool_results.append({"toolResult": tool_result})
            _append_blocks(result, "user", tool_results)

        elif isinstance(message, ModelRequest):
            blocks = [_input_to_block(inp, serializer) for inp in message.parts]
            _append_blocks(result, "user", blocks)

        elif isinstance(message, CompactionSummary):
            # Surface the summary as a user turn so it stays visible and gives a valid opening turn
            _append_blocks(result, "user", [{"text": f"[Summary of earlier conversation]\n{message.summary}"}])

        elif isinstance(message, (ToolResultEvent, ToolErrorEvent)):
            # Loose result whose ToolResultsEvent wrapper never persisted.
            if (
                message.parent_id is not None
                and message.parent_id in valid_tool_ids
                and message.parent_id not in emitted_result_ids
            ):
                emitted_result_ids.add(message.parent_id)
                tool_result = {
                    "toolUseId": message.parent_id,
                    "content": [_input_to_tool_result_block(part, serializer) for part in message.result.parts],
                }
                if isinstance(message, ToolErrorEvent):
                    tool_result["status"] = "error"
                _append_blocks(result, "user", [{"toolResult": tool_result}])

    return [dict(turn) for turn in result]


def normalize_usage(raw: dict[str, Any]) -> Usage:
    """Normalize Converse usage keys to standard format."""
    prompt = float(raw.get("inputTokens", 0))
    completion = float(raw.get("outputTokens", 0))
    total = raw.get("totalTokens")
    cache_read = raw.get("cacheReadInputTokens")
    cache_write = raw.get("cacheWriteInputTokens")
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=float(total) if total is not None else prompt + completion,
        cache_read_input_tokens=float(cache_read) if cache_read else None,
        cache_creation_input_tokens=float(cache_write) if cache_write else None,
    )
