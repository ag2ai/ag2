# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from collections.abc import Iterable, Sequence
from typing import Any

from fast_depends.library.serializer import SerializerProto
from mistralai.client.models import (
    AssistantMessage,
    ContentChunk,
    DocumentURLChunk,
    FileChunk,
    Function,
    ImageGenerationTool,
    ImageURL,
    ImageURLChunk,
    JSONSchema,
    ResponseFormat,
    SystemMessage,
    TextChunk,
    ThinkChunk,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from ag2.compact import CompactionSummary
from ag2.events import (
    BaseEvent,
    BinaryInput,
    BinaryType,
    DataInput,
    FileIdInput,
    Input,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolResult,
    ToolResultsEvent,
    UrlInput,
    Usage,
)
from ag2.exceptions import UnsupportedInputError, UnsupportedToolError
from ag2.response import ResponseProto
from ag2.tools.builtin.image_generation import IMAGE_GENERATION_TOOL_NAME, ImageGenerationToolSchema
from ag2.tools.final import FunctionToolSchema
from ag2.tools.schemas import ToolSchema

PROVIDER = "mistral"

_SERVER_TOOL_ALIASES = {"generate_image": IMAGE_GENERATION_TOOL_NAME}

ChatMessage = AssistantMessage | SystemMessage | ToolMessage | UserMessage


def _ensure_additional_properties_false(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively set ``additionalProperties: false`` on every object node.

    Required by Mistral's strict structured-output mode.
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


def _ensure_object_schema(params: dict[str, Any]) -> dict[str, Any]:
    """Mistral tool parameters follow JSON Schema. Ensure the top level is an object."""
    schema = dict(params)
    schema["type"] = "object"
    schema.setdefault("properties", {})
    return schema


def response_proto_to_format(response: ResponseProto | None) -> ResponseFormat | None:
    """Convert an AG2 ``ResponseProto`` to Mistral's ``ResponseFormat``.

    Built from the SDK model, not a dict: the schema body is ``schema_definition``,
    aliased to ``schema`` on the wire.
    """
    if not response or not response.json_schema:
        return None

    return ResponseFormat(
        type="json_schema",
        json_schema=JSONSchema(
            name=response.name,
            schema_definition=_ensure_additional_properties_false(response.json_schema),
            description=response.description or None,
            strict=True,
        ),
    )


def tool_to_api(t: ToolSchema) -> Tool:
    """Convert an AG2 ``ToolSchema`` to a Mistral tool.

    Function tools plus ``image_generation``. Mistral's other server-side tools
    (web search, code interpreter) are rejected by chat-completions.
    """
    if isinstance(t, FunctionToolSchema):
        return Tool(
            function=Function(
                name=t.function.name,
                description=t.function.description,
                parameters=_ensure_object_schema(t.function.parameters),
            )
        )

    if isinstance(t, ImageGenerationToolSchema):
        # Mistral exposes no knobs on this tool; size/quality/format are ignored.
        return ImageGenerationTool()

    raise UnsupportedToolError(t.type, PROVIDER)


def server_tool_name(name: str) -> str:
    """Map Mistral's server-side tool name onto AG2's.

    A mismatch makes the agent's tool lookup fail and log a spurious not-found.
    """
    return _SERVER_TOOL_ALIASES.get(name, name)


def server_tool_result(content: Any) -> ToolResult:
    """Wrap a server-executed tool result; image results carry a ``url``."""
    text = content if isinstance(content, str) else ""
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None

    if isinstance(payload, dict) and isinstance(payload.get("url"), str):
        return ToolResult(UrlInput(payload["url"], kind=BinaryType.IMAGE))
    return ToolResult(text)


def _image_chunk(url: str, detail: Any) -> ImageURLChunk:
    """Send a bare URL unless a ``detail`` level is set; Mistral accepts either."""
    if detail is None:
        return ImageURLChunk(image_url=url)
    return ImageURLChunk(image_url=ImageURL(url=url, detail=detail))


def _content_from_input(part: Input, serializer: SerializerProto) -> ContentChunk:
    """Convert a single AG2 ``Input`` part to a Mistral content chunk."""
    if isinstance(part, TextInput):
        return TextChunk(text=part.content)

    if isinstance(part, DataInput):
        return TextChunk(text=serializer.encode(part.data).decode())

    if isinstance(part, UrlInput):
        if part.kind is BinaryType.IMAGE:
            return _image_chunk(part.url, part.metadata.get("detail"))
        if part.kind in (BinaryType.DOCUMENT, BinaryType.BINARY):
            return DocumentURLChunk(
                document_url=part.url,
                document_name=part.metadata.get("filename"),
            )
        raise UnsupportedInputError(f"UrlInput({part.kind.value})", PROVIDER)

    if isinstance(part, BinaryInput):
        b64 = base64.b64encode(part.data).decode()
        if part.kind is BinaryType.IMAGE:
            return _image_chunk(f"data:{part.media_type};base64,{b64}", part.vendor_metadata.get("detail"))
        if part.kind in (BinaryType.DOCUMENT, BinaryType.BINARY):
            return DocumentURLChunk(
                document_url=f"data:{part.media_type};base64,{b64}",
                document_name=part.vendor_metadata.get("filename"),
            )
        raise UnsupportedInputError(f"BinaryInput({part.kind.value})", PROVIDER)

    if isinstance(part, FileIdInput):
        return FileChunk(file_id=part.file_id)

    raise UnsupportedInputError(type(part).__name__, PROVIDER)


def _message_content(parts: Sequence[Input], serializer: SerializerProto) -> str | list[ContentChunk]:
    """Collapse a single text part to a bare string; otherwise emit chunks.

    Mistral accepts ``str`` or ``list[ContentChunk]`` for user and tool content.
    """
    chunks = [_content_from_input(p, serializer) for p in parts]
    if len(chunks) == 1 and isinstance(chunks[0], TextChunk):
        return chunks[0].text
    return chunks


def _tool_calls_to_api(message: ModelResponse) -> list[ToolCall] | None:
    if not message.tool_calls.calls:
        return None
    return [
        ToolCall(
            id=call.id,
            function={"name": call.name, "arguments": call.arguments or "{}"},
        )
        for call in message.tool_calls.calls
    ]


def convert_messages(
    system_prompt: Iterable[str],
    messages: Iterable[BaseEvent],
    serializer: SerializerProto,
) -> list[ChatMessage]:
    """Convert AG2 events to Mistral chat messages."""
    result: list[ChatMessage] = []

    prompt_text = "\n".join(s for s in system_prompt if s)
    if prompt_text:
        result.append(SystemMessage(content=prompt_text))

    for message in messages:
        if isinstance(message, ModelRequest):
            result.append(UserMessage(content=_message_content(message.parts, serializer)))

        elif isinstance(message, CompactionSummary):
            # A user turn keeps the summary visible and is a valid opening turn.
            result.append(UserMessage(content=f"[Summary of earlier conversation]\n{message.summary}"))

        elif isinstance(message, ModelResponse):
            result.append(
                AssistantMessage(
                    content=message.message.content if message.message else None,
                    tool_calls=_tool_calls_to_api(message),
                )
            )

        elif isinstance(message, ToolResultsEvent):
            # Batch only. History also holds each constituent ToolResultEvent;
            # mapping those too would send every result twice.
            for r in message.results:
                result.append(
                    ToolMessage(
                        content=_message_content(r.result.parts, serializer),
                        tool_call_id=r.parent_id,
                    )
                )

    return result


def split_content(content: Any) -> tuple[str, str]:
    """Split a Mistral message content value into ``(text, reasoning)``.

    Content is a bare ``str`` or a list of chunks; ``ThinkChunk`` holds the
    reasoning trace, kept apart so it lands on ``ModelReasoning``.
    """
    if content is None:
        return "", ""

    if isinstance(content, str):
        return content, ""

    text_parts: list[str] = []
    reasoning_parts: list[str] = []

    for chunk in content:
        if isinstance(chunk, str):
            text_parts.append(chunk)
        elif isinstance(chunk, TextChunk):
            text_parts.append(chunk.text)
        elif isinstance(chunk, ThinkChunk):
            reasoning_parts.append(_think_text(chunk.thinking))
        # Reference/file/tool chunks carry citation metadata, not answer text.

    return "".join(text_parts), "".join(reasoning_parts)


def _think_text(thinking: Any) -> str:
    """``ThinkChunk.thinking`` is itself a list of chunks (usually ``TextChunk``)."""
    if thinking is None:
        return ""
    if isinstance(thinking, str):
        return thinking
    parts: list[str] = []
    for item in thinking:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, TextChunk):
            parts.append(item.text)
    return "".join(parts)


def json_arguments(arguments: Any) -> str:
    """Mistral returns tool arguments as a JSON string, but tolerate a dict."""
    if arguments is None or arguments == "":
        return "{}"
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments)


def normalize_usage(raw: Any | None) -> Usage:
    """Normalise Mistral's ``UsageInfo`` to AG2 ``Usage``.

    ``prompt_tokens_details`` is a pydantic extra, so it is read defensively.
    """
    if raw is None:
        return Usage()

    prompt = getattr(raw, "prompt_tokens", None)
    completion = getattr(raw, "completion_tokens", None)
    total = getattr(raw, "total_tokens", None)

    prompt_value = float(prompt) if isinstance(prompt, (int, float)) else None
    completion_value = float(completion) if isinstance(completion, (int, float)) else None
    if isinstance(total, (int, float)):
        total_value: float | None = float(total)
    elif prompt_value is not None or completion_value is not None:
        total_value = (prompt_value or 0) + (completion_value or 0)
    else:
        total_value = None

    cached: float | None = None
    details = getattr(raw, "prompt_tokens_details", None)
    raw_cached = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
    if isinstance(raw_cached, (int, float)):
        cached = float(raw_cached)

    return Usage(
        prompt_tokens=prompt_value,
        completion_tokens=completion_value,
        total_tokens=total_value,
        cache_read_input_tokens=cached,
    )
