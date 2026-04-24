# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from fast_depends.library.serializer import SerializerProto
from google.genai import types

from autogen.beta.events import (
    BaseEvent,
    BinaryInput,
    BinaryType,
    DataInput,
    FileIdInput,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolResultsEvent,
    UrlInput,
    Usage,
)
from autogen.beta.exceptions import UnsupportedInputError, UnsupportedToolError
from autogen.beta.files.types import FileProvider
from autogen.beta.response import ResponseProto
from autogen.beta.tools.builtin.code_execution import CodeExecutionToolSchema
from autogen.beta.tools.builtin.skills import SkillsToolSchema
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


def _ensure_object_schema(params: dict[str, Any]) -> dict[str, Any]:
    """Gemini requires every function's parameters schema to be type=object.

    Parameterless functions produce ``{"type": "null"}`` (from pydantic/fast_depends)
    or ``{}`` — both rejected by Gemini with ``INVALID_ARGUMENT``.
    Normalise to ``{"type": "object", "properties": {}}``.
    """
    raw_type = str(params.get("type", "")).lower()
    if not params or raw_type in ("null", "none", ""):
        return {"type": "object", "properties": {}}
    return params


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
                    parameters=_ensure_object_schema(t.function.parameters),
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

        elif isinstance(t, SkillsToolSchema):
            raise UnsupportedToolError(t.type, "gemini")

        else:
            raise UnsupportedToolError(t.type, "gemini")

    result: list[types.Tool] = []
    if function_declarations:
        result.append(types.Tool(function_declarations=function_declarations))
    result.extend(extra_tools)

    return result or None


_URL_EXTENSION_TO_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".flv": "video/x-flv",
    ".wmv": "video/x-ms-wmv",
    ".3gp": "video/3gpp",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".html": "text/html",
    ".csv": "text/csv",
    ".json": "application/json",
    ".xml": "text/xml",
    ".md": "text/markdown",
}


def _mime_from_url(url: str) -> str | None:
    """Infer MIME type from URL path extension, or None if unknown."""
    path = urlparse(url).path
    dot = path.rfind(".")
    if dot != -1:
        ext = path[dot:].lower().split("?", 1)[0]
        mime = _URL_EXTENSION_TO_MIME.get(ext)
        if mime:
            return mime
    return None


def _apply_vendor_metadata(part: types.Part, metadata: dict[str, Any]) -> None:
    """Apply Gemini-specific vendor_metadata fields to a Part."""
    if not metadata:
        return

    if "media_resolution" in metadata:
        part.media_resolution = metadata["media_resolution"]

    if "video_metadata" in metadata:
        vm = metadata["video_metadata"]
        if isinstance(vm, dict):
            part.video_metadata = types.VideoMetadata(**vm)
        else:
            part.video_metadata = vm

    if "display_name" in metadata:
        if part.inline_data is not None:
            part.inline_data.display_name = metadata["display_name"]
        elif part.file_data is not None:
            part.file_data.display_name = metadata["display_name"]


def convert_messages(
    messages: Iterable[BaseEvent],
    serializer: SerializerProto,
) -> list[types.Content]:
    result: list[types.Content] = []

    for message in messages:
        if isinstance(message, ModelResponse):
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
                text_chunks: list[str] = []
                media_parts: list[types.FunctionResponsePart] = []
                for part in r.result.parts:
                    if isinstance(part, TextInput):
                        text_chunks.append(part.content)
                    elif isinstance(part, DataInput):
                        text_chunks.append(serializer.encode(part.data).decode())
                    elif isinstance(part, BinaryInput) and part.kind in (BinaryType.IMAGE, BinaryType.DOCUMENT):
                        media_parts.append(
                            types.FunctionResponsePart.from_bytes(data=part.data, mime_type=part.media_type)
                        )
                    elif isinstance(part, UrlInput) and part.kind in (BinaryType.IMAGE, BinaryType.DOCUMENT):
                        media_parts.append(
                            types.FunctionResponsePart.from_uri(file_uri=part.url, mime_type=_mime_from_url(part.url))
                        )
                    else:
                        raise UnsupportedInputError(type(part).__name__, "gemini")

                if text_chunks:
                    result_value = text_chunks[0] if len(text_chunks) == 1 else text_chunks
                    response_dict: dict[str, Any] = {"result": result_value}
                else:
                    response_dict = {}

                parts_list.append(
                    types.Part.from_function_response(
                        name=r.name or "",
                        response=response_dict,
                        parts=media_parts or None,
                    )
                )
            result.append(types.Content(role="user", parts=parts_list))

        elif isinstance(message, ModelRequest):
            parts: list[types.Part] = []
            for inp in message.parts:
                if isinstance(inp, TextInput):
                    parts.append(types.Part.from_text(text=inp.content))

                elif isinstance(inp, DataInput):
                    parts.append(types.Part.from_text(text=serializer.encode(inp.data).decode()))

                elif isinstance(inp, UrlInput):
                    mime = _mime_from_url(inp.url)
                    if mime is not None:
                        parts.append(types.Part.from_uri(file_uri=inp.url, mime_type=mime))
                    else:
                        parts.append(types.Part(file_data=types.FileData(file_uri=inp.url)))

                elif isinstance(inp, FileIdInput):
                    if (provider := getattr(inp, "provider", None)) and provider is not FileProvider.GEMINI:
                        raise UnsupportedInputError(
                            f"file uploaded via '{provider.value}' cannot be used with '{FileProvider.GEMINI.value}'",
                            "gemini",
                        )
                    file_uri = f"https://generativelanguage.googleapis.com/v1beta/{inp.file_id}"
                    parts.append(types.Part(file_data=types.FileData(file_uri=file_uri)))

                elif isinstance(inp, BinaryInput):
                    part = types.Part.from_bytes(data=inp.data, mime_type=inp.media_type)
                    _apply_vendor_metadata(part, inp.vendor_metadata)
                    parts.append(part)

                else:
                    raise UnsupportedInputError(type(inp).__name__, "gemini")

            if parts:
                result.append(types.Content(role="user", parts=parts))

    return result


def normalize_usage(metadata: Any) -> Usage:
    """Build usage from Gemini UsageMetadata, normalizing to standard keys."""

    cache_read = _to_float(metadata.cached_content_token_count) or None
    return Usage(
        prompt_tokens=_to_float(metadata.prompt_token_count),
        completion_tokens=_to_float(metadata.candidates_token_count),
        total_tokens=_to_float(metadata.total_token_count),
        cache_read_input_tokens=cache_read,
    )


def _to_float(value: Any) -> float | None:
    return float(value) if value is not None else None
