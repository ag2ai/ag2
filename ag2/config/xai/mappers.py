# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from collections.abc import Iterable, Sequence
from typing import Any

from fast_depends.library.serializer import SerializerProto
from xai_sdk import tools as xai_tools
from xai_sdk.chat import (
    Response as XAIResponse,
)
from xai_sdk.chat import (
    assistant,
    chat_pb2,
    system,
    tool_result,
    user,
)
from xai_sdk.chat import (
    file as xai_file,
)
from xai_sdk.chat import (
    image as xai_image,
)
from xai_sdk.chat import (
    text as xai_text,
)
from xai_sdk.chat import (
    tool as xai_tool,
)
from xai_sdk.proto import usage_pb2

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
    ToolResultsEvent,
    UrlInput,
    Usage,
)
from ag2.exceptions import UnsupportedInputError, UnsupportedToolError
from ag2.response import ResponseProto
from ag2.tools.builtin.code_execution import CodeExecutionToolSchema
from ag2.tools.builtin.mcp_server import MCPServerToolSchema
from ag2.tools.builtin.web_search import WebSearchToolSchema
from ag2.tools.builtin.x_search import XSearchToolSchema
from ag2.tools.final import FunctionToolSchema
from ag2.tools.schemas import ToolSchema

from .events import XAIAssistantEvent

PROVIDER = "xai"


def _ensure_object_schema(params: dict[str, Any]) -> dict[str, Any]:
    """xAI tool parameters follow JSON Schema. Ensure top-level is an object."""
    schema = dict(params)
    schema["type"] = "object"
    schema.setdefault("properties", {})
    schema.setdefault("additionalProperties", False)
    return schema


def _ensure_additional_properties_false(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively set ``additionalProperties: false`` on every object node.

    xAI's strict JSON Schema mode (FORMAT_TYPE_JSON_SCHEMA) follows the OpenAI
    convention — every object schema must explicitly disallow extra keys.
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


def response_proto_to_format(response: ResponseProto | None) -> chat_pb2.ResponseFormat | None:
    """Convert AG2 ``ResponseProto`` to xAI's ``chat_pb2.ResponseFormat`` proto.

    xAI's chat.create accepts either a Pydantic class, a Literal alias, or a
    ResponseFormat proto. A plain JSON-schema dict is NOT accepted, so we build
    the proto ourselves with the schema serialized to JSON.
    """
    if not response or not response.json_schema:
        return None

    strict_schema = _ensure_additional_properties_false(response.json_schema)
    return chat_pb2.ResponseFormat(
        format_type=chat_pb2.FORMAT_TYPE_JSON_SCHEMA,
        schema=json.dumps(strict_schema),
    )


def tool_to_api(t: ToolSchema) -> chat_pb2.Tool:
    """Convert an AG2 ToolSchema to an xAI ``chat_pb2.Tool`` proto."""
    if isinstance(t, FunctionToolSchema):
        return xai_tool(
            name=t.function.name,
            description=t.function.description,
            parameters=_ensure_object_schema(t.function.parameters),
        )

    if isinstance(t, WebSearchToolSchema):
        kwargs: dict[str, Any] = {}
        if t.allowed_domains is not None:
            kwargs["allowed_domains"] = t.allowed_domains
        if t.blocked_domains is not None:
            kwargs["excluded_domains"] = t.blocked_domains
        if t.user_location is not None:
            if t.user_location.country is not None:
                kwargs["user_location_country"] = t.user_location.country
            if t.user_location.city is not None:
                kwargs["user_location_city"] = t.user_location.city
            if t.user_location.region is not None:
                kwargs["user_location_region"] = t.user_location.region
            if t.user_location.timezone is not None:
                kwargs["user_location_timezone"] = t.user_location.timezone
        return xai_tools.web_search(**kwargs)

    if isinstance(t, XSearchToolSchema):
        kwargs = {}
        if t.allowed_x_handles is not None:
            kwargs["allowed_x_handles"] = t.allowed_x_handles
        if t.excluded_x_handles is not None:
            kwargs["excluded_x_handles"] = t.excluded_x_handles
        if t.from_date is not None:
            kwargs["from_date"] = t.from_date
        if t.to_date is not None:
            kwargs["to_date"] = t.to_date
        if t.enable_image_understanding is not None:
            kwargs["enable_image_understanding"] = t.enable_image_understanding
        if t.enable_video_understanding is not None:
            kwargs["enable_video_understanding"] = t.enable_video_understanding
        return xai_tools.x_search(**kwargs)

    if isinstance(t, CodeExecutionToolSchema):
        return xai_tools.code_execution()

    if isinstance(t, MCPServerToolSchema):
        kwargs = {"server_url": t.server_url}
        if t.server_label is not None:
            kwargs["server_label"] = t.server_label
        if t.description is not None:
            kwargs["server_description"] = t.description
        if t.allowed_tools is not None:
            kwargs["allowed_tool_names"] = t.allowed_tools
        if t.authorization_token is not None:
            kwargs["authorization"] = f"Bearer {t.authorization_token}"
        elif t.headers is not None and "Authorization" in t.headers:
            kwargs["authorization"] = t.headers["Authorization"]
        if t.headers is not None:
            extra = {k: v for k, v in t.headers.items() if k != "Authorization"}
            if extra:
                kwargs["extra_headers"] = extra
        return xai_tools.mcp(**kwargs)

    raise UnsupportedToolError(t.type, PROVIDER)


def _content_from_input(part: Input, serializer: SerializerProto) -> chat_pb2.Content:
    """Convert a single AG2 ``Input`` part to an xAI ``Content`` proto."""
    if isinstance(part, TextInput):
        return xai_text(part.content)

    if isinstance(part, DataInput):
        return xai_text(serializer.encode(part.data).decode())

    if isinstance(part, UrlInput):
        if part.kind is BinaryType.IMAGE:
            return xai_image(part.url, detail=part.metadata.get("detail", "auto"))
        if part.kind in (BinaryType.DOCUMENT, BinaryType.BINARY):
            return xai_file(url=part.url, filename=part.metadata.get("filename"))
        raise UnsupportedInputError(f"UrlInput({part.kind.value})", PROVIDER)

    if isinstance(part, BinaryInput):
        if part.kind is BinaryType.IMAGE:
            b64 = base64.b64encode(part.data).decode()
            data_url = f"data:{part.media_type};base64,{b64}"
            detail = part.vendor_metadata.get("detail", "auto")
            return xai_image(data_url, detail=detail)
        if part.kind in (BinaryType.DOCUMENT, BinaryType.BINARY):
            filename = part.vendor_metadata.get("filename")
            if not filename:
                suffix = str(part.media_type).rsplit("/", 1)[-1].split("+", 1)[0]
                filename = f"file.{suffix}"
            return xai_file(
                data=part.data,
                filename=filename,
                mime_type=str(part.media_type),
            )
        raise UnsupportedInputError(f"BinaryInput({part.kind.value})", PROVIDER)

    if isinstance(part, FileIdInput):
        # xai_sdk rejects filename/mime_type when referencing by file_id
        return xai_file(file_id=part.file_id)

    raise UnsupportedInputError(type(part).__name__, PROVIDER)


def _tool_result_to_string(parts: Sequence[Input], serializer: SerializerProto) -> str:
    """xAI ``tool_result`` accepts only ``result: str`` (see ``xai_sdk.chat.tool_result``).

    Strategy mirrors other providers (single text → bare str) and aligns with xAI
    docs that recommend JSON for structured results. Multiple text/data fragments
    are emitted as a JSON array so the model sees the part boundary explicitly
    instead of a silent ``\\n`` flatten. Binary/URL inputs are not supported by
    the API for tool results.
    """
    fragments: list[str] = []
    for part in parts:
        if isinstance(part, TextInput):
            fragments.append(part.content)
        elif isinstance(part, DataInput):
            fragments.append(serializer.encode(part.data).decode())
        else:
            raise UnsupportedInputError(f"tool_result/{type(part).__name__}", PROVIDER)

    if len(fragments) == 1:
        return fragments[0]
    return json.dumps(fragments, ensure_ascii=False)


def convert_messages(
    system_prompt: Iterable[str],
    messages: Iterable[BaseEvent],
    serializer: SerializerProto,
) -> tuple[list[chat_pb2.Message], list["XAIResponse"]]:
    """Convert AG2 events to xAI helper messages + persisted assistant ``Response`` objects.

    Returns a 2-tuple:

    * ``messages`` — list of ``chat_pb2.Message`` to pass to ``chat.create(messages=...)``.
    * ``responses`` — list of ``xai_sdk.chat.Response`` objects (restored from
      ``XAIAssistantEvent.proto_bytes``) to ``chat.append(...)`` AFTER chat is
      created. This is the canonical way xai-sdk replays assistant turns with
      tool_calls — there is no public helper to construct them from primitives.

    The order is preserved: an ``XAIAssistantEvent`` followed by a
    ``ToolResultsEvent`` becomes a Response + tool_result message at the
    corresponding offset in the output stream.
    """
    result: list[chat_pb2.Message] = []
    prompt_text = "\n".join(s for s in system_prompt if s)
    if prompt_text:
        result.append(system(prompt_text))

    responses: list[XAIResponse] = []
    skip_next_model_response = False

    for message in messages:
        if isinstance(message, XAIAssistantEvent):
            proto = chat_pb2.GetChatCompletionResponse.FromString(message.proto_bytes)
            responses.append(XAIResponse(proto, None))
            # The companion ModelResponse (emitted alongside) carries the same
            # turn — skip it so we don't double-append the assistant message.
            skip_next_model_response = True

        elif isinstance(message, ModelResponse):
            if skip_next_model_response:
                skip_next_model_response = False
                continue
            # Degraded fallback: no XAIAssistantEvent → reconstruct text-only.
            # tool_calls cannot be replayed without the proto; they're lost.
            if message.message and message.message.content:
                result.append(assistant(message.message.content))

        elif isinstance(message, ToolResultsEvent):
            for r in message.results:
                payload = _tool_result_to_string(r.result.parts, serializer)
                result.append(tool_result(payload, tool_call_id=r.parent_id))

        elif isinstance(message, ModelRequest):
            contents: list[chat_pb2.Content] = [_content_from_input(p, serializer) for p in message.parts]
            result.append(user(*contents))

        elif isinstance(message, CompactionSummary):
            # Surface the summary as a user turn so it stays visible and gives a valid opening turn
            result.append(user(f"[Summary of earlier conversation]\n{message.summary}"))

    return result, responses


def normalize_usage(usage: usage_pb2.SamplingUsage | None) -> Usage:
    """Normalise xai-sdk usage object to AG2 ``Usage``.

    xai-sdk Usage proto fields (see ``usage_pb2.SamplingUsage``):
    ``prompt_tokens``, ``completion_tokens``, ``total_tokens``,
    ``reasoning_tokens``, ``cached_prompt_text_tokens``.
    """
    if usage is None:
        return Usage()

    return Usage(
        prompt_tokens=float(usage.prompt_tokens) if usage.prompt_tokens else None,
        completion_tokens=float(usage.completion_tokens) if usage.completion_tokens else None,
        total_tokens=float(usage.total_tokens) if usage.total_tokens else None,
        cache_read_input_tokens=float(usage.cached_prompt_text_tokens) if usage.cached_prompt_text_tokens else None,
        thinking_tokens=float(usage.reasoning_tokens) if usage.reasoning_tokens else None,
    )
