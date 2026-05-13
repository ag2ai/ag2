# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from base64 import b64encode
from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from ag_ui.core import (
    AssistantMessage,
    AudioInputContent,
    BinaryInputContent,
    DocumentInputContent,
    FunctionCall,
    ImageInputContent,
    InputContentDataSource,
    InputContentUrlSource,
    Message,
    TextInputContent,
    ToolCall,
    ToolMessage,
    UserMessage,
    VideoInputContent,
)
from fast_depends.library.serializer import SerializerProto

from autogen.beta.events import (
    BaseEvent,
    BinaryInput,
    BinaryType,
    DataInput,
    FileIdInput,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolErrorEvent,
    ToolResult,
    ToolResultEvent,
    ToolResultsEvent,
    UrlInput,
)


def events_to_agui_messages(
    history: Iterable[BaseEvent],
    serializer: SerializerProto,
) -> list[Message]:
    messages: list[Message] = []
    seen_tool_result_ids: set[str] = set()
    for event in history:
        if isinstance(event, ModelRequest):
            if not event.parts:
                # Empty turn (e.g. agent.ask() with no positional input) — the
                # forward mapper also skips empty input buffers, so symmetry
                # requires we skip them here.
                continue
            messages.append(_user_message_from_request(event, serializer))
        elif isinstance(event, ModelResponse):
            messages.append(_assistant_message_from_response(event))
        elif isinstance(event, ToolResultsEvent):
            for result in event.results:
                if result.parent_id in seen_tool_result_ids:
                    continue
                seen_tool_result_ids.add(result.parent_id)
                messages.append(_tool_message_from_result(result, serializer))
        elif isinstance(event, ToolResultEvent):
            if event.parent_id in seen_tool_result_ids:
                continue
            seen_tool_result_ids.add(event.parent_id)
            messages.append(_tool_message_from_result(event, serializer))
    return messages


def _user_message_from_request(request: ModelRequest, serializer: SerializerProto) -> UserMessage:
    if len(request.parts) == 1 and isinstance(request.parts[0], TextInput):
        return UserMessage(id=str(uuid4()), content=request.parts[0].content)

    content: list[Any] = []
    for part in request.parts:
        converted = _part_to_content(part, serializer)
        if converted is not None:
            content.append(converted)

    if not content:
        return UserMessage(id=str(uuid4()), content="")

    # If every part flattens to plain text, collapse into a single string for
    # interoperability with clients that don't render structured content.
    if all(isinstance(c, TextInputContent) for c in content):
        return UserMessage(id=str(uuid4()), content="\n".join(c.text for c in content))

    return UserMessage(id=str(uuid4()), content=content)


def _part_to_content(part: BaseEvent, serializer: SerializerProto) -> Any:
    if isinstance(part, TextInput):
        return TextInputContent(text=part.content)

    if isinstance(part, UrlInput):
        source = InputContentUrlSource(value=part.url)
        return _content_for_kind(part.kind, source, fallback_mime=None)

    if isinstance(part, BinaryInput):
        source = InputContentDataSource(
            value=b64encode(part.data).decode("ascii"),
            mime_type=str(part.media_type),
        )
        return _content_for_kind(part.kind, source, fallback_mime=str(part.media_type))

    if isinstance(part, FileIdInput):
        return BinaryInputContent(
            mime_type="application/octet-stream",
            id=part.file_id,
            filename=part.filename,
        )

    if isinstance(part, DataInput):
        return TextInputContent(text=serializer.encode(part.data).decode())

    return None


def _content_for_kind(
    kind: BinaryType,
    source: InputContentUrlSource | InputContentDataSource,
    fallback_mime: str | None,
) -> Any:
    if kind == BinaryType.IMAGE:
        return ImageInputContent(source=source)
    if kind == BinaryType.AUDIO:
        return AudioInputContent(source=source)
    if kind == BinaryType.VIDEO:
        return VideoInputContent(source=source)
    if kind == BinaryType.DOCUMENT:
        return DocumentInputContent(source=source)
    # BinaryType.BINARY: no structured InputContent variant, fall through to
    # BinaryInputContent which carries the raw payload.
    if isinstance(source, InputContentUrlSource):
        return BinaryInputContent(
            mime_type=fallback_mime or source.mime_type or "application/octet-stream",
            url=source.value,
        )
    return BinaryInputContent(
        mime_type=fallback_mime or source.mime_type,
        data=source.value,
    )


def _assistant_message_from_response(response: ModelResponse) -> AssistantMessage:
    tool_calls = [
        ToolCall(
            id=call.id,
            function=FunctionCall(name=call.name, arguments=call.arguments),
        )
        for call in response.tool_calls.calls
    ]
    return AssistantMessage(
        id=str(uuid4()),
        content=response.content,
        tool_calls=tool_calls or None,
    )


def _tool_message_from_result(result: ToolResultEvent, serializer: SerializerProto) -> ToolMessage:
    content = _stringify_result(result.result, serializer)
    error = str(result.error) if isinstance(result, ToolErrorEvent) else None
    return ToolMessage(
        id=str(uuid4()),
        tool_call_id=result.parent_id,
        content=content,
        error=error,
    )


def _stringify_result(result: ToolResult, serializer: SerializerProto) -> str:
    chunks: list[str] = []
    for part in result.parts:
        if isinstance(part, TextInput):
            chunks.append(part.content)
        elif isinstance(part, DataInput):
            chunks.append(serializer.encode(part.data).decode())
        elif isinstance(part, UrlInput):
            chunks.append(part.url)
        elif isinstance(part, FileIdInput):
            chunks.append(f"[file:{part.file_id}]")
        elif isinstance(part, BinaryInput):
            chunks.append(f"[binary:{part.media_type} {len(part.data)}B]")
        else:
            chunks.append(repr(part))
    if len(chunks) == 1:
        return chunks[0]
    return "\n".join(chunks)


__all__ = ("events_to_agui_messages",)
