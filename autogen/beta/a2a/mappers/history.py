# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
from collections.abc import Mapping, Sequence
from typing import Any

from autogen.beta.events import (
    BaseEvent,
    BinaryInput,
    BinaryType,
    DataInput,
    FileIdInput,
    Input,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolErrorEvent,
    ToolResult,
    ToolResultEvent,
    UrlInput,
)

# Discriminant kinds for the ``ag2.history+json`` payload. Each event in the
# sequence carries a ``kind`` field; everything else is kind-specific.
KIND_USER_INPUT = "user_input"
KIND_TOOL_CALL = "tool_call"
KIND_TOOL_RESULT = "tool_result"
KIND_AGENT_MESSAGE = "agent_message"
KIND_MODEL_RESPONSE = "model_response"


def events_to_payload(events: Sequence[BaseEvent]) -> dict[str, Any]:
    """Serialize a sequence of AG2 events into the ``ag2.history+json`` shape.

    Transient events (``ModelMessageChunk``, ``ModelReasoning``, lifecycle
    events) are dropped — they are conversation deltas, not durable state.
    Unknown event types are also skipped silently so future event additions
    don't break older clients/servers.
    """
    out: list[dict[str, Any]] = []
    for ev in events:
        encoded = _event_to_dict(ev)
        if encoded is not None:
            out.append(encoded)
    return {"events": out}


def payload_to_events(payload: Mapping[str, Any]) -> list[BaseEvent]:
    """Reconstruct a list of AG2 events from an ``ag2.history+json`` payload.

    Skips entries with unknown ``kind`` to stay forward-compatible with
    payloads produced by newer code paths.
    """
    raw = payload.get("events") or []
    out: list[BaseEvent] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        ev = _dict_to_event(entry)
        if ev is not None:
            out.append(ev)
    return out


def _event_to_dict(ev: BaseEvent) -> dict[str, Any] | None:
    if isinstance(ev, ModelRequest):
        return {
            "kind": KIND_USER_INPUT,
            "parts": [_input_to_dict(p) for p in ev.parts],
        }
    # ToolErrorEvent is a subclass of ToolResultEvent — check it first so the
    # ``error`` field is captured.
    if isinstance(ev, ToolErrorEvent):
        return {
            "kind": KIND_TOOL_RESULT,
            "parent_id": ev.parent_id,
            "name": ev.name,
            "content": _tool_result_to_text(ev.result),
            "error": str(ev.error),
        }
    if isinstance(ev, ToolResultEvent):
        return {
            "kind": KIND_TOOL_RESULT,
            "parent_id": ev.parent_id,
            "name": ev.name,
            "content": _tool_result_to_text(ev.result),
            "error": None,
        }
    if isinstance(ev, ToolCallEvent):
        return {
            "kind": KIND_TOOL_CALL,
            "id": ev.id,
            "name": ev.name,
            "arguments": ev.arguments,
        }
    if isinstance(ev, ModelResponse):
        return {
            "kind": KIND_MODEL_RESPONSE,
            "content": ev.message.content if ev.message else "",
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in ev.tool_calls.calls],
        }
    if isinstance(ev, ModelMessage):
        return {
            "kind": KIND_AGENT_MESSAGE,
            "content": ev.content,
        }
    return None


def _dict_to_event(entry: Mapping[str, Any]) -> BaseEvent | None:
    kind = entry.get("kind")
    if kind == KIND_USER_INPUT:
        parts = entry.get("parts") or []
        inputs = [_dict_to_input(p) for p in parts if isinstance(p, Mapping)]
        return ModelRequest([inp for inp in inputs if inp is not None])
    if kind == KIND_TOOL_CALL:
        return ToolCallEvent(
            id=str(entry.get("id", "")),
            name=str(entry.get("name", "")),
            arguments=str(entry.get("arguments", "{}")),
        )
    if kind == KIND_TOOL_RESULT:
        content = str(entry.get("content", "") or "")
        result = ToolResult(content)
        error = entry.get("error")
        if error:
            return ToolErrorEvent(
                parent_id=str(entry.get("parent_id", "")),
                name=entry.get("name"),
                error=_RehydratedToolError(error),
                result=result,
            )
        return ToolResultEvent(
            parent_id=str(entry.get("parent_id", "")),
            name=entry.get("name"),
            result=result,
        )
    if kind == KIND_MODEL_RESPONSE:
        calls = entry.get("tool_calls") or []
        tool_calls = [
            ToolCallEvent(
                id=str(c.get("id", "")),
                name=str(c.get("name", "")),
                arguments=str(c.get("arguments", "{}")),
            )
            for c in calls
            if isinstance(c, Mapping)
        ]
        message_text = str(entry.get("content", "") or "")
        return ModelResponse(
            message=ModelMessage(message_text) if message_text else None,
            tool_calls=ToolCallsEvent(tool_calls),
        )
    if kind == KIND_AGENT_MESSAGE:
        return ModelMessage(str(entry.get("content", "") or ""))
    return None


class _RehydratedToolError(Exception):
    """Placeholder error type for ``ToolErrorEvent`` reconstructed from history.

    The original exception type is lost in transit — we only carry the
    rendered string. Subclassing ``Exception`` keeps ``ToolErrorEvent``'s
    invariants intact (e.g. ``str(ev.error)`` round-trips) without pretending
    we have the real type.
    """


_BINARY_PART = "binary"
_URL_PART = "url"
_FILE_ID_PART = "file_id"
_TEXT_PART = "text"
_DATA_PART = "data"


def _input_to_dict(inp: Input) -> dict[str, Any]:
    if isinstance(inp, TextInput):
        return {"type": _TEXT_PART, "text": inp.content}
    if isinstance(inp, BinaryInput):
        return {
            "type": _BINARY_PART,
            "media_type": str(inp.media_type),
            "data": base64.b64encode(inp.data).decode("ascii"),
            "kind": inp.kind.value,
            "vendor_metadata": dict(inp.vendor_metadata),
        }
    if isinstance(inp, UrlInput):
        return {"type": _URL_PART, "url": inp.url, "kind": inp.kind.value}
    if isinstance(inp, FileIdInput):
        return {"type": _FILE_ID_PART, "file_id": inp.file_id, "filename": inp.filename}
    if isinstance(inp, DataInput):
        return {"type": _DATA_PART, "data": inp.data}
    return {"type": _TEXT_PART, "text": str(inp)}


def _dict_to_input(entry: Mapping[str, Any]) -> Input | None:
    kind = entry.get("type")
    if kind == _TEXT_PART:
        return TextInput(str(entry.get("text", "")))
    if kind == _BINARY_PART:
        raw = entry.get("data", "")
        try:
            data = base64.b64decode(raw) if isinstance(raw, str) else b""
        except (ValueError, TypeError):
            return None
        return BinaryInput(
            data,
            media_type=str(entry.get("media_type", "application/octet-stream")),
            vendor_metadata=dict(entry.get("vendor_metadata") or {}),
            kind=_binary_kind_or_default(entry.get("kind")),
        )
    if kind == _URL_PART:
        return UrlInput(
            str(entry.get("url", "")),
            kind=_binary_kind_or_default(entry.get("kind")),
        )
    if kind == _FILE_ID_PART:
        filename = entry.get("filename")
        return FileIdInput(
            str(entry.get("file_id", "")),
            filename=filename if filename else None,
        )
    if kind == _DATA_PART:
        return DataInput(entry.get("data"))
    return None


def _binary_kind_or_default(value: Any) -> BinaryType:
    if isinstance(value, str):
        try:
            return BinaryType(value)
        except ValueError:
            return BinaryType.BINARY
    return BinaryType.BINARY


def _tool_result_to_text(result: ToolResult) -> str:
    chunks: list[str] = []
    for part in result.parts:
        if isinstance(part, TextInput):
            chunks.append(part.content)
        else:
            chunks.append(str(part))
    return "".join(chunks)
