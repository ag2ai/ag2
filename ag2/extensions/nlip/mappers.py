# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from nlip_sdk.nlip import NLIP_Factory, NLIP_Message

from ag2.events import (
    BaseEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolErrorEvent,
    ToolResult,
    ToolResultEvent,
    ToolResultsEvent,
)
from ag2.tools.final.function_tool import FunctionDefinition, FunctionToolSchema

from .errors import RehydratedNlipToolError

LABEL_HISTORY = "ag2_chat_history"
LABEL_CONTEXT = "ag2_context"
LABEL_CLIENT_TOOLS = "ag2_client_tools"
LABEL_TOOL_RESULTS = "ag2_tool_results"
LABEL_TOOL_CALLS = "ag2_tool_calls"
ERROR_CODE_INPUT_REQUIRED = "INPUT_REQUIRED"

# Discriminant kinds for the chat-history payload carried in the
# ``ag2_chat_history`` submessage.
_KIND_USER_INPUT = "user_input"
_KIND_TOOL_CALL = "tool_call"
_KIND_TOOL_CALLS = "tool_calls"
_KIND_TOOL_RESULT = "tool_result"
_KIND_TOOL_RESULTS = "tool_results"
_KIND_AGENT_MESSAGE = "agent_message"
_KIND_MODEL_RESPONSE = "model_response"


@dataclass(slots=True)
class ParsedNlipRequest:
    """Decoded inbound NLIP request, normalized into AG2-shaped buckets."""

    text: str = ""
    history_events: list[BaseEvent] = field(default_factory=list)
    context_update: dict[str, Any] = field(default_factory=dict)
    client_tools: list[FunctionToolSchema] = field(default_factory=list)
    tool_results: list[ToolResultEvent] = field(default_factory=list)


def build_request_message(
    text: str,
    *,
    history_events: Sequence[BaseEvent] = (),
    context: dict[str, Any] | None = None,
    tool_schemas: Sequence[FunctionToolSchema] = (),
    tool_results: Sequence[ToolResultEvent] = (),
) -> NLIP_Message:
    """Build an outgoing NLIP request from an AG2 client.

    NLIP sessions are stateless, so every request carries the full
    conversation history as a JSON submessage (``ag2_chat_history``) in
    addition to the plain-text top-level content — non-AG2 NLIP servers
    can ignore the submessage and just answer the text.
    """
    msg = NLIP_Factory.create_text(text, language="english")
    if history_events:
        msg.add_json(events_to_payload(history_events), label=LABEL_HISTORY)
    if context:
        msg.add_json(context, label=LABEL_CONTEXT)
    if tool_schemas:
        msg.add_json(schemas_to_payload(tool_schemas), label=LABEL_CLIENT_TOOLS)
    if tool_results:
        msg.add_json(results_to_payload(tool_results), label=LABEL_TOOL_RESULTS)
    return msg


def parse_request_message(msg: NLIP_Message) -> ParsedNlipRequest:
    """Decode an incoming NLIP request into AG2-shaped buckets.

    Prefers the structured ``ag2_chat_history`` / ``ag2_context`` /
    ``ag2_client_tools`` / ``ag2_tool_results`` submessages produced by
    :func:`build_request_message`; falls back to the bare top-level text
    when talking to a plain (non-AG2) NLIP client — e.g. a request sent via
    curl with no submessages at all is still usable as a single-turn query.
    """
    parsed = ParsedNlipRequest(text=msg.extract_text() or "")

    history_submsg = msg.find_labeled_submessage(LABEL_HISTORY) if msg.submessages else None
    if history_submsg is not None and isinstance(history_submsg.content, dict):
        parsed.history_events = payload_to_events(history_submsg.content)

    context_submsg = msg.find_labeled_submessage(LABEL_CONTEXT) if msg.submessages else None
    if context_submsg is not None and isinstance(context_submsg.content, dict):
        parsed.context_update = context_submsg.content

    tools_submsg = msg.find_labeled_submessage(LABEL_CLIENT_TOOLS) if msg.submessages else None
    if tools_submsg is not None and isinstance(tools_submsg.content, dict):
        parsed.client_tools = payload_to_schemas(tools_submsg.content)

    results_submsg = msg.find_labeled_submessage(LABEL_TOOL_RESULTS) if msg.submessages else None
    if results_submsg is not None and isinstance(results_submsg.content, dict):
        parsed.tool_results = payload_to_results(results_submsg.content)

    return parsed


def build_response_message(
    text: str,
    *,
    context_update: dict[str, Any] | None = None,
    tool_calls: Sequence[ToolCallEvent] = (),
    input_required: str | None = None,
) -> NLIP_Message:
    """Build an outgoing NLIP response from an AG2-backed server.

    Plain text is the top-level content so any NLIP-conformant client can
    consume it directly — no ``role: `` prefix, since NLIP sessions are
    stateless and a non-AG2 client has no use for it. ``tool_calls`` are
    attached as a JSON submessage (``ag2_tool_calls``) for AG2 clients that
    round-trip client-side tool execution; non-AG2 clients ignore it.
    ``input_required`` attaches an ``error/text`` submessage carrying the
    human-input prompt.
    """
    msg = NLIP_Factory.create_text(text, language="english")
    if context_update:
        msg.add_json(context_update, label=LABEL_CONTEXT)
    if tool_calls:
        msg.add_json(calls_to_payload(tool_calls), label=LABEL_TOOL_CALLS)
    if input_required:
        msg.add_error_code(input_required, label=ERROR_CODE_INPUT_REQUIRED)
    return msg


@dataclass(slots=True)
class ParsedNlipResponse:
    """Decoded NLIP response, as seen by an AG2 client calling a remote server."""

    text: str = ""
    context_update: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[ToolCallEvent] = field(default_factory=list)
    input_required: str | None = None


def parse_response_message(msg: NLIP_Message) -> ParsedNlipResponse:
    """Decode a NLIP response received from a remote server."""
    parsed = ParsedNlipResponse(text=msg.extract_text() or "")

    context_submsg = msg.find_labeled_submessage(LABEL_CONTEXT) if msg.submessages else None
    if context_submsg is not None and isinstance(context_submsg.content, dict):
        parsed.context_update = context_submsg.content

    calls_submsg = msg.find_labeled_submessage(LABEL_TOOL_CALLS) if msg.submessages else None
    if calls_submsg is not None and isinstance(calls_submsg.content, dict):
        parsed.tool_calls = payload_to_calls(calls_submsg.content)

    if msg.submessages:
        for submsg in msg.submessages:
            if submsg.format == "error" and submsg.label == ERROR_CODE_INPUT_REQUIRED:
                parsed.input_required = str(submsg.content)
                break

    return parsed


# --- tool schema / call / result payloads -----------------------------------


def schemas_to_payload(schemas: Iterable[FunctionToolSchema]) -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": s.function.name,
                "description": s.function.description,
                "parameters": s.function.parameters,
            }
            for s in schemas
        ]
    }


def payload_to_schemas(payload: Mapping[str, Any]) -> list[FunctionToolSchema]:
    return [
        FunctionToolSchema(
            function=FunctionDefinition(
                name=t["name"],
                description=t.get("description", "") or "",
                parameters=t.get("parameters", {}) or {},
            )
        )
        for t in payload.get("tools", [])
    ]


def call_to_payload(call: ToolCallEvent) -> dict[str, Any]:
    return {"id": call.id, "name": call.name, "arguments": call.arguments}


def payload_to_call(payload: Mapping[str, Any]) -> ToolCallEvent:
    return ToolCallEvent(
        id=str(payload["id"]),
        name=str(payload["name"]),
        arguments=str(payload.get("arguments", "{}")),
    )


def calls_to_payload(calls: Iterable[ToolCallEvent]) -> dict[str, Any]:
    return {"calls": [call_to_payload(c) for c in calls]}


def payload_to_calls(payload: Mapping[str, Any]) -> list[ToolCallEvent]:
    return [payload_to_call(c) for c in payload.get("calls", []) if isinstance(c, Mapping)]


def _tool_result_to_text(result: ToolResult) -> str:
    return "".join(part.content if isinstance(part, TextInput) else str(part) for part in result.parts)


def result_event_to_payload(ev: ToolResultEvent) -> dict[str, Any]:
    return {
        "id": ev.parent_id,
        "name": ev.name,
        "content": _tool_result_to_text(ev.result),
        "error": str(ev.error) if isinstance(ev, ToolErrorEvent) else None,
    }


def payload_to_result_event(entry: Mapping[str, Any]) -> ToolResultEvent:
    parent_id = str(entry.get("parent_id") or entry.get("id") or "")
    name = entry.get("name")
    content = str(entry.get("content", "") or "")
    result = ToolResult(content)
    error = entry.get("error")
    if error:
        return ToolErrorEvent(parent_id=parent_id, name=name, error=RehydratedNlipToolError(error), result=result)
    return ToolResultEvent(parent_id=parent_id, name=name, result=result)


def results_to_payload(results: Iterable[ToolResultEvent]) -> dict[str, Any]:
    return {"results": [result_event_to_payload(r) for r in results]}


def payload_to_results(payload: Mapping[str, Any]) -> list[ToolResultEvent]:
    return [payload_to_result_event(r) for r in payload.get("results", []) if isinstance(r, Mapping)]


# --- chat history (de)serialization -----------------------------------------


def events_to_payload(events: Sequence[BaseEvent]) -> dict[str, Any]:
    """Serialize AG2 events into the ``ag2_chat_history`` wire shape.

    Transient events (``ModelMessageChunk``, lifecycle events, etc.) are
    dropped — they are conversation deltas, not durable state. Unknown
    event types are skipped silently so future event additions don't break
    older clients/servers.
    """
    out: list[dict[str, Any]] = []
    for ev in events:
        encoded = _event_to_dict(ev)
        if encoded is not None:
            out.append(encoded)
    return {"events": out}


def payload_to_events(payload: Mapping[str, Any]) -> list[BaseEvent]:
    """Reconstruct AG2 events from an ``ag2_chat_history`` payload."""
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
        return {"kind": _KIND_USER_INPUT, "parts": [_input_to_dict(p) for p in ev.parts]}
    if isinstance(ev, ToolResultsEvent):
        return {"kind": _KIND_TOOL_RESULTS, "results": [result_event_to_payload(r) for r in ev.results]}
    if isinstance(ev, ToolCallsEvent):
        return {"kind": _KIND_TOOL_CALLS, "calls": [call_to_payload(c) for c in ev.calls]}
    if isinstance(ev, ToolErrorEvent):
        return result_event_to_payload(ev) | {"kind": _KIND_TOOL_RESULT}
    if isinstance(ev, ToolResultEvent):
        return result_event_to_payload(ev) | {"kind": _KIND_TOOL_RESULT}
    if isinstance(ev, ToolCallEvent):
        return {"kind": _KIND_TOOL_CALL, **call_to_payload(ev)}
    if isinstance(ev, ModelResponse):
        return {
            "kind": _KIND_MODEL_RESPONSE,
            "content": ev.message.content if ev.message else "",
            "tool_calls": [call_to_payload(c) for c in ev.tool_calls.calls],
        }
    if isinstance(ev, ModelMessage):
        return {"kind": _KIND_AGENT_MESSAGE, "content": ev.content}
    return None


def _dict_to_event(entry: Mapping[str, Any]) -> BaseEvent | None:
    kind = entry.get("kind")
    if kind == _KIND_USER_INPUT:
        parts = entry.get("parts") or []
        inputs = [TextInput(str(p.get("text", ""))) for p in parts if isinstance(p, Mapping)]
        return ModelRequest(inputs)
    if kind == _KIND_TOOL_CALL:
        return payload_to_call(entry)
    if kind == _KIND_TOOL_CALLS:
        raw_calls = entry.get("calls") or []
        return ToolCallsEvent([payload_to_call(c) for c in raw_calls if isinstance(c, Mapping)])
    if kind == _KIND_TOOL_RESULT:
        return payload_to_result_event(entry)
    if kind == _KIND_TOOL_RESULTS:
        raw_results = entry.get("results") or []
        return ToolResultsEvent([payload_to_result_event(r) for r in raw_results if isinstance(r, Mapping)])
    if kind == _KIND_MODEL_RESPONSE:
        raw_calls = entry.get("tool_calls") or []
        tool_calls = [payload_to_call(c) for c in raw_calls if isinstance(c, Mapping)]
        message_text = str(entry.get("content", "") or "")
        return ModelResponse(
            message=ModelMessage(message_text) if message_text else None,
            tool_calls=ToolCallsEvent(tool_calls),
        )
    if kind == _KIND_AGENT_MESSAGE:
        return ModelMessage(str(entry.get("content", "") or ""))
    return None


def _input_to_dict(inp: Any) -> dict[str, Any]:
    if isinstance(inp, TextInput):
        return {"text": inp.content}
    return {"text": str(inp)}


__all__ = [
    "LABEL_CLIENT_TOOLS",
    "LABEL_CONTEXT",
    "LABEL_HISTORY",
    "LABEL_TOOL_CALLS",
    "LABEL_TOOL_RESULTS",
    "ParsedNlipRequest",
    "ParsedNlipResponse",
    "build_request_message",
    "build_response_message",
    "events_to_payload",
    "parse_request_message",
    "parse_response_message",
    "payload_to_events",
]
