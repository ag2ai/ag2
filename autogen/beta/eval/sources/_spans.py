# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Span → Trace adapter — reconstruct a :class:`Trace` from captured spans.

The trace-based evaluator grades a :class:`~autogen.beta.eval.Trace`, but the
trace can originate from a stored OpenTelemetry span tree rather than a live
in-memory event stream. This module is the bridge: it takes a normalized list
of :class:`SpanData` (the AG2 telemetry vocabulary, see
``autogen.beta._telemetry_consts``) and reconstructs the typed events scorers
filter on.

It is **pure**: importing it never pulls in the OpenTelemetry SDK. Backends
that read spans from the SDK (in-memory exporter) or from disk/cloud (JSON)
each convert their source into :class:`SpanData` and call :func:`spans_to_trace`.

Coverage today mirrors what ``TelemetryMiddleware`` emits: ``llm`` →
:class:`ModelResponse` (with :class:`Usage`), ``tool`` → :class:`ToolCallEvent`
plus a :class:`ToolResultEvent` / :class:`ToolErrorEvent`, ``human_input`` →
:class:`HumanInputRequest` / :class:`HumanMessage`, and the root ``agent`` span
for wall-clock duration **and ``trace.exception``** (reconstructed when the turn
raised — see :func:`_agent_span_exception`).

Dormant on the OTEL path (so ``failure_attribution``'s halt/loop and
tool-not-found branches don't fire when grading reconstructed traces): ``HaltEvent``
and ``ToolNotFoundEvent`` are **stream-only** AG2 events — emitted by the
alert/policy layer and the tool executor's fallback subscriber, *not* through any
``TelemetryMiddleware`` hook — so they never become spans. Closing this would mean
expanding ``TelemetryMiddleware`` to *subscribe* to the stream for those events
(idiomatic — mirrors ``_HaltCheckMiddleware``, a middleware that already subscribes
to ``HaltEvent``). Deferred: niche, AG2-specific, irrelevant to non-AG2 producers.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from autogen.beta._telemetry_consts import (
    ATTR_HUMAN_INPUT_PROMPT,
    ATTR_HUMAN_INPUT_RESPONSE,
    ATTR_SPAN_TYPE,
    SPAN_TYPE_AGENT,
    SPAN_TYPE_HUMAN_INPUT,
    SPAN_TYPE_LLM,
    SPAN_TYPE_TOOL,
)
from autogen.beta.events import (
    BaseEvent,
    HumanInputRequest,
    HumanMessage,
    ModelMessage,
    ModelResponse,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
    Usage,
)

from ..trace import Trace

__all__ = (
    "SpanData",
    "SpanEvent",
    "span_data_from_dict",
    "span_data_to_dict",
    "spans_to_trace",
)


# gen_ai semantic-convention attribute keys. ``TelemetryMiddleware`` emits these
# inline (single producer site), so they are not in ``_telemetry_consts``; the
# adapter mirrors them here. Stable OTel GenAI semconv names.
_ATTR_USAGE_INPUT = "gen_ai.usage.input_tokens"
_ATTR_USAGE_OUTPUT = "gen_ai.usage.output_tokens"
_ATTR_USAGE_CACHE_CREATE = "gen_ai.usage.cache_creation_input_tokens"
_ATTR_USAGE_CACHE_READ = "gen_ai.usage.cache_read_input_tokens"
_ATTR_USAGE_THINKING = "gen_ai.usage.thinking_tokens"
_ATTR_OUTPUT_MESSAGES = "gen_ai.output.messages"
_ATTR_RESPONSE_MODEL = "gen_ai.response.model"
_ATTR_REQUEST_MODEL = "gen_ai.request.model"
_ATTR_PROVIDER = "gen_ai.provider.name"
_ATTR_FINISH_REASONS = "gen_ai.response.finish_reasons"
_ATTR_TOOL_NAME = "gen_ai.tool.name"
_ATTR_TOOL_CALL_ID = "gen_ai.tool.call.id"
_ATTR_TOOL_ARGS = "gen_ai.tool.call.arguments"
_ATTR_TOOL_RESULT = "gen_ai.tool.call.result"

# OTel records exceptions as a span event named "exception" with these attrs.
_EXCEPTION_EVENT = "exception"
_ATTR_EXC_MESSAGE = "exception.message"

_NS_PER_MS = 1_000_000


@dataclass(frozen=True, slots=True)
class SpanEvent:
    """A point-in-time event recorded on a span (e.g. a recorded exception)."""

    name: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SpanData:
    """Normalized, SDK-free view of one span.

    Backends populate this from their source (the OTel in-memory exporter, an
    on-disk JSON span, or a cloud query result) so :func:`spans_to_trace` never
    depends on any particular span representation.

    Times are nanoseconds since the epoch (OTel's native unit). ``status`` is
    ``"OK"`` / ``"ERROR"`` / ``"UNSET"``.
    """

    name: str
    span_id: str
    parent_id: str | None
    start_ns: int
    end_ns: int
    attributes: Mapping[str, Any] = field(default_factory=dict)
    status: str = "UNSET"
    events: tuple[SpanEvent, ...] = ()


def spans_to_trace(spans: Sequence[SpanData], *, duration_ms: int | None = None) -> Trace:
    """Reconstruct a :class:`Trace` from a list of captured spans.

    Spans are ordered by start time and mapped to the typed events scorers
    consume. ``duration_ms`` defaults to the root ``agent`` span's wall-clock
    span; pass an explicit value to override (e.g. the producer's measured
    duration around ``ask``).
    """
    ordered = sorted(spans, key=lambda s: s.start_ns)

    events: list[BaseEvent] = []
    for span in ordered:
        span_type = span.attributes.get(ATTR_SPAN_TYPE)
        if span_type == SPAN_TYPE_LLM:
            events.append(_llm_span_to_response(span))
        elif span_type == SPAN_TYPE_TOOL:
            events.extend(_tool_span_to_events(span))
        elif span_type == SPAN_TYPE_HUMAN_INPUT:
            events.extend(_human_span_to_events(span))

    resolved_duration = duration_ms if duration_ms is not None else _root_duration_ms(ordered)
    return Trace(events=events, reply=None, exception=_agent_span_exception(ordered), duration_ms=resolved_duration)


def _llm_span_to_response(span: SpanData) -> ModelResponse:
    a = span.attributes
    usage = Usage(
        prompt_tokens=a.get(_ATTR_USAGE_INPUT),
        completion_tokens=a.get(_ATTR_USAGE_OUTPUT),
        cache_creation_input_tokens=a.get(_ATTR_USAGE_CACHE_CREATE),
        cache_read_input_tokens=a.get(_ATTR_USAGE_CACHE_READ),
        thinking_tokens=a.get(_ATTR_USAGE_THINKING),
    )
    return ModelResponse(
        message=_message_from_output(a.get(_ATTR_OUTPUT_MESSAGES)),
        usage=usage,
        model=a.get(_ATTR_RESPONSE_MODEL) or a.get(_ATTR_REQUEST_MODEL),
        provider=a.get(_ATTR_PROVIDER),
        finish_reason=_first_finish_reason(a.get(_ATTR_FINISH_REASONS)),
    )


def _message_from_output(raw: Any) -> ModelMessage | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        messages = json.loads(raw)
    except ValueError:
        return None
    if not messages or not isinstance(messages[0], dict):
        return None
    content = messages[0].get("content")
    return ModelMessage(content) if isinstance(content, str) else None


def _first_finish_reason(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    return None


def _tool_span_to_events(span: SpanData) -> list[BaseEvent]:
    a = span.attributes
    name = a.get(_ATTR_TOOL_NAME, "")
    call_id = a.get(_ATTR_TOOL_CALL_ID)
    arguments = a.get(_ATTR_TOOL_ARGS, "{}")
    call = (
        ToolCallEvent(name, id=call_id, arguments=arguments)
        if call_id is not None
        else ToolCallEvent(name, arguments=arguments)
    )

    if span.status == "ERROR":
        return [call, ToolErrorEvent.from_call(call, _exception_from_span(span))]

    result = a.get(_ATTR_TOOL_RESULT)
    return [call, ToolResultEvent.from_call(call, result if result is not None else "")]


def _exception_from_span(span: SpanData) -> Exception:
    for event in span.events:
        if event.name == _EXCEPTION_EVENT:
            return RuntimeError(str(event.attributes.get(_ATTR_EXC_MESSAGE, "")))
    return RuntimeError("")


def _agent_span_exception(spans: Sequence[SpanData]) -> Exception | None:
    """Reconstruct a top-level ``trace.exception`` from the root agent span if it errored.

    ``TelemetryMiddleware.on_turn`` records the exception on the ``invoke_agent`` span
    (``record_exception`` + ``ERROR`` status) when a turn raises. A handled tool error
    leaves the agent span ``OK`` (surfaced only as a ``ToolErrorEvent``), so this fires
    only when the run actually crashed — matching live ``trace.exception`` semantics.
    """
    roots = [s for s in spans if s.parent_id is None and s.attributes.get(ATTR_SPAN_TYPE) == SPAN_TYPE_AGENT]
    if not roots:
        return None
    root = min(roots, key=lambda s: s.start_ns)
    if root.status == "ERROR" or any(e.name == _EXCEPTION_EVENT for e in root.events):
        return _exception_from_span(root)
    return None


def _human_span_to_events(span: SpanData) -> list[BaseEvent]:
    a = span.attributes
    out: list[BaseEvent] = []
    prompt = a.get(ATTR_HUMAN_INPUT_PROMPT)
    if isinstance(prompt, str):
        out.append(HumanInputRequest(prompt))
    response = a.get(ATTR_HUMAN_INPUT_RESPONSE)
    if isinstance(response, str):
        out.append(HumanMessage(response))
    return out


def _root_duration_ms(spans: Sequence[SpanData]) -> int:
    if not spans:
        return 0
    roots = [s for s in spans if s.parent_id is None and s.attributes.get(ATTR_SPAN_TYPE) == SPAN_TYPE_AGENT]
    if not roots:
        roots = [s for s in spans if s.parent_id is None] or list(spans)
    root = min(roots, key=lambda s: s.start_ns)
    return max(0, (root.end_ns - root.start_ns) // _NS_PER_MS)


def span_data_to_dict(span: SpanData) -> dict[str, Any]:
    """Serialize a :class:`SpanData` to a JSON-safe dict (provisional disk format)."""
    return {
        "name": span.name,
        "span_id": span.span_id,
        "parent_id": span.parent_id,
        "start_ns": span.start_ns,
        "end_ns": span.end_ns,
        "attributes": dict(span.attributes),
        "status": span.status,
        "events": [{"name": e.name, "attributes": dict(e.attributes)} for e in span.events],
    }


def span_data_from_dict(data: dict[str, Any]) -> SpanData:
    """Rebuild a :class:`SpanData` from a dict produced by :func:`span_data_to_dict`."""
    return SpanData(
        name=data.get("name", ""),
        span_id=data.get("span_id", ""),
        parent_id=data.get("parent_id"),
        start_ns=int(data.get("start_ns", 0)),
        end_ns=int(data.get("end_ns", 0)),
        attributes=dict(data.get("attributes", {})),
        status=data.get("status", "UNSET"),
        events=tuple(SpanEvent(e.get("name", ""), dict(e.get("attributes", {}))) for e in data.get("events", [])),
    )
