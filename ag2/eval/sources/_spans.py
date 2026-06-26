# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Span → Trace adapter — reconstruct a :class:`Trace` from captured spans.

The trace-based evaluator grades a :class:`~ag2.eval.Trace`, but the
trace can originate from a stored OpenTelemetry span tree rather than a live
in-memory event stream. This module is the bridge: it takes a normalized list
of :class:`SpanData` and reconstructs the typed events scorers filter on.

It is **pure**: importing it never pulls in the OpenTelemetry SDK. Backends
that read spans from the SDK (in-memory exporter) or from disk/cloud (JSON)
each convert their source into :class:`SpanData` and call :func:`spans_to_trace`.

**Span dialects.** Different producers tag spans differently. Each
:class:`SpanConvention` reads one dialect into the *same* AG2 events, and
:func:`spans_to_trace` auto-detects per span (first convention that recognizes a
span wins) — so a trace from any producer, or a mix, grades identically:

* :class:`AG2GenAIConvention` — AG2's own (``ag2.span.type`` + OTel ``gen_ai.*``,
  emitted by ``TelemetryMiddleware``): ``llm`` → :class:`ModelResponse` (with
  :class:`Usage`), ``tool`` → :class:`ToolCallEvent` + :class:`ToolResultEvent` /
  :class:`ToolErrorEvent`, ``human_input`` → :class:`HumanInputRequest` /
  :class:`HumanMessage`, and the root ``agent`` span for duration **and**
  ``trace.exception`` (when the turn raised).
* :class:`OpenInferenceConvention` — the OpenInference dialect (Arize/Phoenix
  instrumentors, including ``openinference-instrumentation-agno``):
  ``openinference.span.kind`` + ``llm.*`` / ``tool.*``.

A span recognized by no convention is skipped; if a whole trace reconstructs to
**zero** events, :func:`spans_to_trace` logs a warning (a likely unrecognized
dialect) rather than silently grading an empty trace.

Never reconstructed on the OTEL path: ``HaltEvent`` and ``ToolNotFoundEvent`` are
**stream-only** AG2 events — emitted outside the ``TelemetryMiddleware`` hooks —
so they never become spans. Eval therefore has no deterministic detectors for
them (the LLM attributor can still classify a loop / hallucinated tool
semantically). Closing this would mean expanding ``TelemetryMiddleware`` to
*subscribe* to the stream for them (mirrors ``_HaltCheckMiddleware``). Deferred:
niche, AG2-specific.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ag2._telemetry_consts import (
    ATTR_HUMAN_INPUT_PROMPT,
    ATTR_HUMAN_INPUT_RESPONSE,
    ATTR_SPAN_TYPE,
    SPAN_TYPE_AGENT,
    SPAN_TYPE_HUMAN_INPUT,
    SPAN_TYPE_LLM,
    SPAN_TYPE_TOOL,
)
from ag2.events import (
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
    "DEFAULT_CONVENTIONS",
    "AG2GenAIConvention",
    "OpenInferenceConvention",
    "SpanConvention",
    "SpanData",
    "SpanEvent",
    "span_data_from_dict",
    "span_data_to_dict",
    "spans_to_trace",
)

logger = logging.getLogger(__name__)

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

# OpenInference semantic-convention keys (Arize/Phoenix instrumentors). Span kind
# lives on ``openinference.span.kind``; message content is index-flattened.
_OI_SPAN_KIND = "openinference.span.kind"
_OI_KIND_AGENT = "AGENT"
_OI_KIND_LLM = "LLM"
_OI_KIND_TOOL = "TOOL"
_OI_OUTPUT_CONTENT = "llm.output_messages.0.message.content"
_OI_MODEL = "llm.model_name"
_OI_PROVIDER = "llm.provider"
_OI_TOKENS_PROMPT = "llm.token_count.prompt"
_OI_TOKENS_COMPLETION = "llm.token_count.completion"
_OI_TOOL_NAME = "tool.name"
_OI_TOOL_PARAMS = "tool.parameters"
_OI_INPUT_VALUE = "input.value"
_OI_OUTPUT_VALUE = "output.value"

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


@runtime_checkable
class SpanConvention(Protocol):
    """Reads one telemetry dialect off a :class:`SpanData` into AG2 typed events.

    Inspect the span's discriminator attribute and return the typed events it maps
    to — the dialect's *agent/root* span returns ``[]`` (recognized but event-free) —
    or ``None`` when the span isn't this dialect, so the next convention can try.
    Implement one (a single method) to make AG2 grade a new producer's traces.
    """

    def to_events(self, span: SpanData) -> list[BaseEvent] | None: ...


class AG2GenAIConvention:
    """AG2's own dialect: ``ag2.span.type`` + OTel ``gen_ai.*`` (emitted by ``TelemetryMiddleware``)."""

    def to_events(self, span: SpanData) -> list[BaseEvent] | None:
        kind = span.attributes.get(ATTR_SPAN_TYPE)
        if kind == SPAN_TYPE_AGENT:
            return []
        if kind == SPAN_TYPE_LLM:
            return [_llm_span_to_response(span)]
        if kind == SPAN_TYPE_TOOL:
            return _tool_span_to_events(span)
        if kind == SPAN_TYPE_HUMAN_INPUT:
            return _human_span_to_events(span)
        return None


class OpenInferenceConvention:
    """OpenInference dialect (Arize/Phoenix instrumentors, incl. Agno): ``openinference.span.kind`` + ``llm.*`` / ``tool.*``."""

    def to_events(self, span: SpanData) -> list[BaseEvent] | None:
        kind = span.attributes.get(_OI_SPAN_KIND)
        if kind == _OI_KIND_AGENT:
            return []
        if kind == _OI_KIND_LLM:
            return [_oi_llm_to_response(span)]
        if kind == _OI_KIND_TOOL:
            return _oi_tool_to_events(span)
        return None


DEFAULT_CONVENTIONS: tuple[SpanConvention, ...] = (AG2GenAIConvention(), OpenInferenceConvention())


def spans_to_trace(
    spans: Sequence[SpanData],
    *,
    conventions: Sequence[SpanConvention] | None = None,
    duration_ms: int | None = None,
) -> Trace:
    """Reconstruct a :class:`Trace` from captured spans.

    Spans are ordered by start time; each is mapped to typed events by the first
    ``conventions`` entry that recognizes it (default: AG2 ``gen_ai`` + OpenInference,
    auto-detected per span). ``duration_ms`` defaults to the root span's wall-clock;
    pass an explicit value to override (e.g. the producer's measured ``ask`` duration).
    """
    ordered = sorted(spans, key=lambda s: s.start_ns)
    active = DEFAULT_CONVENTIONS if conventions is None else conventions

    events: list[BaseEvent] = []
    for span in ordered:
        for convention in active:
            mapped = convention.to_events(span)
            if mapped is not None:
                events.extend(mapped)
                break

    if ordered and not events:
        logger.warning(
            "spans_to_trace reconstructed 0 events from %d span(s) — the producer's span dialect may be "
            "unrecognized by %s.",
            len(ordered),
            "/".join(type(c).__name__ for c in active) or "(no conventions)",
        )

    resolved_duration = duration_ms if duration_ms is not None else _root_duration_ms(ordered)
    return Trace(events=events, exception=_root_exception(ordered), duration_ms=resolved_duration)


# ── GenAI dialect readers (ag2.span.type + gen_ai.*) ────────────────────────
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


# ── OpenInference dialect readers (openinference.span.kind + llm.*/tool.*) ──
def _oi_llm_to_response(span: SpanData) -> ModelResponse:
    a = span.attributes
    content = a.get(_OI_OUTPUT_CONTENT)
    return ModelResponse(
        message=ModelMessage(content) if isinstance(content, str) else None,
        usage=Usage(prompt_tokens=a.get(_OI_TOKENS_PROMPT), completion_tokens=a.get(_OI_TOKENS_COMPLETION)),
        model=a.get(_OI_MODEL),
        provider=a.get(_OI_PROVIDER),
        finish_reason=None,
    )


def _oi_tool_to_events(span: SpanData) -> list[BaseEvent]:
    a = span.attributes
    name = a.get(_OI_TOOL_NAME, "")
    raw_args = a.get(_OI_TOOL_PARAMS)
    if raw_args is None:
        raw_args = a.get(_OI_INPUT_VALUE, "{}")
    arguments = raw_args if isinstance(raw_args, str) else json.dumps(raw_args)
    call = ToolCallEvent(name, arguments=arguments)

    if span.status == "ERROR":
        return [call, ToolErrorEvent.from_call(call, _exception_from_span(span))]

    result = a.get(_OI_OUTPUT_VALUE)
    return [call, ToolResultEvent.from_call(call, result if result is not None else "")]


# ── shared helpers ──────────────────────────────────────────────────────────
def _exception_from_span(span: SpanData) -> Exception:
    for event in span.events:
        if event.name == _EXCEPTION_EVENT:
            return RuntimeError(str(event.attributes.get(_ATTR_EXC_MESSAGE, "")))
    return RuntimeError("")


def _is_agent_span(span: SpanData) -> bool:
    """The dialect's root/agent span, across known conventions (for duration + exception)."""
    return (
        span.attributes.get(ATTR_SPAN_TYPE) == SPAN_TYPE_AGENT or span.attributes.get(_OI_SPAN_KIND) == _OI_KIND_AGENT
    )


def _root_span(spans: Sequence[SpanData]) -> SpanData | None:
    if not spans:
        return None
    roots = [s for s in spans if s.parent_id is None and _is_agent_span(s)]
    if not roots:
        roots = [s for s in spans if s.parent_id is None] or list(spans)
    return min(roots, key=lambda s: s.start_ns)


def _root_duration_ms(spans: Sequence[SpanData]) -> int:
    root = _root_span(spans)
    return max(0, (root.end_ns - root.start_ns) // _NS_PER_MS) if root is not None else 0


def _root_exception(spans: Sequence[SpanData]) -> Exception | None:
    """Reconstruct a top-level ``trace.exception`` from the root agent span if it errored.

    The producer records the exception on the root span (``record_exception`` + ``ERROR``
    status) when a turn raises. A *handled* tool error leaves the root ``OK`` (surfaced
    only as a ``ToolErrorEvent``), so this fires only when the run actually crashed —
    matching live ``trace.exception`` semantics.
    """
    root = _root_span(spans)
    if root is None:
        return None
    if root.status == "ERROR" or any(e.name == _EXCEPTION_EVENT for e in root.events):
        return _exception_from_span(root)
    return None


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
