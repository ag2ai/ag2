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

**Token accounting is asymmetric between the two, on purpose.** ``Trace.tokens``
reads :class:`UsageEvent`, so every dialect must produce one. AG2 records usage
as its *own* span (``TelemetryMiddleware`` subscribes to the event), which is the
only way spend that never becomes an LLM span — a sub-task rollup, history
compaction, memory aggregation, a live session — reaches a trace at all. A
foreign producer has none of that, so its per-call usage is synthesized from the
LLM span instead. AG2's LLM spans must therefore *not* be synthesized from, or
every main-loop call would be counted twice.

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
    ATTR_USAGE_KIND,
    ATTR_USAGE_LABEL,
    ATTR_USAGE_TOTAL,
    SPAN_TYPE_AGENT,
    SPAN_TYPE_HUMAN_INPUT,
    SPAN_TYPE_LLM,
    SPAN_TYPE_TOOL,
    SPAN_TYPE_USAGE,
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
    UsageEvent,
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
_ATTR_AGENT_NAME = "gen_ai.agent.name"
# ``TelemetryMiddleware`` writes this when the caller named no agent, so it
# identifies nobody and must not be matched against a rollup label.
_AGENT_NAME_UNSET = "unknown"

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
    """AG2's own dialect: ``ag2.span.type`` + OTel ``gen_ai.*`` (emitted by ``TelemetryMiddleware``).

    ``synthesize_usage_from_llm_spans`` is the back-compatibility switch for
    traces captured before AG2 recorded usage spans: those carry token counts on
    their LLM spans only, and would otherwise report zero once ``Trace.tokens``
    reads the accounting event. Whether a trace needs it is a fact about the
    trace, so :func:`spans_to_trace` decides it — turning it on exactly when a
    trace contains no usage span, and re-creating an instance handed to its
    ``conventions=`` argument with that answer. Whatever this is constructed
    with therefore only matters when :meth:`to_events` is driven directly.

    Stateless and safe to reuse across traces — the shared
    :data:`DEFAULT_CONVENTIONS` presents it that way. Facts that need the whole
    span tree, such as cancelling a rollup an instrumented sub-agent already
    accounts for, are settled by :func:`spans_to_trace` after the per-span pass
    rather than by carrying state through it.
    """

    def __init__(self, *, synthesize_usage_from_llm_spans: bool = False) -> None:
        self._synthesize_usage = synthesize_usage_from_llm_spans

    def to_events(self, span: SpanData) -> list[BaseEvent] | None:
        kind = span.attributes.get(ATTR_SPAN_TYPE)
        if kind == SPAN_TYPE_AGENT:
            return []
        if kind == SPAN_TYPE_LLM:
            response = _llm_span_to_response(span)
            if self._synthesize_usage and response.usage:
                return [
                    response,
                    UsageEvent(
                        response.usage,
                        kind="model_call",
                        model=response.model,
                        provider=response.provider,
                    ),
                ]
            return [response]
        if kind == SPAN_TYPE_USAGE:
            return [_usage_span_to_event(span)]
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
            return _oi_llm_to_events(span)
        if kind == _OI_KIND_TOOL:
            return _oi_tool_to_events(span)
        return None


DEFAULT_CONVENTIONS: tuple[SpanConvention, ...] = (AG2GenAIConvention(), OpenInferenceConvention())


def _needs_synthesized_usage(spans: Sequence[SpanData]) -> bool:
    """Whether this trace's AG2 spans carry their token counts on LLM spans alone.

    A trace captured before AG2 recorded usage spans does, and reading only the
    accounting event would report zero for it. A trace that *has* usage spans
    does not, and synthesizing as well would double every main-loop call.

    A property of the trace, not a preference of the caller — which is why
    :func:`_adapt_convention` hands the same answer to a caller-supplied AG2
    reader rather than leaving it on whatever its constructor was given.
    """
    return not any(span.attributes.get(ATTR_SPAN_TYPE) == SPAN_TYPE_USAGE for span in spans)


def _default_conventions(spans: Sequence[SpanData]) -> tuple[SpanConvention, ...]:
    """The default readers, adapted to whether this trace records usage spans."""
    if not _needs_synthesized_usage(spans):
        return DEFAULT_CONVENTIONS
    return (AG2GenAIConvention(synthesize_usage_from_llm_spans=True), OpenInferenceConvention())


def _adapt_convention(convention: SpanConvention, spans: Sequence[SpanData]) -> SpanConvention:
    """Give a caller-supplied AG2 reader this trace's synthesis setting.

    ``conventions=DEFAULT_CONVENTIONS`` and ``conventions=(*DEFAULT_CONVENTIONS,
    mine)`` are the obvious ways to spell "the defaults" and "the defaults plus
    mine". Both hand over an :class:`AG2GenAIConvention` built with synthesis
    off, so without this an archived trace read through either would report
    **zero** — the exact failure the switch exists to prevent, reached by the
    one line a caller would naturally write.

    Only the exact class is re-created: a subclass may read spans differently,
    so its instance is passed through untouched, as is any foreign reader.
    """
    if type(convention) is AG2GenAIConvention:
        return AG2GenAIConvention(synthesize_usage_from_llm_spans=_needs_synthesized_usage(spans))
    return convention


class _SpanTree:
    """Parent/child structure of one trace's spans, for ancestry questions.

    Exists so the walk below has a home and its two lookup tables stop being
    passed around together as a pair.
    """

    def __init__(self, spans: Sequence[SpanData]) -> None:
        self._by_id = {s.span_id: s for s in spans}
        self._agent_ids = {s.span_id for s in spans if s.attributes.get(ATTR_SPAN_TYPE) == SPAN_TYPE_AGENT}

    @property
    def agent_count(self) -> int:
        return len(self._agent_ids)

    def nearest_agent_ancestor(self, span: SpanData) -> str | None:
        """Span id of the closest enclosing agent span, or ``None`` at the top."""
        seen: set[str] = set()
        current = span.parent_id
        while current is not None and current not in seen:
            if current in self._agent_ids:
                return current
            seen.add(current)
            parent = self._by_id.get(current)
            current = parent.parent_id if parent is not None else None
        return None

    def is_nested_agent(self, span_id: str) -> bool:
        """True when this agent span itself runs under another agent span."""
        span = self._by_id.get(span_id)
        return span is not None and self.nearest_agent_ancestor(span) is not None

    def agent_name(self, span_id: str) -> str | None:
        """Name on an agent span, or ``None`` when the producer named nobody.

        ``TelemetryMiddleware`` writes ``"unknown"`` rather than omitting the
        attribute when the caller passed no ``agent_name``; that is a placeholder,
        not an identity, so it is reported as unnamed.
        """
        span = self._by_id.get(span_id)
        name = span.attributes.get(_ATTR_AGENT_NAME) if span is not None else None
        if not isinstance(name, str) or name == _AGENT_NAME_UNSET:
            return None
        return name


@dataclass(frozen=True, slots=True)
class _NestedSpend:
    """Spend recorded under one nested agent, and that agent's name if it has one."""

    name: str | None
    usage: Usage


def _matching_spend(candidates: Sequence[_NestedSpend], rollup: UsageEvent) -> _NestedSpend | None:
    """The nested spend a rollup duplicates, or ``None`` if it duplicates none.

    Named spend must agree with the rollup's label; unnamed spend falls back to
    value equality. Named candidates are tried first so an unnamed one cannot
    consume a rollup that its rightful owner would have claimed.
    """
    for named in (True, False):
        for candidate in candidates:
            if (candidate.name is not None) != named or candidate.usage != rollup.usage:
                continue
            if candidate.name is None or candidate.name == rollup.label:
                return candidate
    return None


def _nested_agent_spend(spans: Sequence[SpanData]) -> list["_NestedSpend"]:
    """What each sub-agent that ran instrumented in this trace spent, and who it was.

    Carries the agent's name alongside the total because value alone does not
    identify whose spend it is. Matching on value alone assumed every nested
    instrumented agent is *also* covered by a rollup on its parent — true of
    ``run_task``/``as_tool``, but not of a plain ``await other.ask(...)`` from
    inside a tool, which produces usage spans and no rollup at all. Its spend
    then had no rollup of its own to cancel and cancelled an unrelated worker's
    of equal value, losing those tokens outright. The name is read off the
    *agent* span, where ``TelemetryMiddleware`` always writes it — not off the
    usage span, which carries no agent name.

    Usage under a nested agent is attributed to the *nearest* enclosing agent,
    so a grandchild's spend cancels the grandchild's rollup rather than the
    child's.
    """
    tree = _SpanTree(spans)
    if tree.agent_count < 2:
        return []

    totals: dict[str, Usage] = {}
    for span in spans:
        if span.attributes.get(ATTR_SPAN_TYPE) != SPAN_TYPE_USAGE:
            continue
        owner = tree.nearest_agent_ancestor(span)
        # The outermost agent emits the rollups; only nested ones duplicate.
        if owner is None or not tree.is_nested_agent(owner):
            continue
        totals[owner] = totals.get(owner, Usage()) + _usage_span_to_event(span).usage
    return [_NestedSpend(name=tree.agent_name(owner), usage=total) for owner, total in totals.items() if total]


def _drop_duplicated_rollups(events: list[BaseEvent], duplicated: Sequence["_NestedSpend"]) -> list[BaseEvent]:
    """Remove ``"subtask"`` rollups an instrumented sub-agent already accounts for.

    A span tree flattens every instrumented agent into one trace, so a worker's
    own per-call accounting and the parent's rollup over it are both present;
    counting both reports that spend twice. The ordinary case — an
    uninstrumented worker — passes ``duplicated`` empty and the rollup is kept
    as the only record of that spend.

    A rollup is cancelled only by spend that both **matches its value** and
    **names the same agent**: the rollup's label is the sub-agent's name and the
    nested spend's name comes off that agent's own span. Value alone let an
    instrumented agent that produces no rollup cancel someone else's.

    Named spend that finds no rollup cancels nothing — it is simply an agent the
    parent never rolled up. Spend whose agent span was left unnamed (the caller
    passed no ``agent_name``, so the span says ``"unknown"``) identifies nobody
    and falls back to value-only matching, which is the best available and what
    this did for every case before.

    Each entry cancels at most one rollup, so the residual ambiguity is bounded:
    two *unnamed* workers with identical spend, one instrumented and one not,
    can still cancel the wrong one — but the total stays correct, only the label
    is misattributed.
    """
    if not duplicated:
        return events

    remaining = list(duplicated)
    kept: list[BaseEvent] = []
    for event in events:
        if isinstance(event, UsageEvent) and event.kind == "subtask":
            match = _matching_spend(remaining, event)
            if match is not None:
                remaining.remove(match)
                continue
        kept.append(event)
    return kept


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
    active = (
        _default_conventions(ordered)
        if conventions is None
        else tuple(_adapt_convention(c, ordered) for c in conventions)
    )

    events: list[BaseEvent] = []
    for span in ordered:
        for convention in active:
            mapped = convention.to_events(span)
            if mapped is not None:
                events.extend(mapped)
                break

    # Settled here rather than inside a convention: whether a rollup duplicates
    # a nested agent's own accounting is a fact about the whole span tree, and
    # threading it through the per-span pass made the reader stateful and
    # single-use.
    #
    # Applied whoever supplied the readers. Choosing a reader says what a *span*
    # means; it does not make the same delegation bill twice. Gating this on
    # ``conventions is None`` meant ``conventions=DEFAULT_CONVENTIONS`` — which
    # reads as a no-op — double-counted every instrumented sub-agent. It is a
    # no-op for a trace with no rollups, which is every foreign dialect.
    events = _drop_duplicated_rollups(events, _nested_agent_spend(ordered))

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


def _usage_span_to_event(span: SpanData) -> UsageEvent:
    """Rebuild the accounting event ``TelemetryMiddleware`` recorded.

    Its LLM-span counterpart is *not* also read as usage: a main-loop call
    produces both spans, so counting each would double every direct model call.
    """
    a = span.attributes
    usage = Usage(
        prompt_tokens=a.get(_ATTR_USAGE_INPUT),
        completion_tokens=a.get(_ATTR_USAGE_OUTPUT),
        total_tokens=a.get(ATTR_USAGE_TOTAL),
        cache_creation_input_tokens=a.get(_ATTR_USAGE_CACHE_CREATE),
        cache_read_input_tokens=a.get(_ATTR_USAGE_CACHE_READ),
        thinking_tokens=a.get(_ATTR_USAGE_THINKING),
    )
    return UsageEvent(
        usage,
        kind=a.get(ATTR_USAGE_KIND, "model_call"),
        # ``None``, not ``""`` — the live event documents ``None`` for anything
        # that is not a sub-task rollup, and callers test it with ``is None``.
        label=a.get(ATTR_USAGE_LABEL),
        model=a.get(_ATTR_RESPONSE_MODEL),
        provider=a.get(_ATTR_PROVIDER),
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
def _oi_llm_to_events(span: SpanData) -> list[BaseEvent]:
    """One LLM span → the response, plus a synthesized ``UsageEvent``.

    A foreign producer has no ``UsageEvent`` of its own to record, so its
    per-call usage is the best accounting available and is synthesized here.
    The AG2 convention deliberately does *not* do this: AG2 records usage as its
    own span, and synthesizing from the LLM span as well would count every
    main-loop call twice.
    """
    a = span.attributes
    content = a.get(_OI_OUTPUT_CONTENT)
    usage = Usage(prompt_tokens=a.get(_OI_TOKENS_PROMPT), completion_tokens=a.get(_OI_TOKENS_COMPLETION))
    model = a.get(_OI_MODEL)
    provider = a.get(_OI_PROVIDER)
    events: list[BaseEvent] = [
        ModelResponse(
            message=ModelMessage(content) if isinstance(content, str) else None,
            usage=usage,
            model=model,
            provider=provider,
            finish_reason=None,
        )
    ]
    # Declaring no counts is not the same as a call that cost nothing.
    if usage:
        events.append(UsageEvent(usage, kind="model_call", model=model, provider=provider))
    return events


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
