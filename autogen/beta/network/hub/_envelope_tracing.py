# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""OTel-touching span helpers shared by the Hub and ``HubTelemetryListener``.

Centralising every OpenTelemetry import here lets ``core.py`` guard its
import of this module with a single ``try/except ImportError`` and stay
OTel-free when tracing is not configured. The module is only imported
when a ``tracer_provider`` is actually in play.

Two responsibilities:

* **Span serialisation** — :func:`span_to_record` / :func:`write_span`
  turn an ended OTel span into the hub-native JSONL record written to
  ``/telemetry/spans.jsonl``. Both the Hub (envelope spans) and the
  listener (channel / agent / task spans) use these.
* **Envelope-span lifecycle** — :class:`EnvelopeTracer` owns the
  ``network.envelope`` span that the Hub brackets around WAL append and
  dispatch, including W3C traceparent injection into ``Envelope.trace_id``
  *before* the WAL write so the WAL is the source of truth.
"""

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.propagate import inject
from opentelemetry.trace import Link, SpanContext, SpanKind, Status, StatusCode

from autogen.beta._telemetry_consts import (
    ATTR_LINK_KIND,
    ATTR_NET_AUDIENCE,
    ATTR_NET_CAUSATION_ID,
    ATTR_NET_CHANNEL_ID,
    ATTR_NET_DISPATCH_FAILURES,
    ATTR_NET_ENVELOPE_ID,
    ATTR_NET_EVENT_TYPE,
    ATTR_NET_SENDER_ID,
    ATTR_SPAN_TYPE,
    LINK_IN_CHANNEL,
    LINK_TRIGGERED_BY,
    OTEL_INSTRUMENTING_MODULE,
    OTEL_SCHEMA_URL,
    SPAN_TYPE_ENVELOPE,
    TRACEPARENT_DEP_KEY,
)

from .layout import spans_path

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.trace import Span

    from autogen.beta.knowledge import KnowledgeStore

    from ..envelope import Envelope

__all__ = (
    "TRACEPARENT_DEP_KEY",
    "EnvelopeTracer",
    "get_tracer",
    "iso_to_ns",
    "serialize_span",
    "span_to_record",
    "write_span",
)

# ``TRACEPARENT_DEP_KEY`` is imported from ``autogen.beta._telemetry_consts``
# and re-exported here (kept in ``__all__``) for callers that reach for it on
# the hub-tracing surface.


def get_tracer(tracer_provider: "TracerProvider | None" = None) -> trace.Tracer:
    provider = tracer_provider or trace.get_tracer_provider()
    return provider.get_tracer(OTEL_INSTRUMENTING_MODULE, schema_url=OTEL_SCHEMA_URL)


def _iso_from_ns(ns: "int | None") -> "str | None":
    if ns is None:
        return None
    # ``timezone.utc`` (not ``datetime.UTC``) to keep the Python 3.10 floor.
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def iso_to_ns(iso: "str | None") -> "int | None":
    """Parse an ISO-8601 timestamp into nanoseconds since epoch (for span start/end times)."""
    if not iso:
        return None
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1_000_000_000)


def span_to_record(span: "ReadableSpan") -> dict:
    """Flatten an ended span into the hub-native JSONL record (see design §4.1).

    Hex trace/span ids are W3C-compatible, so the disk record and the
    OTLP export reference the same span. Safe to call only after
    ``span.end()`` — ``end_time`` is ``None`` before then.
    """
    ctx = span.get_span_context()
    parent = span.parent
    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
        "parent_span_id": format(parent.span_id, "016x") if parent is not None else None,
        "name": span.name,
        "kind": span.kind.name,
        "start": _iso_from_ns(span.start_time),
        "end": _iso_from_ns(span.end_time),
        "status": span.status.status_code.name,
        "status_message": span.status.description,
        "attributes": dict(span.attributes or {}),
        "events": [
            {
                "name": ev.name,
                "at": _iso_from_ns(ev.timestamp),
                "attributes": dict(ev.attributes or {}),
            }
            for ev in (span.events or ())
        ],
        "links": [
            {
                "trace_id": format(link.context.trace_id, "032x"),
                "span_id": format(link.context.span_id, "016x"),
                "attributes": dict(link.attributes or {}),
            }
            for link in (span.links or ())
        ],
    }


def serialize_span(span: "ReadableSpan") -> "tuple[dict, str]":
    """Return ``(record, jsonl_line)`` for an ended span — shared by Hub and listener."""
    record = span_to_record(span)
    line = json.dumps(record, default=str, sort_keys=True) + "\n"
    return record, line


async def write_span(store: "KnowledgeStore", span: "ReadableSpan") -> int:
    """Serialise an ended span to one JSONL line and append it. Returns bytes written."""
    _, line = serialize_span(span)
    await store.append(spans_path(), line)
    return len(line.encode("utf-8"))


class EnvelopeTracer:
    """Owns the Hub's ``network.envelope`` span (design §5).

    The Hub starts the span before WAL append, injects its traceparent
    into the envelope, runs WAL+dispatch inside the span's context, then
    ends the span and writes it to disk. OTel export is automatic via the
    shared ``TracerProvider``; the disk write mirrors it to
    ``/telemetry/spans.jsonl``.
    """

    def __init__(
        self,
        tracer_provider: "TracerProvider | None",
        store: "KnowledgeStore",
        *,
        span_attributes: "dict[str, str] | None" = None,
    ) -> None:
        self._tracer = get_tracer(tracer_provider)
        self._store = store
        self._stamp = dict(span_attributes or {})
        self._bytes_written = 0

    @property
    def bytes_written(self) -> int:
        return self._bytes_written

    def start_envelope_span(
        self,
        envelope: "Envelope",
        *,
        channel_span_context: "SpanContext | None" = None,
    ) -> "Span":
        """Start (but do not enter) the ``network.envelope`` span as a new-trace root.

        Adds SpanLinks to the channel span (if the caller resolved one)
        and to whatever span is current at post time (the triggering
        work). The span is *not* made current here — the Hub enters it
        explicitly around WAL+dispatch.
        """
        links: list[Link] = []
        if channel_span_context is not None and channel_span_context.is_valid:
            links.append(Link(channel_span_context, attributes={ATTR_LINK_KIND: LINK_IN_CHANNEL}))
        caller = trace.get_current_span().get_span_context()
        if caller.is_valid:
            links.append(Link(caller, attributes={ATTR_LINK_KIND: LINK_TRIGGERED_BY}))

        attributes: dict = {
            **self._stamp,
            ATTR_SPAN_TYPE: SPAN_TYPE_ENVELOPE,
            ATTR_NET_CHANNEL_ID: envelope.channel_id,
            ATTR_NET_SENDER_ID: envelope.sender_id,
            ATTR_NET_EVENT_TYPE: envelope.event_type,
        }
        if envelope.envelope_id:
            attributes[ATTR_NET_ENVELOPE_ID] = envelope.envelope_id
        if envelope.causation_id:
            attributes[ATTR_NET_CAUSATION_ID] = envelope.causation_id
        if envelope.audience is not None:
            attributes[ATTR_NET_AUDIENCE] = list(envelope.audience)

        # Force the span to be a trace root (Trace B), regardless of whether a
        # span is active in the ambient context. Without an explicit empty
        # ``Context()``, OTel parents the span under the current span — so an
        # envelope posted from inside an agent turn / tool (e.g. set_context →
        # EV_CONTEXT_SET) would nest under it and merge the two traces. The
        # causal relationship is preserved as the ``triggered_by`` SpanLink
        # captured above, not as a parent edge.
        return self._tracer.start_span(
            f"network.envelope {envelope.event_type}",
            kind=SpanKind.PRODUCER,
            context=Context(),
            links=links,
            attributes=attributes,
        )

    def inject_traceparent(self, envelope: "Envelope", span: "Span") -> None:
        """Write the span's W3C traceparent into ``envelope.trace_id`` (before WAL)."""
        carrier: dict[str, str] = {}
        inject(carrier, context=trace.set_span_in_context(span))
        traceparent = carrier.get("traceparent")
        if traceparent is not None:
            envelope.trace_id = traceparent

    async def finish_envelope_span(self, span: "Span", *, dispatch_failures: int = 0) -> None:
        """End the span (status from dispatch outcome) and mirror it to disk."""
        if dispatch_failures:
            span.set_attribute(ATTR_NET_DISPATCH_FAILURES, dispatch_failures)
            span.set_status(Status(StatusCode.ERROR, f"{dispatch_failures} dispatch failure(s)"))
        span.end()
        self._bytes_written += await write_span(self._store, span)

    def use_span(self, span: "Span") -> object:
        """Context manager that makes ``span`` current without ending it on exit."""
        return trace.use_span(span, end_on_exit=False)
