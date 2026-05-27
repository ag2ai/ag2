# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""OTLP-JSON → :class:`SpanData` parser.

A trace fetched from an OTLP store (e.g. Tempo's ``GET /api/traces/{id}``)
arrives as OTLP-JSON — resource batches of scope spans, with attributes in the
``{"key": ..., "value": {"stringValue"|"intValue"|...}}`` form, times as
nanosecond strings, and a numeric/string status code. This module flattens that
into the SDK-free :class:`SpanData` the pure ``spans_to_trace`` adapter
consumes, so it depends on neither the OpenTelemetry SDK nor any HTTP client.

It is generic OTLP-JSON, not Tempo-specific: any backend that returns this
shape can reuse it.
"""

from typing import Any

from ._spans import SpanData, SpanEvent

__all__ = ("otlp_json_to_spans",)


def otlp_json_to_spans(doc: dict[str, Any]) -> list[SpanData]:
    """Parse an OTLP-JSON trace document into a flat list of :class:`SpanData`.

    Accepts both the ``"batches"`` key (Tempo's trace-by-id response) and the
    ``"resourceSpans"`` key (raw OTLP export), and both ``"scopeSpans"`` and the
    legacy ``"instrumentationLibrarySpans"``.
    """
    batches = doc.get("batches") or doc.get("resourceSpans") or []
    spans: list[SpanData] = []
    for batch in batches:
        scopes = batch.get("scopeSpans") or batch.get("instrumentationLibrarySpans") or []
        for scope in scopes:
            for span in scope.get("spans", []):
                spans.append(_span_to_data(span))
    return spans


def _span_to_data(span: dict[str, Any]) -> SpanData:
    return SpanData(
        name=span.get("name", ""),
        span_id=span.get("spanId", ""),
        parent_id=span.get("parentSpanId") or None,
        start_ns=int(span.get("startTimeUnixNano", 0)),
        end_ns=int(span.get("endTimeUnixNano", 0)),
        attributes=_flatten_attributes(span.get("attributes", [])),
        status=_status(span.get("status", {})),
        events=tuple(
            SpanEvent(event.get("name", ""), _flatten_attributes(event.get("attributes", [])))
            for event in span.get("events", [])
        ),
    )


def _flatten_attributes(attributes: list[dict[str, Any]]) -> dict[str, Any]:
    """``[{"key": k, "value": {"<type>Value": v}}, ...]`` → ``{k: v}`` with native types."""
    return {attr["key"]: _attr_value(attr.get("value", {})) for attr in attributes if "key" in attr}


def _attr_value(value: dict[str, Any]) -> Any:
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])  # OTLP-JSON encodes ints as strings
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        return [_attr_value(v) for v in value["arrayValue"].get("values", [])]
    return None


def _status(status: dict[str, Any]) -> str:
    """Map an OTLP status (numeric or string code) to ``OK`` / ``ERROR`` / ``UNSET``."""
    code = status.get("code")
    if code in (2, "STATUS_CODE_ERROR", "ERROR"):
        return "ERROR"
    if code in (1, "STATUS_CODE_OK", "OK"):
        return "OK"
    return "UNSET"
