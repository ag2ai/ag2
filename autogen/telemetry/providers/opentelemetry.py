# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry Provider

This module provides a telemetry provider that tracks interactions using OpenTelemetry.

Uses the following spans:
<ALL>

Uses the following events:
<ALL>

Main provider for telemetry for AG2, provides telemetry data on all activity in an AG2 workflow.

Data can be sent to an OpenTelemetry collector endpoint using either HTTP or gRPC.

Code example:
```python
from autogen.telemetry.telemetry_core import InstrumentationManager, telemetry_context
from autogen.telemetry.providers.opentelemetry import OpenTelemetryProvider

otel_provider = OpenTelemetryProvider(
    protocol="http",
    collector_endpoint="http://192.168.0.115:4318/v1/traces",
    service_name="ag2"
)

with telemetry_context(instrumentation_manager, "My Program Name", {"my_program_id": "456"}) as telemetry:
    ... AG2 workflow code here ...
    ... Telemetry sent to OpenTelemetry collector ...

```

Developer note:

For OpenTelemetry, spans need to be started AND ended so that a valid trace is produced. In code, all
'start_spans' should have a corresponding 'end_span'. If you don't see your trace, it is likely to be
because of this. So, when handling exceptions or multiple returns within a function, ensure that the
'end_span' is called in all cases.

"""
import contextlib
import json
import uuid
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPSpanExporterGRPC
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPSpanExporterHTTP
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span
from opentelemetry.trace import SpanKind as OTelSpanKind

from ..base_telemetry import EventKind, SpanContext, SpanKind, TelemetryProvider, _is_list_of_string_dicts


class OpenTelemetryProvider(TelemetryProvider):
    def __init__(self, collector_endpoint: str, service_name: str, protocol: str = "http"):
        """Initialize the OpenTelemetry provider.

        Args:
            protocol: The protocol to use for the collector endpoint, "grpc" or "http".
            collector_endpoint: The endpoint to send telemetry data to.
            service_name: The name of the service sending telemetry data.
        """
        assert protocol in ["http", "grpc"], f"Invalid protocol: {protocol}, must be 'http' or 'grpc'"

        # Create and set the global TracerProvider
        resource = Resource.create({"service.name": service_name})
        tracer_provider = TracerProvider(resource=resource)

        # Set up the OTLP exporter
        otlp_exporter = (
            OTLPSpanExporterHTTP(endpoint=collector_endpoint, headers={})
            if protocol == "http"
            else OTLPSpanExporterGRPC(endpoint=collector_endpoint)
        )

        # Add BatchSpanProcessor with the OTLP exporter
        span_processor = BatchSpanProcessor(otlp_exporter)

        tracer_provider.add_span_processor(span_processor)

        # Set the global TracerProvider
        trace.set_tracer_provider(tracer_provider)

        # Get a tracer
        self._tracer = trace.get_tracer("ag2.telemetry")

        # Store active spans
        self._active_spans = {}

    def start_trace(self, name: str, core_span_id: str, attributes: Dict[str, Any] = None) -> SpanContext:
        """Start a new trace with no parent context."""
        if attributes is None:
            attributes = {}

        # Create a new trace context
        trace_id = format(uuid.uuid4().int & ((1 << 128) - 1), "032x")

        # Convert attributes to OpenTelemetry compatible formats
        formatted_attributes = {key: self.convert_attribute_value(value) for key, value in attributes.items()}

        # Start the span with OpenTelemetry
        span = self._tracer.start_span(
            name=name,
            attributes={
                **formatted_attributes,
                "ag2.trace.id": trace_id,
                "ag2.span.type": "trace",
                "ag2.span.core_span_id": core_span_id,
            },
            kind=OTelSpanKind.INTERNAL,
        )

        # Create our setup span (captures object, like agents, creations))
        context = SpanContext(
            kind=SpanKind.WORKFLOW, trace_id=trace_id, attributes=attributes, core_span_id=core_span_id
        )

        # Store the active span
        self._active_spans[core_span_id] = span

        # DEBUG - REMOVE UPON REVIEW
        # print(f"OpenTelemetry: Started trace {trace_id}")

        return context

    def start_span(
        self,
        kind: SpanKind,
        core_span_id: str,
        parent_context: Optional[SpanContext] = None,
        attributes: Dict[str, Any] = None,
    ) -> SpanContext:
        """Start a new span, optionally as a child of a parent span."""
        if attributes is None:
            attributes = {}

        # If we have a parent context, use its trace ID
        trace_id = parent_context.trace_id if parent_context else format(uuid.uuid4().int & ((1 << 128) - 1), "032x")

        # Get parent span if it exists
        parent_span = None
        if parent_context:
            parent_span = self._active_spans.get(parent_context.core_span_id)

        # Start the span
        context_manager = trace.use_span(parent_span) if parent_span else contextlib.nullcontext()

        # Convert attributes to OpenTelemetry compatible formats
        formatted_attributes = {key: self.convert_attribute_value(value) for key, value in attributes.items()}

        # Create our span context
        context = SpanContext(
            kind=kind.value,
            trace_id=trace_id,
            parent_span_id=parent_context.core_span_id if parent_context else None,
            attributes=formatted_attributes,
            core_span_id=core_span_id,
        )

        with context_manager:
            span = self._tracer.start_span(
                name=kind.value,
                attributes={
                    **formatted_attributes,
                    "ag2.trace.id": context.trace_id,
                    "ag2.span.type": context.kind,
                    "ag2.span.core_span_id": context.core_span_id,
                },
                kind=OTelSpanKind.INTERNAL,
            )

        # Store the active span
        self._active_spans[core_span_id] = span

        # DEBUG - REMOVE UPON REVIEW
        # print(f"OpenTelemetry: Started span, {kind.value}, core_span_id: {core_span_id}")

        return context

    def set_span_attribute(self, context: SpanContext, key: str, value: Any) -> None:
        """Set an attribute on an active span"""
        span: Span = self._active_spans.get(context.core_span_id)
        if span:
            converted_value = self.convert_attribute_value(value)

            # Update both to keep in sync
            span.set_attribute(key, converted_value)
            context.set_attribute(key, converted_value)

    def end_span(self, context: SpanContext) -> None:
        """End a span identified by the given context."""

        # DEBUG - REMOVE UPON REVIEW
        # print(f"OpenTelemetry: Ended span, {context.kind}, core_span_id: {context.core_span_id}")

        span: Span = self._active_spans.get(context.core_span_id)
        if span:
            span.end()
            del self._active_spans[context.core_span_id]
        else:
            print(f"WARNING: OpenTelemetry: Span not found, core_span_id: {context.core_span_id}")

    def record_event(self, span_context: SpanContext, event_kind: EventKind, attributes: Dict[str, Any] = None) -> None:
        """Record an event in the given span."""
        if attributes is None:
            attributes = {}

        formatted_attributes = {}

        for key, value in attributes.items():
            formatted_attributes[key] = self.convert_attribute_value(value)

        span: Span = self._active_spans.get(span_context.core_span_id)
        if span:
            span.add_event(name=event_kind.value, attributes=formatted_attributes)

        # DEBUG - REMOVE UPON REVIEW
        # print(f"OpenTelemetry: Recorded event, {event_kind.value}, {span_context.core_span_id}")

    def convert_attribute_value(self, value: Any) -> Any:
        """Convert an attribute value to an OpenTelemetry Attribute value

        Documentation:
        https://opentelemetry.io/docs/specs/otel/common/#attribute

        Attributes must be non-null and non-empty string.

        Attributes can be either:
        1. A primitive type: string, boolean, double precision floating point (IEEE 754-1985) or signed 64 bit integer.
        2. An array of primitive type values. The array MUST be homogeneous, i.e., it MUST NOT contain values of different types.

        If needed, non-string values should be represented as JSON-encoded strings.
        """
        if value is None:
            return ""
        if isinstance(value, (str, int, bool, float)):
            return value
        if _is_list_of_string_dicts(value) or isinstance(value, dict):
            try:
                return json.dumps(value)
            except (TypeError, ValueError):
                pass  # Convert to string if fails

        return str(value)
