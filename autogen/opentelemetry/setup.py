# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Optional

from a2a.utils.telemetry import SpanKind
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Attributes, Resource
from opentelemetry.sdk.trace import Tracer, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult

from autogen.doc_utils import export_module

from .consts import INSTRUMENTING_LIBRARY_VERSION, INSTRUMENTING_MODULE_NAME, OTEL_SCHEMA


class DropNoiseSampler(Sampler):
    def should_sample(
        self,
        parent_context: Optional["Context"],
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Attributes = None,
        links: Sequence["trace.Link"] | None = None,
        trace_state: trace.TraceState | None = None,
    ) -> "SamplingResult":
        decision = Decision.RECORD_ONLY if name.startswith("a2a.") else Decision.RECORD_AND_SAMPLE
        return SamplingResult(decision, attributes=None, trace_state=trace_state)

    def get_description(self) -> str:
        return "Drop a2a.server noisy spans"


@export_module("autogen.opentelemetry")
def setup_instrumentation(service_name: str, endpoint: str = "http://127.0.0.1:4317") -> Tracer:
    """Set up OpenTelemetry instrumentation for AG2.

    Configures the OpenTelemetry tracer provider with an OTLP exporter and returns
    a tracer configured for AG2's instrumentation.

    Args:
        service_name: The name of the service to use in traces.
        endpoint: The OTLP endpoint URL. Defaults to "http://127.0.0.1:4317".

    Returns:
        A configured OpenTelemetry tracer instance.

    Usage:
        from autogen.opentelemetry import setup_instrumentation

        tracer = setup_instrumentation("my-service", endpoint="http://localhost:4317")
    """
    resource = Resource.create(attributes={"service.name": service_name})
    tracer_provider = TracerProvider(
        resource=resource,
        # sampler=DropNoiseSampler(),
    )
    exporter = OTLPSpanExporter(endpoint=endpoint)
    processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)

    return tracer_provider.get_tracer(
        instrumenting_module_name=INSTRUMENTING_MODULE_NAME,
        instrumenting_library_version=INSTRUMENTING_LIBRARY_VERSION,
        schema_url=OTEL_SCHEMA,
    )
