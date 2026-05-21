# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from opentelemetry.sdk.trace import TracerProvider
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from autogen.a2a import A2aAgentServer
from autogen.doc_utils import export_module
from autogen.opentelemetry.setup import get_tracer
from autogen.opentelemetry.utils import TRACE_PROPAGATOR

from .agent import instrument_agent


@export_module("autogen.opentelemetry")
def instrument_a2a_server(server: A2aAgentServer, *, tracer_provider: TracerProvider) -> A2aAgentServer:
    """Instrument an A2A server with OpenTelemetry tracing.

    Adds OpenTelemetry middleware to the server to trace incoming requests and
    instruments the server's agent for full observability.

    Args:
        server: The A2A agent server to instrument.
        tracer_provider: The OpenTelemetry tracer provider to use for creating spans.

    Returns:
        The instrumented server instance.

    Usage:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from autogen.opentelemetry import instrument_a2a_server

        resource = Resource.create(attributes={"service.name": "my-service"})
        tracer_provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:4317")
        processor = BatchSpanProcessor(exporter)
        tracer_provider.add_span_processor(processor)
        trace.set_tracer_provider(tracer_provider)

        server = A2aAgentServer(agent)
        instrument_a2a_server(server, tracer_provider=tracer_provider)
    """
    tracer = get_tracer(tracer_provider)

    if getattr(server, "__otel_instrumented__", False):
        return server

    class OTELMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if "traceparent" in request.headers:
                try:
                    span_context = TRACE_PROPAGATOR.extract(request.headers)
                except Exception:
                    span_context = None

                if span_context is not None:
                    with tracer.start_as_current_span("a2a-execution", context=span_context):
                        return await call_next(request)
            return await call_next(request)

    server.add_middleware(OTELMiddleware)

    server.agent = instrument_agent(server.agent, tracer_provider=tracer_provider)
    server.__otel_instrumented__ = True
    return server
