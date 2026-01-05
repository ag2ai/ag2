# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from opentelemetry.trace import Tracer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from autogen.a2a import A2aAgentServer
from autogen.doc_utils import export_module
from autogen.opentelemetry.utils import TRACE_PROPAGATOR

from .agent import instrument_agent


@export_module("autogen.opentelemetry")
def instrument_a2a_server(server: A2aAgentServer, tracer: Tracer) -> A2aAgentServer:
    """Instrument an A2A server with OpenTelemetry tracing.

    Adds OpenTelemetry middleware to the server to trace incoming requests and
    instruments the server's agent for full observability.

    Args:
        server: The A2A agent server to instrument.
        tracer: The OpenTelemetry tracer to use for creating spans.

    Returns:
        The instrumented server instance.

    Usage:
        from autogen.opentelemetry import setup_instrumentation, instrument_a2a_server

        tracer = setup_instrumentation("my-service")
        server = A2aAgentServer(agent)
        instrument_a2a_server(server, tracer)
    """

    class OTELMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            span_context = TRACE_PROPAGATOR.extract(request.headers)
            with tracer.start_as_current_span("a2a-execution", context=span_context):
                return await call_next(request)

    server.add_middleware(OTELMiddleware)

    instrument_agent(server.agent, tracer)
    return server
