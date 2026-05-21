# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from opentelemetry.sdk.trace import TracerProvider

from autogen.agentchat import Agent
from autogen.doc_utils import export_module
from autogen.opentelemetry.setup import get_tracer

from .agent_instrumentators import (
    instrument_code_execution,
    instrument_create_or_get_executor,
    instrument_execute_function,
    instrument_generate_oai_reply,
    instrument_generate_reply,
    instrument_human_input,
    instrument_initiate_chat,
    instrument_initiate_chats,
    instrument_remote_reply,
    instrument_resume,
    instrument_run_chat,
)


@export_module("autogen.opentelemetry")
def instrument_agent(agent: Agent, *, tracer_provider: TracerProvider) -> Agent:
    """Instrument an agent with OpenTelemetry tracing.

    Instruments various agent methods to emit OpenTelemetry spans for:
    - Agent invocations (generate_reply, a_generate_reply)
    - Conversations (initiate_chat, a_initiate_chat, resume)
    - Tool execution (execute_function, a_execute_function)
    - Code execution
    - Human input requests
    - Remote agent calls

    Args:
        agent: The agent instance to instrument.
        tracer_provider: The OpenTelemetry tracer provider to use for creating spans.

    Returns:
        The instrumented agent instance (same object, modified in place).

    Usage:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from autogen.opentelemetry import instrument_agent

        resource = Resource.create(attributes={"service.name": "my-service"})
        tracer_provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:4317")
        processor = BatchSpanProcessor(exporter)
        tracer_provider.add_span_processor(processor)
        trace.set_tracer_provider(tracer_provider)

        agent = AssistantAgent("assistant")
        instrument_agent(agent, tracer_provider=tracer_provider)
    """
    tracer = get_tracer(tracer_provider)

    agent = instrument_initiate_chats(agent, tracer=tracer)
    agent = instrument_generate_reply(agent, tracer=tracer)
    agent = instrument_generate_oai_reply(agent, tracer=tracer)
    agent = instrument_initiate_chat(agent, tracer=tracer)
    agent = instrument_resume(agent, tracer=tracer)
    agent = instrument_run_chat(agent, tracer=tracer)
    agent = instrument_remote_reply(agent, tracer=tracer)
    agent = instrument_execute_function(agent, tracer=tracer)
    agent = instrument_create_or_get_executor(
        agent,
        instrumentator=partial(instrument_agent, tracer_provider=tracer_provider),
    )
    agent = instrument_human_input(agent, tracer=tracer)
    agent = instrument_code_execution(agent, tracer=tracer)

    return agent
