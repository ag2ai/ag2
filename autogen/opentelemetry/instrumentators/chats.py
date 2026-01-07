# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from opentelemetry.sdk.trace import TracerProvider

from autogen.agentchat import chat as chat_module
from autogen.agentchat import conversable_agent as conversable_agent_module
from autogen.doc_utils import export_module
from autogen.opentelemetry.consts import SpanType
from autogen.opentelemetry.setup import get_tracer


@export_module("autogen.opentelemetry")
def instrument_chats(*, tracer_provider: TracerProvider) -> None:
    """Instrument the standalone initiate_chats and a_initiate_chats functions.

    This adds a parent span that groups all sequential/parallel chats together,
    making it easy to trace multi-agent workflows.

    Usage:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from autogen.instrumentation import instrument_chats

        resource = Resource.create(attributes={"service.name": "my-service"})
        tracer_provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:4317")
        processor = BatchSpanProcessor(exporter)
        tracer_provider.add_span_processor(processor)
        trace.set_tracer_provider(tracer_provider)

        instrument_chats(tracer_provider=tracer_provider)

        # Now initiate_chats calls will be traced with a parent span
        from autogen import initiate_chats
        results = initiate_chats(chat_queue)
    """
    tracer = get_tracer(tracer_provider)
    # Instrument sync initiate_chats
    old_initiate_chats = chat_module.initiate_chats

    def initiate_chats_traced(chat_queue: list[dict[str, Any]]) -> list:
        with tracer.start_as_current_span("initiate_chats") as span:
            span.set_attribute("ag2.span.type", SpanType.MULTI_CONVERSATION.value)
            span.set_attribute("gen_ai.operation.name", "initiate_chats")
            span.set_attribute("ag2.chats.count", len(chat_queue))
            span.set_attribute("ag2.chats.mode", "sequential")

            # Capture recipient names
            recipients = [
                chat_info.get("recipient", {}).name
                if hasattr(chat_info.get("recipient"), "name")
                else str(chat_info.get("recipient"))
                for chat_info in chat_queue
            ]
            span.set_attribute("ag2.chats.recipients", json.dumps(recipients))

            results = old_initiate_chats(chat_queue)

            # Capture chat IDs
            chat_ids = [str(r.chat_id) for r in results if hasattr(r, "chat_id")]
            span.set_attribute("ag2.chats.ids", json.dumps(chat_ids))

            # Capture summaries
            summaries = [r.summary for r in results if hasattr(r, "summary")]
            span.set_attribute("ag2.chats.summaries", json.dumps(summaries))

            return results

    # Patch in all locations where initiate_chats may have been imported
    chat_module.initiate_chats = initiate_chats_traced
    conversable_agent_module.initiate_chats = initiate_chats_traced

    # Instrument async a_initiate_chats
    old_a_initiate_chats = chat_module.a_initiate_chats

    async def a_initiate_chats_traced(chat_queue: list[dict[str, Any]]) -> dict:
        with tracer.start_as_current_span("initiate_chats") as span:
            span.set_attribute("ag2.span.type", SpanType.MULTI_CONVERSATION.value)
            span.set_attribute("gen_ai.operation.name", "initiate_chats")
            span.set_attribute("ag2.chats.count", len(chat_queue))
            span.set_attribute("ag2.chats.mode", "parallel")

            # Capture recipient names
            recipients = [
                chat_info.get("recipient", {}).name
                if hasattr(chat_info.get("recipient"), "name")
                else str(chat_info.get("recipient"))
                for chat_info in chat_queue
            ]
            span.set_attribute("ag2.chats.recipients", json.dumps(recipients))

            # Capture prerequisites if any
            has_prerequisites = any("prerequisites" in chat_info for chat_info in chat_queue)
            if has_prerequisites:
                prerequisites = {
                    chat_info.get("chat_id", i): chat_info.get("prerequisites", [])
                    for i, chat_info in enumerate(chat_queue)
                }
                span.set_attribute("ag2.chats.prerequisites", json.dumps(prerequisites))

            results = await old_a_initiate_chats(chat_queue)

            # Capture chat IDs (results is a dict for async version)
            chat_ids = [str(r.chat_id) for r in results.values() if hasattr(r, "chat_id")]
            span.set_attribute("ag2.chats.ids", json.dumps(chat_ids))

            # Capture summaries (results is a dict for async version)
            summaries = [r.summary for r in results.values() if hasattr(r, "summary")]
            span.set_attribute("ag2.chats.summaries", json.dumps(summaries))

            return results

    # Patch in all locations where a_initiate_chats may have been imported
    chat_module.a_initiate_chats = a_initiate_chats_traced
    conversable_agent_module.a_initiate_chats = a_initiate_chats_traced
