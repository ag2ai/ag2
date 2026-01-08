# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from opentelemetry.trace import Tracer

from autogen import Agent
from autogen.opentelemetry.consts import SpanType


def instrument_chats(agent: Agent, *, tracer: Tracer) -> Agent:
    if hasattr(agent, "initiate_chats"):
        old_initiate_chats = agent.initiate_chats

        def initiate_chats_traced(chat_queue: list[dict[str, Any]]) -> list:
            with tracer.start_as_current_span("agent.initiate_chats") as span:
                span.set_attribute("ag2.span.type", SpanType.MULTI_CONVERSATION.value)
                span.set_attribute("gen_ai.operation.name", "initiate_chats")
                span.set_attribute("gen_ai.agent.name", agent.name)
                span.set_attribute("ag2.chats.count", len(chat_queue))
                span.set_attribute("ag2.chats.mode", "sequential")

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

        agent.initiate_chats = initiate_chats_traced

    if hasattr(agent, "a_initiate_chats"):
        old_a_initiate_chats = agent.a_initiate_chats

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

        agent.a_initiate_chats = a_initiate_chats_traced

    return agent
