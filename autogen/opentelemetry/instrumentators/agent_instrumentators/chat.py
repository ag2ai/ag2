# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from opentelemetry.trace import Tracer

from autogen import Agent
from autogen.opentelemetry.consts import SpanType
from autogen.opentelemetry.utils import (
    aggregate_usage,
    get_model_name,
    get_provider_name,
    messages_to_otel,
)


def instrument_initiate_chat(agent: Agent, *, tracer: Tracer) -> Agent:
    # Instrument `a_initiate_chat` as a conversation span
    if hasattr(agent, "a_initiate_chat") and not hasattr(agent.a_initiate_chat, "__otel_wrapped__"):
        old_a_initiate_chat = agent.a_initiate_chat

        async def a_initiate_traced_chat(
            *args: Any,
            max_turns: int | None = None,
            message: str | dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            with tracer.start_as_current_span(f"conversation {agent.name}") as span:
                # Set AG2 span type and OTEL GenAI semantic convention attributes
                span.set_attribute("ag2.span.type", SpanType.CONVERSATION.value)
                span.set_attribute("gen_ai.operation.name", "conversation")
                span.set_attribute("gen_ai.agent.name", agent.name)

                # Set provider and model from recipient's LLM config (first positional arg)
                if args:
                    recipient = args[0]
                    provider = get_provider_name(recipient)
                    if provider:
                        span.set_attribute("gen_ai.provider.name", provider)
                    model = get_model_name(recipient)
                    if model:
                        span.set_attribute("gen_ai.request.model", model)

                if max_turns:
                    span.set_attribute("gen_ai.conversation.max_turns", max_turns)

                # Capture input message
                if message is not None:
                    if isinstance(message, str):
                        input_msg = {"role": "user", "content": message}
                    elif isinstance(message, dict):
                        input_msg = {"role": message.get("role", "user"), **message}
                    else:
                        input_msg = None

                    if input_msg:
                        otel_input = messages_to_otel([input_msg])
                        span.set_attribute("gen_ai.input.messages", json.dumps(otel_input))

                result = await old_a_initiate_chat(*args, max_turns=max_turns, message=message, **kwargs)

                span.set_attribute("gen_ai.conversation.id", str(result.chat_id))
                span.set_attribute("gen_ai.conversation.turns", len(result.chat_history))

                # Capture output messages (full chat history)
                if result.chat_history:
                    otel_output = messages_to_otel(result.chat_history)
                    span.set_attribute("gen_ai.output.messages", json.dumps(otel_output))

                usage_including_cached_inference = result.cost["usage_including_cached_inference"]
                total_cost = usage_including_cached_inference.pop("total_cost")
                span.set_attribute("gen_ai.usage.cost", total_cost)

                usage = aggregate_usage(usage_including_cached_inference)
                if usage:
                    model, input_tokens, output_tokens = usage
                    span.set_attribute("gen_ai.response.model", model)
                    span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                    span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

                return result

        a_initiate_traced_chat.__otel_wrapped__ = True
        agent.a_initiate_chat = a_initiate_traced_chat

    # Instrument `initiate_chat` (sync) as a conversation span
    if hasattr(agent, "initiate_chat") and not hasattr(agent.initiate_chat, "__otel_wrapped__"):
        old_initiate_chat = agent.initiate_chat

        def initiate_traced_chat(
            *args: Any,
            max_turns: int | None = None,
            message: str | dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            with tracer.start_as_current_span(f"conversation {agent.name}") as span:
                span.set_attribute("ag2.span.type", SpanType.CONVERSATION.value)
                span.set_attribute("gen_ai.operation.name", "conversation")
                span.set_attribute("gen_ai.agent.name", agent.name)

                # Set provider and model from recipient's LLM config (first positional arg)
                if args:
                    recipient = args[0]
                    provider = get_provider_name(recipient)
                    if provider:
                        span.set_attribute("gen_ai.provider.name", provider)
                    model = get_model_name(recipient)
                    if model:
                        span.set_attribute("gen_ai.request.model", model)

                if max_turns:
                    span.set_attribute("gen_ai.conversation.max_turns", max_turns)

                if message is not None:
                    if isinstance(message, str):
                        input_msg = {"role": "user", "content": message}
                    elif isinstance(message, dict):
                        input_msg = {"role": message.get("role", "user"), **message}
                    else:
                        input_msg = None

                    if input_msg:
                        otel_input = messages_to_otel([input_msg])
                        span.set_attribute("gen_ai.input.messages", json.dumps(otel_input))

                result = old_initiate_chat(*args, max_turns=max_turns, message=message, **kwargs)

                span.set_attribute("gen_ai.conversation.id", str(result.chat_id))
                span.set_attribute("gen_ai.conversation.turns", len(result.chat_history))

                if result.chat_history:
                    otel_output = messages_to_otel(result.chat_history)
                    span.set_attribute("gen_ai.output.messages", json.dumps(otel_output))

                usage_including_cached_inference = result.cost["usage_including_cached_inference"]
                total_cost = usage_including_cached_inference.pop("total_cost")
                span.set_attribute("gen_ai.usage.cost", total_cost)

                usage = aggregate_usage(usage_including_cached_inference)
                if usage:
                    model, input_tokens, output_tokens = usage
                    span.set_attribute("gen_ai.response.model", model)
                    span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                    span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

                return result

        initiate_traced_chat.__otel_wrapped__ = True
        agent.initiate_chat = initiate_traced_chat

    return agent


def instrument_resume(agent: Agent, *, tracer: Tracer) -> Agent:
    # Instrument `a_resume` as a resumed conversation span
    if hasattr(agent, "a_resume") and not hasattr(agent.a_resume, "__otel_wrapped__"):
        old_a_resume = agent.a_resume

        async def a_resume_traced(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            with tracer.start_as_current_span(f"conversation {agent.name}") as span:
                span.set_attribute("ag2.span.type", SpanType.CONVERSATION.value)
                span.set_attribute("gen_ai.operation.name", "conversation")
                span.set_attribute("gen_ai.agent.name", agent.name)
                span.set_attribute("gen_ai.conversation.resumed", True)
                return await old_a_resume(*args, **kwargs)

        a_resume_traced.__otel_wrapped__ = True
        agent.a_resume = a_resume_traced

    return agent


def instrument_run_chat(agent: Agent, *, tracer: Tracer) -> Agent:
    # Instrument `run_chat` as a conversation span (GroupChatManager, sync)
    if hasattr(agent, "run_chat") and not hasattr(agent.run_chat, "__otel_wrapped__"):
        old_run_chat = agent.run_chat

        def run_chat_traced(
            messages: list[dict[str, Any]] | None = None,
            sender: Agent | None = None,
            config: Any = None,  # GroupChat
            **kwargs: Any,
        ) -> Any:
            with tracer.start_as_current_span(f"conversation {agent.name}") as span:
                span.set_attribute("ag2.span.type", SpanType.CONVERSATION.value)
                span.set_attribute("gen_ai.operation.name", "conversation")
                span.set_attribute("gen_ai.agent.name", agent.name)

                # Capture input messages
                if messages:
                    otel_input = messages_to_otel(messages)
                    span.set_attribute("gen_ai.input.messages", json.dumps(otel_input))

                result = old_run_chat(messages=messages, sender=sender, config=config, **kwargs)

                # Capture output messages from groupchat
                if config and hasattr(config, "messages") and config.messages:
                    otel_output = messages_to_otel(config.messages)
                    span.set_attribute("gen_ai.output.messages", json.dumps(otel_output))

                return result

        run_chat_traced.__otel_wrapped__ = True
        agent.run_chat = run_chat_traced

    # Instrument `a_run_chat` as a conversation span (GroupChatManager)
    if hasattr(agent, "a_run_chat") and not hasattr(agent.a_run_chat, "__otel_wrapped__"):
        old_a_run_chat = agent.a_run_chat

        async def a_run_chat_traced(
            messages: list[dict[str, Any]] | None = None,
            sender: Agent | None = None,
            config: Any = None,  # GroupChat
            **kwargs: Any,
        ) -> Any:
            with tracer.start_as_current_span(f"conversation {agent.name}") as span:
                span.set_attribute("ag2.span.type", SpanType.CONVERSATION.value)
                span.set_attribute("gen_ai.operation.name", "conversation")
                span.set_attribute("gen_ai.agent.name", agent.name)

                # Capture input messages
                if messages:
                    otel_input = messages_to_otel(messages)
                    span.set_attribute("gen_ai.input.messages", json.dumps(otel_input))

                result = await old_a_run_chat(messages=messages, sender=sender, config=config, **kwargs)

                # Capture output messages from groupchat
                if config and hasattr(config, "messages") and config.messages:
                    otel_output = messages_to_otel(config.messages)
                    span.set_attribute("gen_ai.output.messages", json.dumps(otel_output))

                return result

        a_run_chat_traced.__otel_wrapped__ = True
        agent.a_run_chat = a_run_chat_traced

    return agent


def instrument_initiate_chats(agent: Agent, *, tracer: Tracer) -> Agent:
    if hasattr(agent, "initiate_chats") and not hasattr(agent.initiate_chats, "__otel_wrapped__"):
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

        initiate_chats_traced.__otel_wrapped__ = True
        agent.initiate_chats = initiate_chats_traced

    if hasattr(agent, "a_initiate_chats") and not hasattr(agent.a_initiate_chats, "__otel_wrapped__"):
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

        a_initiate_chats_traced.__otel_wrapped__ = True
        agent.a_initiate_chats = a_initiate_chats_traced

    return agent
