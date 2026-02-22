# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import functools
import json
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.context import Context
from opentelemetry.trace import Tracer

from autogen import Agent
from autogen.io import IOStream
from autogen.opentelemetry.consts import SpanType
from autogen.opentelemetry.utils import (
    get_model_name,
    get_provider_name,
    messages_to_otel,
    reply_to_otel_message,
)


def instrument_generate_reply(agent: Agent, *, tracer: Tracer) -> Agent:
    # Instrument `a_generate_reply` as an invoke_agent span
    if not hasattr(agent.a_generate_reply, "__otel_wrapped__"):
        old_a_generate_reply = agent.a_generate_reply

        async def a_generate_traced_reply(
            messages: list[dict[str, Any]] | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            with tracer.start_as_current_span(f"invoke_agent {agent.name}") as span:
                span.set_attribute("ag2.span.type", SpanType.AGENT.value)
                span.set_attribute("gen_ai.operation.name", "invoke_agent")
                span.set_attribute("gen_ai.agent.name", agent.name)

                # Set provider and model from agent's LLM config
                provider = get_provider_name(agent)
                if provider:
                    span.set_attribute("gen_ai.provider.name", provider)
                model = get_model_name(agent)
                if model:
                    span.set_attribute("gen_ai.request.model", model)

                # Capture input messages
                if messages:
                    otel_input = messages_to_otel(messages)
                    span.set_attribute("gen_ai.input.messages", json.dumps(otel_input))

                reply = await old_a_generate_reply(messages, *args, **kwargs)

                # Capture output message
                if reply is not None:
                    otel_output = reply_to_otel_message(reply)
                    span.set_attribute("gen_ai.output.messages", json.dumps(otel_output))

                return reply

        a_generate_traced_reply.__otel_wrapped__ = True
        agent.a_generate_reply = a_generate_traced_reply

    # Instrument `generate_reply` (sync) as an invoke_agent span
    if hasattr(agent, "generate_reply") and not hasattr(agent.generate_reply, "__otel_wrapped__"):
        old_generate_reply = agent.generate_reply

        def generate_traced_reply(
            messages: list[dict[str, Any]] | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            with tracer.start_as_current_span(f"invoke_agent {agent.name}") as span:
                span.set_attribute("ag2.span.type", SpanType.AGENT.value)
                span.set_attribute("gen_ai.operation.name", "invoke_agent")
                span.set_attribute("gen_ai.agent.name", agent.name)

                # Set provider and model from agent's LLM config
                provider = get_provider_name(agent)
                if provider:
                    span.set_attribute("gen_ai.provider.name", provider)
                model = get_model_name(agent)
                if model:
                    span.set_attribute("gen_ai.request.model", model)

                if messages:
                    otel_input = messages_to_otel(messages)
                    span.set_attribute("gen_ai.input.messages", json.dumps(otel_input))

                reply = old_generate_reply(messages, *args, **kwargs)

                if reply is not None:
                    otel_output = reply_to_otel_message(reply)
                    span.set_attribute("gen_ai.output.messages", json.dumps(otel_output))

                return reply

        generate_traced_reply.__otel_wrapped__ = True
        agent.generate_reply = generate_traced_reply

    return agent


def instrument_generate_oai_reply(agent: Agent, *, tracer: Tracer) -> Agent:
    # Instrument `a_generate_oai_reply` to propagate context to executor thread
    # Critical because a_generate_oai_reply uses run_in_executor which
    # creates a new thread that doesn't inherit OpenTelemetry context so
    # will create new traces instead of being a child span.
    if hasattr(agent, "a_generate_oai_reply") and not hasattr(agent.a_generate_oai_reply, "__otel_wrapped__"):

        async def a_generate_oai_reply_with_context(
            messages: list[dict[str, Any]] | None = None,
            sender: Agent | None = None,
            config: Any | None = None,
            **kwargs: Any,
        ) -> tuple[bool, str | dict[str, Any] | None]:
            # Capture current OpenTelemetry context BEFORE run_in_executor
            current_context = otel_context.get_current()

            iostream = IOStream.get_default()

            def _generate_oai_reply_with_context(
                self_agent: Any,
                captured_context: Context,
                iostream: IOStream,
                *args: Any,
                **kw: Any,
            ) -> tuple[bool, str | dict[str, Any] | None]:
                # Attach the captured context in this thread
                token = otel_context.attach(captured_context)
                try:
                    with IOStream.set_default(iostream):
                        return self_agent.generate_oai_reply(*args, **kw)
                finally:
                    otel_context.detach(token)

            return await asyncio.get_event_loop().run_in_executor(
                None,
                functools.partial(
                    _generate_oai_reply_with_context,
                    self_agent=agent,
                    captured_context=current_context,
                    iostream=iostream,
                    messages=messages,
                    sender=sender,
                    config=config,
                    **kwargs,
                ),
            )

        a_generate_oai_reply_with_context.__otel_wrapped__ = True
        agent.a_generate_oai_reply = a_generate_oai_reply_with_context

        # Also update the reply function in _reply_func_list
        for i, reply_func_entry in enumerate(agent._reply_func_list):
            func = reply_func_entry.get("reply_func")
            if getattr(func, "__name__", None) == "a_generate_oai_reply":
                # Create a wrapper that matches the expected signature (self, messages, sender, config)
                async def a_generate_oai_reply_func_with_context(
                    self_agent: Any,
                    messages: list[dict[str, Any]] | None = None,
                    sender: Agent | None = None,
                    config: Any | None = None,
                ) -> tuple[bool, str | dict[str, Any] | None]:
                    return await self_agent.a_generate_oai_reply(messages, sender, config)

                agent._reply_func_list[i]["reply_func"] = a_generate_oai_reply_func_with_context
                break

    return agent
