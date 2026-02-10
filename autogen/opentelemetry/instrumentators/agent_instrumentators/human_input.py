# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from opentelemetry.trace import Tracer

from autogen import Agent
from autogen.opentelemetry.consts import SpanType


def instrument_human_input(agent: Agent, *, tracer: Tracer) -> Agent:
    # Instrument `get_human_input` as an await_human_input span
    if hasattr(agent, "get_human_input") and not hasattr(agent.get_human_input, "__otel_wrapped__"):
        old_get_human_input = agent.get_human_input

        def get_human_input_traced(
            prompt: str,
            *args: Any,
            **kwargs: Any,
        ) -> str:
            with tracer.start_as_current_span(f"await_human_input {agent.name}") as span:
                span.set_attribute("ag2.span.type", SpanType.HUMAN_INPUT.value)
                span.set_attribute("gen_ai.operation.name", "await_human_input")
                span.set_attribute("gen_ai.agent.name", agent.name)
                span.set_attribute("ag2.human_input.prompt", prompt)

                response = old_get_human_input(prompt, *args, **kwargs)

                # Opt-in: capture response (may contain sensitive data)
                span.set_attribute("ag2.human_input.response", response)
                return response

        get_human_input_traced.__otel_wrapped__ = True
        agent.get_human_input = get_human_input_traced

    # Instrument `a_get_human_input` as an async await_human_input span
    if hasattr(agent, "a_get_human_input") and not hasattr(agent.a_get_human_input, "__otel_wrapped__"):
        old_a_get_human_input = agent.a_get_human_input

        async def a_get_human_input_traced(
            prompt: str,
            *args: Any,
            **kwargs: Any,
        ) -> str:
            with tracer.start_as_current_span(f"await_human_input {agent.name}") as span:
                span.set_attribute("ag2.span.type", SpanType.HUMAN_INPUT.value)
                span.set_attribute("gen_ai.operation.name", "await_human_input")
                span.set_attribute("gen_ai.agent.name", agent.name)
                span.set_attribute("ag2.human_input.prompt", prompt)

                response = await old_a_get_human_input(prompt, *args, **kwargs)

                span.set_attribute("ag2.human_input.response", response)
                return response

        a_get_human_input_traced.__otel_wrapped__ = True
        agent.a_get_human_input = a_get_human_input_traced

    return agent
