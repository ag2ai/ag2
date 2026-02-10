# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from opentelemetry.trace import Tracer

from autogen import Agent
from autogen.opentelemetry.consts import SpanType


def instrument_execute_function(agent: Agent, *, tracer: Tracer) -> Agent:
    # Instrument `execute_function` as an execute_tool span
    if hasattr(agent, "execute_function") and not hasattr(agent.execute_function, "__otel_wrapped__"):
        old_execute_function = agent.execute_function

        def execute_function_traced(
            func_call: dict[str, Any],
            call_id: str | None = None,
            verbose: bool = False,
        ) -> tuple[bool, dict[str, Any]]:
            func_name = func_call.get("name", "")
            with tracer.start_as_current_span(f"execute_tool {func_name}") as span:
                span.set_attribute("ag2.span.type", SpanType.TOOL.value)
                span.set_attribute("gen_ai.operation.name", "execute_tool")
                span.set_attribute("gen_ai.tool.name", func_name)
                span.set_attribute("gen_ai.tool.type", "function")
                if call_id:
                    span.set_attribute("gen_ai.tool.call.id", call_id)

                # Opt-in: Add tool call arguments
                arguments = func_call.get("arguments", "{}")
                if isinstance(arguments, str):
                    span.set_attribute("gen_ai.tool.call.arguments", arguments)
                else:
                    span.set_attribute("gen_ai.tool.call.arguments", json.dumps(arguments))

                is_success, result = old_execute_function(func_call, call_id, verbose)

                if not is_success:
                    span.set_attribute("error.type", "ExecutionError")
                else:
                    # Opt-in: Add tool call result (only on success)
                    content = result.get("content", "")
                    span.set_attribute("gen_ai.tool.call.result", str(content))

                return is_success, result

        execute_function_traced.__otel_wrapped__ = True
        agent.execute_function = execute_function_traced

    # Instrument `a_execute_function` as an async execute_tool span
    if hasattr(agent, "a_execute_function") and not hasattr(agent.a_execute_function, "__otel_wrapped__"):
        old_a_execute_function = agent.a_execute_function

        async def a_execute_function_traced(
            func_call: dict[str, Any],
            call_id: str | None = None,
            verbose: bool = False,
        ) -> tuple[bool, dict[str, Any]]:
            func_name = func_call.get("name", "")
            with tracer.start_as_current_span(f"execute_tool {func_name}") as span:
                span.set_attribute("ag2.span.type", SpanType.TOOL.value)
                span.set_attribute("gen_ai.operation.name", "execute_tool")
                span.set_attribute("gen_ai.tool.name", func_name)
                span.set_attribute("gen_ai.tool.type", "function")
                if call_id:
                    span.set_attribute("gen_ai.tool.call.id", call_id)

                # Opt-in: Add tool call arguments
                arguments = func_call.get("arguments", "{}")
                if isinstance(arguments, str):
                    span.set_attribute("gen_ai.tool.call.arguments", arguments)
                else:
                    span.set_attribute("gen_ai.tool.call.arguments", json.dumps(arguments))

                is_success, result = await old_a_execute_function(func_call, call_id, verbose)

                if not is_success:
                    span.set_attribute("error.type", "ExecutionError")
                else:
                    # Opt-in: Add tool call result (only on success)
                    content = result.get("content", "")
                    span.set_attribute("gen_ai.tool.call.result", str(content))

                return is_success, result

        a_execute_function_traced.__otel_wrapped__ = True
        agent.a_execute_function = a_execute_function_traced

    return agent
