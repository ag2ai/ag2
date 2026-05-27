# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

from opentelemetry.trace import Tracer

from autogen import Agent
from autogen.opentelemetry.consts import SpanType


def instrument_code_execution(agent: Agent, *, tracer: Tracer) -> Agent:
    # Instrument `_generate_code_execution_reply_using_executor` as execute_code span
    # NOTE: The method is registered in _reply_func_list during __init__, so we need to
    # update both the method AND the registered callback
    if hasattr(agent, "_reply_func_list"):
        # Find the original reply func in _reply_func_list
        original_code_exec_func = None
        original_code_exec_index = None
        for i, reply_func_tuple in enumerate(agent._reply_func_list):
            func_name = getattr(reply_func_tuple.get("reply_func"), "__name__", None)
            if func_name == "_generate_code_execution_reply_using_executor":
                original_code_exec_func = reply_func_tuple["reply_func"]
                original_code_exec_index = i
                break

        if original_code_exec_func is not None:
            # Create traced wrapper that accepts self as first arg (like unbound method)
            def generate_code_execution_reply_traced(
                self_agent: Any,
                messages: list[dict[str, Any]] | None = None,
                sender: Agent | None = None,
                config: dict[str, Any] | None = None,
            ) -> tuple[bool, str | None]:
                # Check if code execution is disabled
                if self_agent._code_execution_config is False:
                    return False, None

                with tracer.start_as_current_span(f"execute_code {self_agent.name}") as span:
                    span.set_attribute("ag2.span.type", SpanType.CODE_EXECUTION.value)
                    span.set_attribute("gen_ai.operation.name", "execute_code")
                    span.set_attribute("gen_ai.agent.name", self_agent.name)

                    # Call original method
                    is_final, result = original_code_exec_func(self_agent, messages, sender, config)

                    # Parse the result to extract exit code and output
                    # Result format: "exitcode: X (status)\nCode output: ..."
                    if is_final and result and result.startswith("exitcode:"):
                        parts = result.split("\n", 1)
                        exitcode_part = parts[0]  # "exitcode: X (status)"
                        try:
                            exit_code = int(exitcode_part.split(":")[1].split("(")[0].strip())
                            span.set_attribute("ag2.code_execution.exit_code", exit_code)
                            if exit_code != 0:
                                span.set_attribute("error.type", "CodeExecutionError")
                        except (ValueError, IndexError):
                            pass

                        if len(parts) > 1:
                            output = parts[1].replace("Code output: ", "", 1).strip()
                            # Truncate output if too long
                            if len(output) > 4096:
                                output = output[:4096] + "... (truncated)"
                            span.set_attribute("ag2.code_execution.output", output)

                    return is_final, result

            generate_code_execution_reply_traced.__otel_wrapped__ = True
            # Update _reply_func_list to use the traced function
            agent._reply_func_list[original_code_exec_index]["reply_func"] = generate_code_execution_reply_traced

    return agent


def instrument_create_or_get_executor(agent: Agent, *, instrumentator: Callable[[Agent], Agent]) -> Agent:
    # Instrument `_create_or_get_executor` to auto-instrument dynamically created executors
    if hasattr(agent, "_create_or_get_executor") and not hasattr(agent._create_or_get_executor, "__otel_wrapped__"):
        old_create_or_get_executor = agent._create_or_get_executor

        @contextmanager
        def create_or_get_executor_traced(
            executor_kwargs: dict[str, Any] | None = None,
            tools: Any = None,
            agent_name: str = "executor",
            agent_human_input_mode: str = "NEVER",
        ) -> Generator[Agent, None, None]:
            with old_create_or_get_executor(
                executor_kwargs=executor_kwargs,
                tools=tools,
                agent_name=agent_name,
                agent_human_input_mode=agent_human_input_mode,
            ) as executor:
                # Instrument the dynamically created executor
                instrumentator(executor)
                yield executor

        create_or_get_executor_traced.__otel_wrapped__ = True
        agent._create_or_get_executor = create_or_get_executor_traced

    return agent
