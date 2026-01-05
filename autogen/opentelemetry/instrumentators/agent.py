# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import functools
import json
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.context import Context
from opentelemetry.trace import Tracer

from autogen.agentchat import Agent
from autogen.doc_utils import export_module
from autogen.io import IOStream
from autogen.opentelemetry.consts import SpanType
from autogen.opentelemetry.utils import (
    TRACE_PROPAGATOR,
    aggregate_usage,
    get_model_name,
    get_provider_name,
    messages_to_otel,
    reply_to_otel_message,
)


@export_module("autogen.opentelemetry")
def instrument_agent(agent: Agent, tracer: Tracer) -> Agent:
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
        tracer: The OpenTelemetry tracer to use for creating spans.

    Returns:
        The instrumented agent instance (same object, modified in place).

    Usage:
        from autogen.opentelemetry import setup_instrumentation, instrument_agent

        tracer = setup_instrumentation("my-service")
        agent = AssistantAgent("assistant")
        instrument_agent(agent, tracer)
    """
    # Instrument `a_generate_reply` as an invoke_agent span
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

    agent.a_generate_reply = a_generate_traced_reply

    # Instrument `generate_reply` (sync) as an invoke_agent span
    if hasattr(agent, "generate_reply"):
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

        agent.generate_reply = generate_traced_reply

    # Instrument `a_generate_oai_reply` to propagate context to executor thread
    # Critical because a_generate_oai_reply uses run_in_executor which
    # creates a new thread that doesn't inherit OpenTelemetry context so
    # will create new traces instead of being a child span.
    if hasattr(agent, "a_generate_oai_reply"):

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

    # Instrument `a_initiate_chat` as a conversation span
    if hasattr(agent, "a_initiate_chat"):
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

        agent.a_initiate_chat = a_initiate_traced_chat

    # Instrument `initiate_chat` (sync) as a conversation span
    if hasattr(agent, "initiate_chat"):
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

        agent.initiate_chat = initiate_traced_chat

    # Instrument `a_resume` as a resumed conversation span
    if hasattr(agent, "a_resume"):
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

        agent.a_resume = a_resume_traced

    # Instrument `run_chat` as a conversation span (GroupChatManager, sync)
    if hasattr(agent, "run_chat"):
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

        agent.run_chat = run_chat_traced

    # Instrument `a_run_chat` as a conversation span (GroupChatManager)
    if hasattr(agent, "a_run_chat"):
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

        agent.a_run_chat = a_run_chat_traced

    # Instrument `a_generate_remote_reply` as a remote invoke_agent span
    if hasattr(agent, "a_generate_remote_reply"):
        old_httpx_client_factory = agent._httpx_client_factory

        def httpx_client_factory_traced():
            httpx_client = old_httpx_client_factory()
            TRACE_PROPAGATOR.inject(httpx_client.headers)
            return httpx_client

        agent._httpx_client_factory = httpx_client_factory_traced

        # Find the original reply func in _reply_func_list
        original_reply_func = None
        original_reply_func_index = None
        for i, reply_func_tuple in enumerate(agent._reply_func_list):
            if getattr(reply_func_tuple["reply_func"], "__name__", None) == "a_generate_remote_reply":
                original_reply_func = reply_func_tuple["reply_func"]
                original_reply_func_index = i
                break

        if original_reply_func is not None:
            # Create traced wrapper that accepts self as first arg (like unbound method)
            async def a_generate_remote_reply_traced(
                self_agent: Any,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                with tracer.start_as_current_span(f"invoke_agent {self_agent.name}") as span:
                    span.set_attribute("ag2.span.type", SpanType.AGENT.value)
                    span.set_attribute("gen_ai.operation.name", "invoke_agent")
                    span.set_attribute("gen_ai.agent.name", self_agent.name)
                    span.set_attribute("gen_ai.agent.remote", True)
                    if hasattr(self_agent, "url") and self_agent.url:
                        span.set_attribute("server.address", self_agent.url)
                    return await original_reply_func(self_agent, *args, **kwargs)

            # Update _reply_func_list to use the traced function
            agent._reply_func_list[original_reply_func_index]["reply_func"] = a_generate_remote_reply_traced

    # Instrument `execute_function` as an execute_tool span
    if hasattr(agent, "execute_function"):
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

        agent.execute_function = execute_function_traced

    # Instrument `a_execute_function` as an async execute_tool span
    if hasattr(agent, "a_execute_function"):
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

        agent.a_execute_function = a_execute_function_traced

    # Instrument `_create_or_get_executor` to auto-instrument dynamically created executors
    if hasattr(agent, "_create_or_get_executor"):
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
                instrument_agent(executor, tracer)
                yield executor

        agent._create_or_get_executor = create_or_get_executor_traced

    # Instrument `get_human_input` as an await_human_input span
    if hasattr(agent, "get_human_input"):
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

        agent.get_human_input = get_human_input_traced

    # Instrument `a_get_human_input` as an async await_human_input span
    if hasattr(agent, "a_get_human_input"):
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

        agent.a_get_human_input = a_get_human_input_traced

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

            # Update _reply_func_list to use the traced function
            agent._reply_func_list[original_code_exec_index]["reply_func"] = generate_code_execution_reply_traced

    return agent
