# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from opentelemetry.trace import Tracer

from autogen import Agent
from autogen.opentelemetry.consts import SpanType
from autogen.opentelemetry.utils import TRACE_PROPAGATOR


def instrument_remote_reply(agent: Agent, *, tracer: Tracer) -> Agent:
    # Instrument `a_generate_remote_reply` as a remote invoke_agent span
    if hasattr(agent, "a_generate_remote_reply"):
        # Find the original reply func in _reply_func_list
        original_reply_func = None
        original_reply_func_index = None
        for i, reply_func_tuple in enumerate(agent._reply_func_list):
            if getattr(reply_func_tuple["reply_func"], "__name__", None) == "a_generate_remote_reply":
                original_reply_func = reply_func_tuple["reply_func"]
                original_reply_func_index = i
                break

        if original_reply_func and not hasattr(original_reply_func, "__otel_wrapped__"):
            old_httpx_client_factory = agent._httpx_client_factory

            def httpx_client_factory_traced():
                httpx_client = old_httpx_client_factory()
                TRACE_PROPAGATOR.inject(httpx_client.headers)
                return httpx_client

            agent._httpx_client_factory = httpx_client_factory_traced

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

            a_generate_remote_reply_traced.__otel_wrapped__ = True
            # Update _reply_func_list to use the traced function
            agent._reply_func_list[original_reply_func_index]["reply_func"] = a_generate_remote_reply_traced

    return agent
