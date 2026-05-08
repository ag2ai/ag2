# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Extensibility smoke: HITL hooks, Plugin, custom middleware, tool approval.

Real LLM calls via Gemini 3 Flash Preview.
"""

from typing import Annotated

import pytest

from autogen.beta import Agent, Context, Inject, Variable
from autogen.beta.events import HumanInputRequest, HumanMessage
from autogen.beta.middleware import BaseMiddleware, LoggingMiddleware
from autogen.beta.plugin import Plugin

pytestmark = pytest.mark.asyncio


async def test_custom_middleware_on_turn(provider_config) -> None:
    """User middleware can wrap on_turn to inspect events and replies."""
    events_observed: list[str] = []

    class TraceMiddleware(BaseMiddleware):
        async def on_turn(self, call_next, event, context):
            events_observed.append(f"start:{type(event).__name__}")
            result = await call_next(event, context)
            events_observed.append(f"end:{type(result).__name__}")
            return result

    agent = Agent(
        "traced",
        config=provider_config,
        middleware=[lambda event, context: TraceMiddleware(event, context)],
    )
    await agent.ask("Say 'ok'.")

    assert events_observed == ["start:ModelRequest", "end:ModelResponse"]


async def test_custom_middleware_on_llm_call(provider_config) -> None:
    """User middleware can wrap on_llm_call and modify/observe each LLM round-trip."""
    llm_call_count = 0

    class CountingMiddleware(BaseMiddleware):
        async def on_llm_call(self, call_next, events, context):
            nonlocal llm_call_count
            llm_call_count += 1
            return await call_next(events, context)

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent = Agent(
        "counter",
        prompt="Use the add tool then report the result.",
        config=provider_config,
        tools=[add],
        middleware=[lambda event, context: CountingMiddleware(event, context)],
    )
    await agent.ask("What is 7 + 8? Use the tool.")

    # 1 LLM call for tool dispatch, 1 for final reply with result
    assert llm_call_count >= 2


async def test_logging_middleware_doesnt_crash(provider_config) -> None:
    """The built-in LoggingMiddleware runs cleanly end-to-end."""
    agent = Agent(
        "logged",
        config=provider_config,
        middleware=[LoggingMiddleware()],
    )
    reply = await agent.ask("Say 'ok'.")
    assert reply.body is not None


async def test_add_middleware_at_runtime(provider_config) -> None:
    """add_middleware appends the inner-most middleware after construction."""
    seen = []

    class T(BaseMiddleware):
        async def on_turn(self, call_next, event, context):
            seen.append("T")
            return await call_next(event, context)

    agent = Agent("rt", config=provider_config)
    agent.add_middleware(lambda event, context: T(event, context))
    await agent.ask("Hi.")

    assert "T" in seen


async def test_insert_middleware_outermost(provider_config) -> None:
    """insert_middleware places the new middleware as the outermost wrapper."""
    order: list[str] = []

    class A(BaseMiddleware):
        async def on_turn(self, call_next, event, context):
            order.append("A-pre")
            r = await call_next(event, context)
            order.append("A-post")
            return r

    class B(BaseMiddleware):
        async def on_turn(self, call_next, event, context):
            order.append("B-pre")
            r = await call_next(event, context)
            order.append("B-post")
            return r

    agent = Agent(
        "ordering",
        config=provider_config,
        middleware=[lambda e, c: A(e, c)],
    )
    agent.insert_middleware(lambda e, c: B(e, c))  # B becomes outermost
    await agent.ask("Hi.")

    assert order == ["B-pre", "A-pre", "A-post", "B-post"]


async def test_hitl_hook_provides_input(provider_config) -> None:
    """A registered hitl_hook is invoked when a tool calls ``ctx.input(...)``."""
    inputs_requested: list[str] = []

    async def auto_approve(event: HumanInputRequest) -> HumanMessage:
        inputs_requested.append(event.content)
        return HumanMessage("yes")

    async def confirm_then_act(action: str, ctx: Context) -> str:
        """Ask the human to confirm before performing an action."""
        answer = await ctx.input(f"Approve action '{action}'? (yes/no)", timeout=5.0)
        if str(answer).lower().startswith("y"):
            return f"performed: {action}"
        return "cancelled"

    agent = Agent(
        "hitl",
        prompt="Use confirm_then_act for any user-requested action. Report the result.",
        config=provider_config,
        tools=[confirm_then_act],
        hitl_hook=auto_approve,
    )
    reply = await agent.ask("Use confirm_then_act with action='delete_file' and report what happened.")
    assert reply.body is not None
    assert len(inputs_requested) >= 1
    assert "approve" in inputs_requested[0].lower()
    assert "performed" in reply.body.lower() or "delete_file" in reply.body.lower()


async def test_plugin_contributes_tool_and_prompt(provider_config) -> None:
    """Plugin can add tools, prompts, dependencies, and observers in one bundle."""

    def secret_word() -> str:
        """Return the secret word."""
        return "moonlight"

    plugin = Plugin(
        prompt="When asked for the secret word, call the secret_word tool.",
        tools=[secret_word],
    )

    agent = Agent("plugged", config=provider_config, plugins=[plugin])
    reply = await agent.ask("What is the secret word?")
    assert reply.body is not None
    assert "moonlight" in reply.body.lower()


async def test_plugin_with_dependencies_and_variables(provider_config) -> None:
    """Plugin propagates dependencies and variables onto the agent."""

    captured = {}

    def get_user(user: Annotated[str, Inject()]) -> str:
        """Look up the current user."""
        captured["user"] = user
        return f"user is {user}"

    def get_role(role: Annotated[str, Variable()]) -> str:
        """Look up the current role from variables."""
        captured["role"] = role
        return f"role is {role}"

    plugin = Plugin(
        prompt="Use get_user and get_role to look up identity. Report both.",
        tools=[get_user, get_role],
        dependencies={"user": "alice"},
        variables={"role": "admin"},
    )

    agent = Agent("identity", config=provider_config, plugins=[plugin])
    reply = await agent.ask("Who am I and what's my role?")
    assert reply.body is not None
    assert captured["user"] == "alice"
    assert captured["role"] == "admin"


async def test_multiple_plugins_compose(provider_config) -> None:
    """Multiple plugins compose without overwriting each other's contributions."""

    def tool_a() -> str:
        """Returns 'a'."""
        return "result_a"

    def tool_b() -> str:
        """Returns 'b'."""
        return "result_b"

    plugin_a = Plugin(prompt="If asked for A, call tool_a.", tools=[tool_a])
    plugin_b = Plugin(prompt="If asked for B, call tool_b.", tools=[tool_b])

    agent = Agent("composed", config=provider_config, plugins=[plugin_a, plugin_b])
    reply = await agent.ask("Get me both A and B by calling the tools.")
    assert reply.body is not None
    body = reply.body.lower()
    assert "result_a" in body
    assert "result_b" in body
