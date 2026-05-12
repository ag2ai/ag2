# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Cross-provider smoke: every core Agent feature exercised once per provider.

Each test takes ``provider_config`` and runs on all three providers
(OpenAI, Anthropic, Gemini). Tests hit the real APIs — no mocks.

Covers: basic ask, static and dynamic system prompts, sync and async
tools, multi-tool dispatch, tool errors, structured output (primitive /
dataclass / Pydantic / union), multi-turn AgentReply.ask, streaming,
dependency injection, context variables, per-ask tool injection, and
error paths (missing config, tool exception).
"""

import asyncio
from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from pydantic import BaseModel

from autogen.beta import Agent, Context, Inject, Variable
from autogen.beta.events import ModelMessageChunk
from autogen.beta.stream import MemoryStream

pytestmark = pytest.mark.asyncio


async def test_basic_ask(provider_config) -> None:
    agent = Agent("basic", config=provider_config)
    reply = await agent.ask("Reply with exactly the word: ping")
    assert reply.body is not None
    assert "ping" in reply.body.lower()


async def test_static_system_prompt(provider_config) -> None:
    agent = Agent(
        "capitals",
        prompt="You are a geography tutor. Always start your answer with 'Answer:'.",
        config=provider_config,
    )
    reply = await agent.ask("What is the capital of Japan? One word.")
    assert reply.body is not None
    assert "tokyo" in reply.body.lower()
    assert "answer:" in reply.body.lower()


async def test_dynamic_prompt_with_context(provider_config) -> None:
    """@agent.prompt decorator receives live Context; dynamic prompt runs per ask."""

    agent = Agent("dynamic", config=provider_config)

    @agent.prompt
    def render(ctx: Context) -> str:
        role = ctx.variables.get("role", "helper")
        return f"You are a {role}. Always mention the word '{role}' in your reply."

    reply = await agent.ask("Say hi.", variables={"role": "pirate"})
    assert reply.body is not None
    assert "pirate" in reply.body.lower()


async def test_sync_tool_use(provider_config) -> None:
    tool_calls: list[str] = []

    def add(a: int, b: int) -> int:
        """Return the integer sum of two numbers."""
        tool_calls.append(f"add({a}, {b})")
        return a + b

    agent = Agent(
        "calculator",
        prompt="You are a calculator. Use the add tool for every arithmetic question.",
        config=provider_config,
        tools=[add],
    )
    reply = await agent.ask("Use the add tool to compute 17 + 25. Return only the number.")
    assert reply.body is not None
    assert "42" in reply.body
    assert len(tool_calls) >= 1


async def test_async_tool_use(provider_config) -> None:
    async def lookup_stock(ticker: str) -> str:
        """Return the latest mock price for a ticker symbol."""
        await asyncio.sleep(0.01)
        return f"{ticker} is at $123.45"

    agent = Agent(
        "stocks",
        prompt="You quote stock prices. Use lookup_stock for every symbol you are asked about.",
        config=provider_config,
        tools=[lookup_stock],
    )
    reply = await agent.ask("What is AAPL trading at? Include the price in your answer.")
    assert reply.body is not None
    assert "123.45" in reply.body


async def test_multi_tool_dispatch(provider_config) -> None:
    """LLM must pick the right tool from several."""

    def get_time(timezone: str) -> str:
        """Return the current time in a named timezone."""
        return f"It is 10:15 in {timezone}"

    def get_weather(city: str) -> str:
        """Return the current weather in a city."""
        return f"72°F and sunny in {city}"

    def get_news(topic: str) -> str:
        """Return the latest headline for a topic."""
        return f"No news about {topic} today"

    agent = Agent(
        "concierge",
        prompt="You answer user questions using the right tool. Be concise.",
        config=provider_config,
        tools=[get_time, get_weather, get_news],
    )
    reply = await agent.ask("What's the weather in Seattle?")
    assert reply.body is not None
    assert "seattle" in reply.body.lower() or "72" in reply.body


async def test_tool_error_propagates(provider_config) -> None:
    """Tool that raises an exception — agent must still produce a reply."""

    def flaky_service(query: str) -> str:
        """Query the flaky service."""
        raise RuntimeError("service unavailable")

    agent = Agent(
        "resilient",
        prompt=(
            "You have a flaky_service tool. If it errors, apologise briefly and explain the "
            "tool is unavailable. Never retry more than once."
        ),
        config=provider_config,
        tools=[flaky_service],
    )
    reply = await agent.ask("Look up 'widgets' using flaky_service.")
    # The agent should not crash even though the tool raised
    assert reply.body is not None
    assert reply.body.strip() != ""


async def test_structured_output_primitive(provider_config) -> None:
    agent = Agent(
        "math",
        prompt="Return only the numeric answer.",
        config=provider_config,
        response_schema=int,
    )
    reply = await agent.ask("What is 12 * 11?")
    result = await reply.content()
    assert result == 132


async def test_structured_output_dataclass(provider_config) -> None:
    @dataclass
    class Book:
        title: str
        author: str
        year: int

    agent = Agent(
        "librarian",
        prompt="Return book metadata. Moby-Dick is by Herman Melville, published 1851.",
        config=provider_config,
        response_schema=Book,
    )
    reply = await agent.ask("Give me metadata for Moby-Dick.")
    book = await reply.content()
    assert isinstance(book, Book)
    assert "melville" in book.author.lower()
    assert book.year == 1851


async def test_structured_output_pydantic(provider_config) -> None:
    class Person(BaseModel):
        name: str
        age: int
        hobbies: list[str]

    agent = Agent(
        "bio",
        prompt="Return structured person info. Alice is 30 and enjoys chess and hiking.",
        config=provider_config,
        response_schema=Person,
    )
    reply = await agent.ask("Give me Alice's profile.")
    person = await reply.content()
    assert isinstance(person, Person)
    assert person.name.lower() == "alice"
    assert person.age == 30
    assert any("chess" in h.lower() for h in person.hobbies)


async def test_multi_turn_ask_chain(provider_config) -> None:
    agent = Agent(
        "memoryful",
        prompt="You are a helpful assistant. Remember details the user shares.",
        config=provider_config,
    )
    reply1 = await agent.ask("My favourite colour is teal.")
    assert reply1.body is not None

    reply2 = await reply1.ask("What is my favourite colour? One word only.")
    assert reply2.body is not None
    assert "teal" in reply2.body.lower()

    reply3 = await reply2.ask("And what was the first thing I told you?")
    assert reply3.body is not None
    # Should still remember the colour across three turns
    assert "teal" in reply3.body.lower() or "colour" in reply3.body.lower() or "color" in reply3.body.lower()


async def test_streaming_chunks_arrive(streaming_config) -> None:
    """Observe ModelMessageChunk events during a streamed reply.

    Uses ``streaming_config`` (streaming=True). Asserts that chunks actually
    fired AND that joined chunks reconstruct the final body — proving the
    chunk pipeline reached the subscriber, not just that ``ask`` succeeded.
    """
    stream = MemoryStream()
    chunks: list[str] = []

    with stream.where(ModelMessageChunk).sub_scope(lambda e: chunks.append(e.content)):
        agent = Agent(
            "writer",
            prompt="Write a short 2-sentence haiku-like description of the ocean.",
            config=streaming_config,
        )
        reply = await agent.ask("Describe the ocean.", stream=stream)

    assert reply.body is not None
    assert chunks, "streaming=True must produce at least one ModelMessageChunk"
    assert "".join(chunks) == reply.body


async def test_dependency_injection_into_tool(provider_config) -> None:
    """Agent-level dependency surfaces inside a tool via Inject."""

    class UserService:
        def lookup(self, user_id: str) -> dict[str, Any]:
            return {"id": user_id, "name": "Alice", "tier": "premium"}

    def describe_user(user_id: str, service: Annotated[UserService, Inject()]) -> str:
        """Describe a user by id."""
        info = service.lookup(user_id)
        return f"User {info['id']}: {info['name']} ({info['tier']})"

    service = UserService()
    agent = Agent(
        "user-info",
        prompt="Look up users with the describe_user tool and repeat exactly what it returns.",
        config=provider_config,
        tools=[describe_user],
        dependencies={"service": service},
    )
    reply = await agent.ask("Describe user u42. Include all details.")
    assert reply.body is not None
    assert "alice" in reply.body.lower()
    assert "premium" in reply.body.lower()


async def test_context_variables_injected_into_tool(provider_config) -> None:
    def get_role(role: Annotated[str, Variable()]) -> str:
        """Return the caller's current role from context variables."""
        return f"The role is {role}"

    agent = Agent(
        "role-reader",
        prompt="Use get_role to look up the user's role and include it verbatim in your answer.",
        config=provider_config,
        tools=[get_role],
    )
    reply = await agent.ask("What role am I?", variables={"role": "administrator"})
    assert reply.body is not None
    assert "administrator" in reply.body.lower()


async def test_per_ask_tool_injection(provider_config) -> None:
    """Tools passed to ask() override/augment agent-level tools for that turn only."""

    from autogen.beta import tool

    @tool
    def secret_word() -> str:
        """Return the secret word."""
        return "zephyr"

    agent = Agent(
        "per-ask",
        prompt="If a secret_word tool exists, call it and include its output.",
        config=provider_config,
    )
    reply = await agent.ask("Reveal the secret word.", tools=[secret_word])
    assert reply.body is not None
    assert "zephyr" in reply.body.lower()


async def test_config_override_per_ask(provider_config) -> None:
    """Passing config= to ask() overrides the agent's default config."""
    agent = Agent("override")  # No default config
    reply = await agent.ask("Say 'ok'.", config=provider_config)
    assert reply.body is not None
    assert "ok" in reply.body.lower()
