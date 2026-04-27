# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass

import pytest

from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig


@pytest.fixture()
def anthropic_config() -> AnthropicConfig:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return AnthropicConfig(model="claude-haiku-4-5", api_key=api_key, temperature=0)


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_system_prompt(anthropic_config: AnthropicConfig) -> None:
    agent = Agent(
        name="french_agent",
        prompt="You must always respond in French, no matter what language the user uses.",
        config=anthropic_config,
    )

    reply = await agent.ask("What is the capital of France?")

    assert reply.body is not None
    body_lower = reply.body.lower()
    assert any(word in body_lower for word in ["paris", "france", "est", "la", "le", "de"])


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_tool_use(anthropic_config: AnthropicConfig) -> None:
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"The weather in {city} is sunny and 22°C."

    agent = Agent(
        name="weather_agent",
        prompt="You are a weather assistant. Use the get_weather tool to answer weather questions.",
        config=anthropic_config,
        tools=[get_weather],
    )

    reply = await agent.ask("What's the weather in Paris?")

    assert reply.body is not None
    assert "22" in reply.body or "sunny" in reply.body.lower()


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_structured_output_primitive(anthropic_config: AnthropicConfig) -> None:
    agent = Agent(
        name="math_agent",
        prompt="You are a math assistant. Return only the numeric answer.",
        config=anthropic_config,
        response_schema=int,
    )

    reply = await agent.ask("What is 15 * 7?")
    result = await reply.content()

    assert result == 105


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_structured_output_dataclass(anthropic_config: AnthropicConfig) -> None:
    @dataclass
    class City:
        name: str
        country: str
        population: int

    agent = Agent(
        name="geo_agent",
        prompt="You are a geography assistant. Provide city information.",
        config=anthropic_config,
        response_schema=City,
    )

    reply = await agent.ask("Tell me about Paris, France. Population is approximately 2161000.")
    result = await reply.content()

    assert isinstance(result, City)
    assert result.name.lower() == "paris"
    assert result.country.lower() == "france"


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_multi_turn(anthropic_config: AnthropicConfig) -> None:
    agent = Agent(
        name="memory_agent",
        prompt="You are a helpful assistant. Be concise.",
        config=anthropic_config,
    )

    reply = await agent.ask("My name is Alice.")
    assert reply.body is not None

    reply2 = await reply.ask("What is my name?")
    assert reply2.body is not None
    assert "Alice" in reply2.body


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_multi_turn_after_empty_args_tool_call(anthropic_config: AnthropicConfig) -> None:
    """A follow-up question after an empty-args tool call must not crash."""

    def discover_agents(capability: str = "") -> str:
        """Discover available agents, optionally filtered by capability."""
        return "Available agents: researcher, writer, coder"

    agent = Agent(
        name="hub_agent",
        prompt="You have a discover_agents tool. Use it when asked about available agents. Be concise.",
        config=anthropic_config,
        tools=[discover_agents],
    )

    reply = await agent.ask("What agents are available?")
    assert reply.body is not None

    # This second turn re-serializes the first tool call — the original crash site
    reply2 = await reply.ask("Tell me more about the researcher agent.")
    assert reply2.body is not None
