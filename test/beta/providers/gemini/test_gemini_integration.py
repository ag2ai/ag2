# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass

import pytest

from autogen.beta import Agent
from autogen.beta.config import GeminiConfig
from autogen.beta.config.gemini.events import GeminiToolCallEvent
from autogen.beta.streams.redis.serializer import Serializer, deserialize, serialize


@pytest.fixture()
def gemini_config() -> GeminiConfig:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")
    return GeminiConfig(model="gemini-3.1-flash-lite-preview", api_key=api_key, temperature=0)


@pytest.mark.gemini
@pytest.mark.asyncio()
async def test_system_prompt(gemini_config: GeminiConfig) -> None:
    agent = Agent(
        name="french_agent",
        prompt="You must always respond in French, no matter what language the user uses.",
        config=gemini_config,
    )

    reply = await agent.ask("What is the capital of France?")

    assert reply.body is not None
    body_lower = reply.body.lower()
    assert any(word in body_lower for word in ["paris", "france", "est", "la", "le", "de"])


@pytest.mark.gemini
@pytest.mark.asyncio()
async def test_tool_use(gemini_config: GeminiConfig) -> None:
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"The weather in {city} is sunny and 22°C."

    agent = Agent(
        name="weather_agent",
        prompt="You are a weather assistant. Use the get_weather tool to answer weather questions.",
        config=gemini_config,
        tools=[get_weather],
    )

    reply = await agent.ask("What's the weather in Paris?")

    assert reply.body is not None
    assert "22" in reply.body or "sunny" in reply.body.lower()


@pytest.mark.gemini
@pytest.mark.asyncio()
async def test_structured_output_primitive(gemini_config: GeminiConfig) -> None:
    agent = Agent(
        name="math_agent",
        prompt="You are a math assistant. Return only the numeric answer.",
        config=gemini_config,
        response_schema=int,
    )

    reply = await agent.ask("What is 15 * 7?")
    result = await reply.content()

    assert result == 105


@pytest.mark.gemini
@pytest.mark.asyncio()
async def test_structured_output_dataclass(gemini_config: GeminiConfig) -> None:
    @dataclass
    class City:
        name: str
        country: str
        population: int

    agent = Agent(
        name="geo_agent",
        prompt="You are a geography assistant. Provide city information.",
        config=gemini_config,
        response_schema=City,
    )

    reply = await agent.ask("Tell me about Paris, France. Population is approximately 2161000.")
    result = await reply.content()

    assert isinstance(result, City)
    assert result.name.lower() == "paris"
    assert result.country.lower() == "france"


@pytest.mark.gemini
@pytest.mark.asyncio()
async def test_multi_turn(gemini_config: GeminiConfig) -> None:
    agent = Agent(
        name="memory_agent",
        prompt="You are a helpful assistant. Be concise.",
        config=gemini_config,
    )

    reply = await agent.ask("My name is Alice.")
    assert reply.body is not None

    reply2 = await reply.ask("What is my name?")
    assert reply2.body is not None
    assert "Alice" in reply2.body


@pytest.mark.gemini
@pytest.mark.asyncio()
async def test_multi_turn_after_empty_args_tool_call(gemini_config: GeminiConfig) -> None:
    """A follow-up question after an empty-args tool call must not crash."""

    def discover_agents(capability: str = "") -> str:
        """Discover available agents, optionally filtered by capability."""
        return "Available agents: researcher, writer, coder"

    agent = Agent(
        name="hub_agent",
        prompt="You have a discover_agents tool. Use it when asked about available agents. Be concise.",
        config=gemini_config,
        tools=[discover_agents],
    )

    reply = await agent.ask("What agents are available?")
    assert reply.body is not None

    reply2 = await reply.ask("Tell me more about the researcher agent.")
    assert reply2.body is not None


@pytest.mark.gemini
@pytest.mark.asyncio()
async def test_thinking_level_low_reports_thinking_tokens(gemini_config: GeminiConfig) -> None:
    """thinking_level='low' is accepted by the SDK on Gemini 3 thinking models
    and ``thoughts_token_count`` is surfaced in ``Usage.thinking_tokens``."""
    config = gemini_config.copy(thinking_level="low")

    agent = Agent(
        name="thinker",
        prompt="You are a careful reasoner. Be concise.",
        config=config,
    )

    reply = await agent.ask("What is 17 * 23? Think briefly, then answer with just the number.")

    assert reply.body is not None
    usage = reply.response.usage
    assert usage.thinking_tokens is not None and usage.thinking_tokens > 0


@pytest.mark.gemini
@pytest.mark.asyncio()
async def test_thinking_budget_reports_thinking_tokens(gemini_config: GeminiConfig) -> None:
    """thinking_budget shorthand is accepted on Gemini 2.5 thinking models."""
    config = gemini_config.copy(model="gemini-2.5-flash", thinking_budget=512)

    agent = Agent(
        name="budgeted-thinker",
        prompt="You are a careful reasoner. Be concise.",
        config=config,
    )

    reply = await agent.ask("What is 17 * 23? Think briefly, then answer with just the number.")

    assert reply.body is not None
    usage = reply.response.usage
    assert usage.thinking_tokens is not None and usage.thinking_tokens > 0


@pytest.mark.gemini
@pytest.mark.asyncio()
async def test_history_round_trip_preserves_thought_signature(gemini_config: GeminiConfig) -> None:
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"The weather in {city} is sunny and 22°C."

    agent = Agent(
        name="weather_agent",
        prompt="You are a weather assistant. Use the get_weather tool to answer weather questions. Be concise.",
        config=gemini_config,
        tools=[get_weather],
    )

    reply1 = await agent.ask("What's the weather in Paris?")
    assert reply1.body is not None

    events = list(await reply1.history.get_events())
    round_tripped = [deserialize(serialize(e, Serializer.JSON), Serializer.JSON) for e in events]

    for ev in round_tripped:
        if isinstance(ev, GeminiToolCallEvent) and ev.thought_signature is not None:
            assert isinstance(ev.thought_signature, bytes), (
                f"thought_signature corrupted to {type(ev.thought_signature).__name__}: {ev.thought_signature!r}"
            )

    await reply1.history.replace(round_tripped)

    reply2 = await reply1.ask("What about London?")
    assert reply2.body is not None
