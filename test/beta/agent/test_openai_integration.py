# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from autogen.beta import Agent
from autogen.beta.config import OpenAIConfig


@pytest.fixture()
def openai_config() -> OpenAIConfig:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return OpenAIConfig(model="gpt-4o-mini", api_key=api_key, temperature=0)


@pytest.mark.openai
@pytest.mark.asyncio()
async def test_basic_ask(openai_config: OpenAIConfig) -> None:
    agent = Agent(
        name="test_agent",
        prompt="You are a helpful assistant. Be concise.",
        config=openai_config,
    )

    reply = await agent.ask("What is 2 + 2?")

    assert reply.body is not None
    assert "4" in reply.body


@pytest.mark.openai
@pytest.mark.asyncio()
async def test_system_prompt(openai_config: OpenAIConfig) -> None:
    agent = Agent(
        name="french_agent",
        prompt="You must always respond in French, no matter what language the user uses.",
        config=openai_config,
    )

    reply = await agent.ask("What is the capital of France?")

    assert reply.body is not None
    # Check for common French words that would appear in a response about Paris
    body_lower = reply.body.lower()
    assert any(word in body_lower for word in ["paris", "france", "est", "la", "le", "de"])


@pytest.mark.openai
@pytest.mark.asyncio()
async def test_tool_use(openai_config: OpenAIConfig) -> None:
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"The weather in {city} is sunny and 22°C."

    agent = Agent(
        name="weather_agent",
        prompt="You are a weather assistant. Use the get_weather tool to answer weather questions.",
        config=openai_config,
        tools=[get_weather],
    )

    reply = await agent.ask("What's the weather in Paris?")

    assert reply.body is not None
    assert "22" in reply.body or "sunny" in reply.body.lower()
