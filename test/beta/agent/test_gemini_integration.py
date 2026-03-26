# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from autogen.beta import Agent
from autogen.beta.config import GeminiConfig


@pytest.fixture()
def gemini_config() -> GeminiConfig:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")
    return GeminiConfig(model="gemini-2.5-flash", api_key=api_key, temperature=0)


@pytest.mark.gemini
@pytest.mark.asyncio()
async def test_basic_ask(gemini_config: GeminiConfig) -> None:
    agent = Agent(
        name="test_agent",
        prompt="You are a helpful assistant. Be concise.",
        config=gemini_config,
    )

    reply = await agent.ask("What is 2 + 2?")

    assert reply.body is not None
    assert "4" in reply.body


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
