# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass

import pytest

from ag2 import Agent
from ag2.config import OpenAIConfig, OpenAIResponsesConfig
from ag2.events import ImageInput
from ag2.tools import ImageGenerationTool


@pytest.fixture()
def openai_config() -> OpenAIConfig:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return OpenAIConfig(model="gpt-5.4-nano", api_key=api_key, temperature=0)


@pytest.fixture()
def openai_responses_config() -> OpenAIResponsesConfig:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return OpenAIResponsesConfig(model="gpt-5.4-nano", api_key=api_key)


@pytest.mark.openai
@pytest.mark.asyncio()
async def test_image_generation(openai_responses_config: OpenAIResponsesConfig) -> None:
    """ImageGenerationTool on the Responses API returns image bytes via reply.files."""
    agent = Agent(
        name="image_agent",
        prompt="You generate images when asked.",
        config=openai_responses_config,
        tools=[ImageGenerationTool(size="1024x1024", output_format="png")],
    )

    reply = await agent.ask("Generate an image of a single red circle on a white background.")

    assert reply.files, "expected at least one generated image in reply.files"
    assert len(reply.files[0].data) > 0


@pytest.mark.openai
@pytest.mark.asyncio()
@pytest.mark.timeout(120)  # two sequential image-generation calls (generate + edit)
async def test_image_editing(openai_responses_config: OpenAIResponsesConfig) -> None:
    """A generated image fed back in as ImageInput is edited and returned via reply.files."""
    agent = Agent(
        name="image_agent",
        prompt="You generate and edit images using the image generation tool.",
        config=openai_responses_config,
        tools=[ImageGenerationTool(size="1024x1024", output_format="png")],
    )

    generated = await agent.ask("Generate an image of a single blue circle on a white background.")
    assert generated.files, "expected an image to edit"
    source = generated.files[0]

    edited = await agent.ask(
        "Add a small red square next to the circle. Keep everything else the same.",
        ImageInput(data=source.data, media_type=source.metadata.get("media_type", "image/png")),
    )

    assert edited.files, "expected an edited image in reply.files"
    assert len(edited.files[0].data) > 0


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


@pytest.mark.openai
@pytest.mark.asyncio()
async def test_structured_output_primitive(openai_config: OpenAIConfig) -> None:
    agent = Agent(
        name="math_agent",
        prompt="You are a math assistant. Return only the numeric answer.",
        config=openai_config,
        response_schema=int,
    )

    reply = await agent.ask("What is 15 * 7?")
    result = await reply.content()

    assert result == 105


@pytest.mark.openai
@pytest.mark.asyncio()
async def test_structured_output_dataclass(openai_config: OpenAIConfig) -> None:
    @dataclass
    class City:
        name: str
        country: str
        population: int

    agent = Agent(
        name="geo_agent",
        prompt="You are a geography assistant. Provide city information.",
        config=openai_config,
        response_schema=City,
    )

    reply = await agent.ask("Tell me about Paris, France. Population is approximately 2161000.")
    result = await reply.content()

    assert isinstance(result, City)
    assert result.name.lower() == "paris"
    assert result.country.lower() == "france"


@pytest.mark.openai
@pytest.mark.asyncio()
async def test_multi_turn(openai_config: OpenAIConfig) -> None:
    agent = Agent(
        name="memory_agent",
        prompt="You are a helpful assistant. Be concise.",
        config=openai_config,
    )

    reply = await agent.ask("My name is Alice.")
    assert reply.body is not None

    reply2 = await reply.ask("What is my name?")
    assert reply2.body is not None
    assert "Alice" in reply2.body


@pytest.mark.openai
@pytest.mark.asyncio()
async def test_multi_turn_after_empty_args_tool_call(openai_config: OpenAIConfig) -> None:
    """A follow-up question after an empty-args tool call must not crash."""

    def discover_agents(capability: str = "") -> str:
        """Discover available agents, optionally filtered by capability."""
        return "Available agents: researcher, writer, coder"

    agent = Agent(
        name="hub_agent",
        prompt="You have a discover_agents tool. Use it when asked about available agents. Be concise.",
        config=openai_config,
        tools=[discover_agents],
    )

    reply = await agent.ask("What agents are available?")
    assert reply.body is not None

    reply2 = await reply.ask("Tell me more about the researcher agent.")
    assert reply2.body is not None
