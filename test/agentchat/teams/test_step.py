# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the step() function - Phase 1 of Teams.

These tests use real LLM calls (Anthropic Claude) to verify the step() function
works end-to-end with tool execution.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv

# Load API keys from ~/.env
load_dotenv(os.path.expanduser("~/.env"))

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat.teams._step import StepResult, step


def get_anthropic_config() -> LLMConfig:
    """Create an Anthropic LLMConfig for testing."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return LLMConfig({
        "model": "claude-sonnet-4-20250514",
        "api_key": api_key,
        "api_type": "anthropic",
        "temperature": 0.0,
        "max_tokens": 1024,
    })


class TestStepBasic:
    """Test basic step() functionality."""

    @pytest.mark.asyncio
    async def test_simple_text_response(self) -> None:
        """Test that step() returns a simple text response with no tools."""
        llm_config = get_anthropic_config()
        agent = ConversableAgent(
            name="assistant",
            llm_config=llm_config,
            system_message="You are a helpful assistant. Be concise.",
        )

        result = await step(agent, "What is 2 + 2? Reply with just the number.")

        assert isinstance(result, StepResult)
        assert "4" in result.content
        assert len(result.tool_calls_made) == 0
        assert len(result.messages) >= 2  # user msg + assistant response

    @pytest.mark.asyncio
    async def test_step_with_message_list(self) -> None:
        """Test that step() works with a list of messages."""
        llm_config = get_anthropic_config()
        agent = ConversableAgent(
            name="assistant",
            llm_config=llm_config,
            system_message="You are a helpful assistant. Be concise.",
        )

        messages = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice! How can I help you?"},
            {"role": "user", "content": "What is my name? Reply with just the name."},
        ]

        result = await step(agent, messages)

        assert isinstance(result, StepResult)
        assert "Alice" in result.content

    @pytest.mark.asyncio
    async def test_step_no_llm_raises(self) -> None:
        """Test that step() raises ValueError when agent has no LLM config."""
        agent = ConversableAgent(
            name="no_llm_agent",
            llm_config=False,
        )

        with pytest.raises(ValueError, match="no LLM client"):
            await step(agent, "Hello")


class TestStepWithTools:
    """Test step() with tool execution."""

    @pytest.mark.asyncio
    async def test_single_tool_call(self) -> None:
        """Test that step() executes a single tool call and returns final response."""
        llm_config = get_anthropic_config()
        agent = ConversableAgent(
            name="weather_assistant",
            llm_config=llm_config,
            system_message="You are a weather assistant. Use the get_weather tool to answer questions. Be concise.",
        )

        @agent.register_for_llm(description="Get the current weather for a city")
        @agent.register_for_execution()
        def get_weather(city: str) -> str:
            """Get the current weather for a city."""
            return f"The weather in {city} is 22°C and sunny."

        result = await step(agent, "What's the weather in London?")

        assert isinstance(result, StepResult)
        assert len(result.tool_calls_made) >= 1
        assert result.tool_calls_made[0].name == "get_weather"
        assert result.tool_calls_made[0].arguments == {"city": "London"}
        assert result.tool_calls_made[0].is_success is True
        assert "22" in result.content or "sunny" in result.content.lower()

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self) -> None:
        """Test that step() handles multiple sequential tool calls."""
        llm_config = get_anthropic_config()
        agent = ConversableAgent(
            name="math_assistant",
            llm_config=llm_config,
            system_message=(
                "You are a math assistant. Use the provided tools to compute results. "
                "You must use the tools, don't compute in your head. Be concise."
            ),
        )

        @agent.register_for_llm(description="Add two numbers together")
        @agent.register_for_execution()
        def add(a: int, b: int) -> str:
            """Add two numbers."""
            return str(a + b)

        @agent.register_for_llm(description="Multiply two numbers together")
        @agent.register_for_execution()
        def multiply(a: int, b: int) -> str:
            """Multiply two numbers."""
            return str(a * b)

        result = await step(agent, "What is (3 + 4) * 5? Use the tools step by step.")

        assert isinstance(result, StepResult)
        assert len(result.tool_calls_made) >= 2
        tool_names = [tc.name for tc in result.tool_calls_made]
        assert "add" in tool_names
        assert "multiply" in tool_names
        assert "35" in result.content

    @pytest.mark.asyncio
    async def test_tool_error_handling(self) -> None:
        """Test that step() handles tool execution errors gracefully."""
        llm_config = get_anthropic_config()
        agent = ConversableAgent(
            name="error_assistant",
            llm_config=llm_config,
            system_message="You are a helpful assistant. If a tool fails, explain the error. Be concise.",
        )

        @agent.register_for_llm(description="Divide two numbers")
        @agent.register_for_execution()
        def divide(a: float, b: float) -> str:
            """Divide a by b."""
            if b == 0:
                raise ValueError("Cannot divide by zero")
            return str(a / b)

        result = await step(agent, "What is 10 divided by 0?")

        assert isinstance(result, StepResult)
        # The tool call should have failed
        if result.tool_calls_made:
            failed_call = next((tc for tc in result.tool_calls_made if not tc.is_success), None)
            if failed_call:
                assert "zero" in failed_call.result.lower() or "error" in failed_call.result.lower()
        # The agent should mention the error in its response
        assert (
            "zero" in result.content.lower() or "error" in result.content.lower() or "cannot" in result.content.lower()
        )


class TestStepAsync:
    """Test async-specific behavior of step()."""

    @pytest.mark.asyncio
    async def test_async_tool_execution(self) -> None:
        """Test that step() works with async tools."""
        llm_config = get_anthropic_config()
        agent = ConversableAgent(
            name="async_assistant",
            llm_config=llm_config,
            system_message="You are a helpful assistant. Use the lookup tool when asked. Be concise.",
        )

        @agent.register_for_llm(description="Look up a value by key")
        @agent.register_for_execution()
        async def lookup(key: str) -> str:
            """Look up a value by key."""
            data = {"name": "Alice", "age": "30", "city": "London"}
            await asyncio.sleep(0.01)  # Simulate async work
            return data.get(key, f"Key '{key}' not found")

        result = await step(agent, "Look up the value for key 'name'.")

        assert isinstance(result, StepResult)
        assert len(result.tool_calls_made) >= 1
        assert result.tool_calls_made[0].name == "lookup"
        assert "Alice" in result.content

    @pytest.mark.asyncio
    async def test_max_turns_exceeded(self) -> None:
        """Test that step() respects max_turns limit.

        We set max_turns=1 so even a single tool call round will exceed it,
        since after executing the tool we'd need another LLM call to get the
        final text response.
        """
        llm_config = get_anthropic_config()
        agent = ConversableAgent(
            name="loop_assistant",
            llm_config=llm_config,
            system_message="You are a helpful assistant. Always use the ping tool before answering any question.",
        )

        @agent.register_for_llm(description="Ping the server - must be called before answering")
        @agent.register_for_execution()
        def ping() -> str:
            """Ping the server."""
            return "pong"

        # max_turns=1 means only 1 LLM call allowed.
        # If that call makes a tool call, we'd need a 2nd call to get text response,
        # which would exceed max_turns.
        with pytest.raises(RuntimeError, match="max_turns"):
            await step(agent, "Hello, please ping", max_turns=1)


class TestStepListContent:
    """Test step() handling of list content from Responses API."""

    @pytest.mark.asyncio
    async def test_list_content_extracted(self) -> None:
        """Content returned as list of dicts (Responses API format) is joined into text."""
        llm_config = LLMConfig({"model": "test-model", "api_key": "k", "api_type": "anthropic"})
        agent = ConversableAgent(name="test", llm_config=llm_config)

        # Build a mock response whose extract_text_or_completion_object returns
        # a dict with content as a list (Responses API format)
        mock_response = MagicMock()
        mock_response.usage = None
        mock_response.cost = 0

        extracted_dict = {
            "content": [
                {"type": "output_text", "text": "Hello from"},
                {"type": "output_text", "text": "Responses API"},
            ],
            "tool_calls": None,
        }

        agent.client = MagicMock()
        agent.client.create.return_value = mock_response
        agent.client.extract_text_or_completion_object.return_value = [extracted_dict]
        agent.client._config_list = []

        result = await step(agent, "test prompt")

        assert result.content == "Hello from\nResponses API"
        assert len(result.tool_calls_made) == 0

    @pytest.mark.asyncio
    async def test_list_content_empty_blocks(self) -> None:
        """List content with no text blocks yields empty string."""
        llm_config = LLMConfig({"model": "test-model", "api_key": "k", "api_type": "anthropic"})
        agent = ConversableAgent(name="test", llm_config=llm_config)

        mock_response = MagicMock()
        mock_response.usage = None
        mock_response.cost = 0

        extracted_dict = {
            "content": [{"type": "other", "data": "..."}],
            "tool_calls": None,
        }

        agent.client = MagicMock()
        agent.client.create.return_value = mock_response
        agent.client.extract_text_or_completion_object.return_value = [extracted_dict]
        agent.client._config_list = []

        result = await step(agent, "test prompt")

        assert result.content == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
