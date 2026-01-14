# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for AnthropicCompletionsClient (V2) with real API calls.

These tests require:
- ANTHROPIC_API_KEY environment variable set
- Anthropic account with access to Claude Sonnet 4.5+ models
- pytest markers: @pytest.mark.anthropic, @pytest.mark.integration
- @run_for_optional_imports decorator to handle optional dependencies

Run with:
    pytest test/llm_clients/test_anthropic_v2_integration.py -m "anthropic and integration"
"""

import json
import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from autogen import AssistantAgent, UserProxyAgent
from autogen.import_utils import run_for_optional_imports
from test.credentials import Credentials


@pytest.fixture
def anthropic_v2_llm_config(credentials_anthropic_claude_sonnet: Credentials) -> dict:
    """Create LLM config for Anthropic V2 client."""
    return {
        "config_list": [
            {
                "model": "claude-sonnet-4-5",
                "api_key": credentials_anthropic_claude_sonnet.api_key,
                "api_type": "anthropic_v2",
            }
        ],
    }


@pytest.fixture
def anthropic_v2_llm_config_vision(credentials_anthropic_claude_sonnet: Credentials) -> dict:
    """Create LLM config for Anthropic V2 client with vision model."""
    return {
        "config_list": [
            {
                "api_type": "anthropic_v2",
                "model": "claude-3-5-haiku-20241022",  # Vision-capable model
                "api_key": credentials_anthropic_claude_sonnet.api_key,
            }
        ],
        "temperature": 0.3,
    }


class TestAnthropicV2StructuredOutputs:
    """Test structured outputs with Pydantic models (from notebook Example 1)."""

    @pytest.mark.anthropic
    @pytest.mark.integration
    @run_for_optional_imports("anthropic", "anthropic")
    def test_structured_output_math_reasoning(self, anthropic_v2_llm_config):
        """Test structured output with math reasoning using Pydantic models."""
        # Define the structured output schema
        class Step(BaseModel):
            """A single step in mathematical reasoning."""

            explanation: str
            output: str

        class MathReasoning(BaseModel):
            """Structured output for mathematical problem solving."""

            steps: list[Step]
            final_answer: str

            def format(self) -> str:
                """Format the response for display."""
                steps_output = "\n".join(
                    f"Step {i + 1}: {step.explanation}\n  Output: {step.output}" for i, step in enumerate(self.steps)
                )
                return f"{steps_output}\n\nFinal Answer: {self.final_answer}"

        # Configure LLM with structured output
        llm_config = anthropic_v2_llm_config.copy()
        llm_config["config_list"][0]["response_format"] = MathReasoning

        # Create agents
        user_proxy = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
        )

        math_assistant = AssistantAgent(
            name="MathAssistant",
            system_message="You are a math tutor. Solve problems step by step.",
            llm_config=llm_config,
        )

        # Ask the assistant to solve a math problem
        chat_result = user_proxy.run(
            math_assistant,
            message="Solve the equation: 3x + 7 = 22",
            max_turns=1,
        )

        # Verify chat result
        assert chat_result is not None
        assert chat_result.chat_history is not None
        assert len(chat_result.chat_history) > 0

        # Verify the response contains structured output
        # The response should be formatted by the MathReasoning.format() method
        last_message = chat_result.chat_history[-1]
        assert "content" in last_message or "text" in str(last_message)

        # Verify cost tracking
        assert chat_result.cost is not None
        assert isinstance(chat_result.cost, dict)
        # Cost should have entries for the model used
        total_cost = sum(
            model_usage.get("cost", 0)
            for usage_type in chat_result.cost.values()
            if isinstance(usage_type, dict)
            for model_usage in usage_type.values()
            if isinstance(model_usage, dict)
        )
        assert total_cost >= 0


class TestAnthropicV2StrictToolUse:
    """Test strict tool use for type-safe function calls (from notebook Example 2)."""

    @pytest.mark.anthropic
    @pytest.mark.integration
    @run_for_optional_imports("anthropic", "anthropic")
    def test_strict_tool_use_weather(self, anthropic_v2_llm_config):
        """Test strict tool use with weather API."""
        # Define a tool function
        def get_weather(location: str, unit: str = "celsius") -> str:
            """Get the weather for a location.

            Args:
                location: The city and state, e.g. San Francisco, CA
                unit: Temperature unit (celsius or fahrenheit)
            """
            return f"Weather in {location}: 22°{unit.upper()[0]}, partly cloudy"

        # Configure LLM with strict tool
        llm_config_strict = anthropic_v2_llm_config.copy()
        llm_config_strict["functions"] = [
            {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "strict": True,  # Enable strict schema validation
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            }
        ]

        # Create agents
        weather_assistant = AssistantAgent(
            name="WeatherAssistant",
            system_message="You help users get weather information. Use the get_weather function.",
            llm_config=llm_config_strict,
        )

        user_proxy = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False,
        )

        # Register function on both agents
        weather_assistant.register_function({"get_weather": get_weather})
        user_proxy.register_function({"get_weather": get_weather})

        # Query the weather
        chat_result = user_proxy.initiate_chat(
            weather_assistant,
            message="What's the weather in Boston, MA?",
            max_turns=2,
        )

        # Verify chat result
        assert chat_result is not None
        assert chat_result.chat_history is not None
        assert len(chat_result.chat_history) > 0

        # Verify tool call was made with correct types
        tool_call_found = False
        for message in chat_result.chat_history:
            if message.get("tool_calls"):
                tool_call = message["tool_calls"][0]
                args = json.loads(tool_call["function"]["arguments"])
                assert tool_call["function"]["name"] == "get_weather"
                assert "location" in args
                assert isinstance(args["location"], str)
                assert args["location"] == "Boston, MA" or "Boston" in args["location"]
                # If unit is provided, it should be a valid enum value
                if "unit" in args:
                    assert args["unit"] in ["celsius", "fahrenheit"]
                tool_call_found = True
                break

        assert tool_call_found, "Tool call should have been made"

        # Verify cost tracking
        assert chat_result.cost is not None
        total_cost = sum(
            model_usage.get("cost", 0)
            for usage_type in chat_result.cost.values()
            if isinstance(usage_type, dict)
            for model_usage in usage_type.values()
            if isinstance(model_usage, dict)
        )
        assert total_cost >= 0


class TestAnthropicV2CombinedFeatures:
    """Test combined structured outputs + strict tools (from notebook Example 3)."""

    @pytest.mark.anthropic
    @pytest.mark.integration
    @run_for_optional_imports("anthropic", "anthropic")
    def test_combined_structured_output_and_strict_tools(self, anthropic_v2_llm_config):
        """Test combined strict tools + structured output."""
        # Define calculator tool
        def calculate(operation: str, a: float, b: float) -> float:
            """Perform a calculation.

            Args:
                operation: The operation to perform (add, subtract, multiply, divide)
                a: First number
                b: Second number
            """
            if operation == "add":
                return a + b
            elif operation == "subtract":
                return a - b
            elif operation == "multiply":
                return a * b
            elif operation == "divide":
                return a / b if b != 0 else 0
            return 0

        # Result model for structured output
        class CalculationResult(BaseModel):
            """Structured output for calculation results."""

            problem: str
            steps: list[str]
            result: float
            verification: str

        # Configure with BOTH features
        llm_config_combined = anthropic_v2_llm_config.copy()
        llm_config_combined["config_list"][0]["response_format"] = CalculationResult
        llm_config_combined["functions"] = [
            {
                "name": "calculate",
                "description": "Perform arithmetic calculation",
                "strict": True,  # Enable strict mode
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["operation", "a", "b"],
                },
            }
        ]

        # Create agents
        calc_assistant = AssistantAgent(
            name="MathAssistant",
            system_message="You solve math problems using tools and provide structured results.",
            llm_config=llm_config_combined,
        )

        user_proxy = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
            code_execution_config=False,
        )

        # Register function on both agents
        calc_assistant.register_function({"calculate": calculate})
        user_proxy.register_function({"calculate": calculate})

        # Run calculation
        chat_result = user_proxy.run(
            calc_assistant,
            message="add 3 and 555",
            max_turns=2,
        )

        # Verify chat result
        assert chat_result is not None
        assert chat_result.chat_history is not None
        assert len(chat_result.chat_history) > 0

        # Verify tool call was made
        tool_call_found = False
        for message in chat_result.chat_history:
            if message.get("tool_calls"):
                tool_call = message["tool_calls"][0]
                args = json.loads(tool_call["function"]["arguments"])
                assert tool_call["function"]["name"] == "calculate"
                assert args["operation"] == "add"
                assert isinstance(args["a"], (int, float))
                assert isinstance(args["b"], (int, float))
                tool_call_found = True
                break

        assert tool_call_found, "Tool call should have been made"

        # Verify cost tracking
        assert chat_result.cost is not None
        total_cost = sum(
            model_usage.get("cost", 0)
            for usage_type in chat_result.cost.values()
            if isinstance(usage_type, dict)
            for model_usage in usage_type.values()
            if isinstance(model_usage, dict)
        )
        assert total_cost >= 0


class TestAnthropicV2Vision:
    """Test vision and image input capabilities (from notebook Examples 4-5)."""

    @pytest.mark.anthropic
    @pytest.mark.integration
    @run_for_optional_imports("anthropic", "anthropic")
    def test_vision_simple_image_description(self, anthropic_v2_llm_config_vision):
        """Test simple image description with vision model."""
        # Create vision assistant
        vision_assistant = AssistantAgent(
            name="VisionBot",
            llm_config=anthropic_v2_llm_config_vision,
            system_message="You are an AI assistant with vision capabilities. You can analyze images and provide detailed, accurate descriptions.",
        )

        # Create user proxy
        user_proxy_vision = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
        )

        # Test image URL
        IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/3/3b/BlkStdSchnauzer2.jpg"

        # Formal image input format
        message_with_image = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
            ],
        }

        # Initiate chat with image
        chat_result = user_proxy_vision.initiate_chat(
            vision_assistant, message=message_with_image, max_turns=1, summary_method="last_msg"
        )

        # Verify chat result
        assert chat_result is not None
        assert chat_result.summary is not None
        assert len(chat_result.summary) > 0

        # Verify the response mentions something about the image (dog, schnauzer, etc.)
        summary_lower = chat_result.summary.lower()
        assert any(
            keyword in summary_lower
            for keyword in ["dog", "schnauzer", "animal", "pet", "image", "photo", "picture"]
        )

        # Verify cost tracking
        assert chat_result.cost is not None
        total_cost = sum(
            model_usage.get("cost", 0)
            for usage_type in chat_result.cost.values()
            if isinstance(usage_type, dict)
            for model_usage in usage_type.values()
            if isinstance(model_usage, dict)
        )
        assert total_cost >= 0

    @pytest.mark.anthropic
    @pytest.mark.integration
    @run_for_optional_imports("anthropic", "anthropic")
    def test_vision_detailed_image_analysis(self, anthropic_v2_llm_config_vision):
        """Test detailed image analysis."""
        # Create vision assistant
        vision_assistant = AssistantAgent(
            name="VisionBot",
            llm_config=anthropic_v2_llm_config_vision,
            system_message="You are an AI assistant with vision capabilities. You can analyze images and provide detailed, accurate descriptions.",
        )

        # Create user proxy
        user_proxy_vision = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
        )

        # Test image URL
        IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/3/3b/BlkStdSchnauzer2.jpg"

        # Detailed analysis request
        detailed_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Analyze this image in detail. What breed is this dog? What are its characteristics?",
                },
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
            ],
        }

        # Initiate chat
        chat_result = user_proxy_vision.initiate_chat(
            vision_assistant,
            message=detailed_message,
            max_turns=1,
            clear_history=True,  # Start fresh conversation
        )

        # Verify chat result
        assert chat_result is not None
        assert chat_result.summary is not None
        assert len(chat_result.summary) > 0

        # Verify the response contains detailed analysis
        summary_lower = chat_result.summary.lower()
        # Should mention breed or characteristics
        assert any(
            keyword in summary_lower
            for keyword in [
                "schnauzer",
                "breed",
                "dog",
                "characteristic",
                "feature",
                "appearance",
                "black",
                "standard",
            ]
        )

        # Verify cost tracking
        assert chat_result.cost is not None
        total_cost = sum(
            model_usage.get("cost", 0)
            for usage_type in chat_result.cost.values()
            if isinstance(usage_type, dict)
            for model_usage in usage_type.values()
            if isinstance(model_usage, dict)
        )
        assert total_cost >= 0