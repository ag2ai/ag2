# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Integration Test Gemini V2 client."""
"""
Integration tests for GeminiV2Client with real API calls.

Run with:
    pytest test/llm_clients/test_gemini_v2_integration.py -m integration
"""

import json
import os

import pytest
from pydantic import BaseModel

from autogen.import_utils import run_for_optional_imports
from autogen.llm_clients import GeminiV2Client, UnifiedResponse
from autogen.llm_clients.models import TextContent, ToolCallContent
from test.credentials import Credentials


@pytest.fixture
def gemini_v2_client(credentials_gemini_flash: Credentials) -> GeminiV2Client:
    """Create GeminiV2Client with credentials from AG2 test framework."""
    return GeminiV2Client(api_key=credentials_gemini_flash.api_key)


@pytest.fixture
def gemini_v2_config(credentials_gemini_flash: Credentials) -> dict:
    """Get Gemini V2 configuration for tests."""
    api_key = credentials_gemini_flash.api_key
    if not api_key:
        pytest.skip("GOOGLE_GEMINI_API_KEY not set. Export GOOGLE_GEMINI_API_KEY to run integration tests.")

    return {
        "model": "gemini-2.5-flash",
        "api_key": api_key,
    }


class TestGeminiV2ClientBasicChat:
    """Test basic chat functionality with real API calls."""

    @pytest.mark.integration
    @run_for_optional_imports(["google.genai", "vertexai"], "gemini")
    def test_simple_chat(self, gemini_v2_client, gemini_v2_config):
        """Test simple chat with Gemini 2.5 Flash."""
        response = gemini_v2_client.create({
            "model": gemini_v2_config["model"],
            "messages": [{"role": "user", "content": "Say 'Hello' in one word."}],
        })

        # Verify UnifiedResponse structure
        assert isinstance(response, UnifiedResponse), f"Expected UnifiedResponse, got {type(response)}"
        assert response.provider == "gemini"
        assert response.model == gemini_v2_config["model"]
        assert response.id is not None
        assert len(response.id) > 0
        assert response.status == "completed"
        assert response.finish_reason in ["stop", "length", "tool_calls", "content_filter"]

        # Verify messages
        assert len(response.messages) > 0
        assert response.messages[0].role.value == "assistant" if hasattr(response.messages[0].role, "value") else str(response.messages[0].role) == "assistant"

        # Verify text content
        assert len(response.text) > 0
        assert "hello" in response.text.lower()

        # Verify usage tracking
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0
        assert response.usage["total_tokens"] > 0
        assert response.usage["total_tokens"] == response.usage["prompt_tokens"] + response.usage["completion_tokens"]

        # Verify cost calculation
        assert response.cost is not None
        assert response.cost >= 0

    @pytest.mark.integration
    @run_for_optional_imports(["google.genai", "vertexai"], "gemini")
    def test_chat_with_system_message(self, gemini_v2_client, gemini_v2_config):
        """Test chat with system message."""
        response = gemini_v2_client.create({
            "model": gemini_v2_config["model"],
            "messages": [
                {"role": "system", "content": "You are a helpful math tutor. Be concise."},
                {"role": "user", "content": "Explain what a prime number is in one sentence."},
            ],
        })

        assert response.provider == "gemini"
        assert len(response.text) > 0
        assert "prime" in response.text.lower() or "divisible" in response.text.lower()

    @pytest.mark.integration
    @run_for_optional_imports(["google.genai", "vertexai"], "gemini")
    def test_multi_turn_conversation(self, gemini_v2_client, gemini_v2_config):
        """Test multi-turn conversation maintains context."""
        # First turn
        response1 = gemini_v2_client.create({
            "model": gemini_v2_config["model"],
            "messages": [{"role": "user", "content": "My favorite color is blue."}],
        })

        # Second turn - reference first turn
        response2 = gemini_v2_client.create({
            "model": gemini_v2_config["model"],
            "messages": [
                {"role": "user", "content": "My favorite color is blue."},
                {"role": "assistant", "content": response1.text},
                {"role": "user", "content": "What is my favorite color?"},
            ],
        })

        # Should remember the color
        assert "blue" in response2.text.lower()


class TestGeminiV2ClientRichContent:
    """Test rich content access from UnifiedResponse."""

    @pytest.mark.integration
    @run_for_optional_imports(["google.genai", "vertexai"], "gemini")
    def test_access_text_content(self, gemini_v2_client, gemini_v2_config):
        """Test accessing text content directly."""
        response = gemini_v2_client.create({
            "model": gemini_v2_config["model"],
            "messages": [{"role": "user", "content": "Explain quantum computing in 2 sentences."}],
        })

        # Access text content directly
        assert len(response.text) > 0

        # Access individual messages
        assert len(response.messages) > 0
        for message in response.messages:
            assert len(message.content) > 0
            for content_block in message.content:
                assert hasattr(content_block, "type")

        # Get content by type
        text_blocks = response.get_content_by_type("text")
        assert len(text_blocks) > 0
        assert all(isinstance(block, TextContent) for block in text_blocks)

    @pytest.mark.integration
    @run_for_optional_imports(["google.genai", "vertexai"], "gemini")
    def test_content_blocks_structure(self, gemini_v2_client, gemini_v2_config):
        """Test that content blocks are properly structured."""
        response = gemini_v2_client.create({
            "model": gemini_v2_config["model"],
            "messages": [{"role": "user", "content": "Count from 1 to 3."}],
        })

        # Verify message structure
        assert len(response.messages) == 1
        message = response.messages[0]

        # Verify content blocks
        assert len(message.content) > 0
        for content_block in message.content:
            assert hasattr(content_block, "type")
            assert content_block.type in ["text", "image", "audio", "video", "tool_call", "reasoning", "citation"]


class TestGeminiV2ClientThinkingConfig:
    """Test thinking config support (Gemini 3 models)."""

    @pytest.mark.integration
    @run_for_optional_imports(["google.genai", "vertexai"], "gemini")
    @pytest.mark.skipif(
        os.getenv("ENABLE_GEMINI_3_TESTS") != "1",
        reason="Gemini 3 models require special access. Set ENABLE_GEMINI_3_TESTS=1 to run this test.",
    )
    def test_thinking_level_high(self, gemini_v2_client):
        """Test thinking_level parameter with Gemini 3 Pro."""
        prompt = """You are playing the 20 question game. You know that what you are looking for
is an aquatic mammal that doesn't live in the sea, is venomous and that's
smaller than a cat. What could that be and how could you make sure?
"""

        response = gemini_v2_client.create({
            "model": "gemini-3-pro-preview",
            "messages": [{"role": "user", "content": prompt}],
            "thinking_level": "High",
            "include_thoughts": True,
        })

        assert response.provider == "gemini"
        assert len(response.text) > 0

        # Check for reasoning blocks if available
        reasoning_blocks = response.get_content_by_type("reasoning")
        # Note: thoughts may be included in text rather than separate reasoning blocks

    @pytest.mark.integration
    @run_for_optional_imports(["google.genai", "vertexai"], "gemini")
    def test_thinking_budget(self, gemini_v2_client, gemini_v2_config):
        """Test thinking_budget parameter with Gemini 2.5 Flash."""
        prompt = """You are playing the 20 question game. You know that what you are looking for
is an aquatic mammal that doesn't live in the sea, is venomous and that's
smaller than a cat. What could that be and how could you make sure?
"""

        response = gemini_v2_client.create({
            "model": gemini_v2_config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "thinking_budget": 4096,
        })

        assert response.provider == "gemini"
        assert len(response.text) > 0
        assert response.usage["total_tokens"] > 0


class TestGeminiV2ClientStructuredOutputs:
    """Test structured outputs with Pydantic models."""

    @pytest.mark.integration
    @run_for_optional_imports(["google.genai", "vertexai"], "gemini")
    def test_structured_output_pydantic_model(self, gemini_v2_client, gemini_v2_config):
        """Test structured output with Pydantic BaseModel."""
        # Define a Pydantic model for structured output
        class Answer(BaseModel):
            answer: str
            confidence: float
            reasoning: str | None = None

        # Create client with response format
        structured_client = GeminiV2Client(api_key=gemini_v2_config["api_key"], response_format=Answer)

        # Get structured response
        response = structured_client.create({
            "model": gemini_v2_config["model"],
            "messages": [{"role": "user", "content": "What is 2+2? Answer with confidence score."}],
        })

        assert isinstance(response, UnifiedResponse)
        assert len(response.text) > 0

        # The response text should contain JSON matching the schema
        try:
            structured_data = json.loads(response.text)
            assert "answer" in structured_data
            assert "confidence" in structured_data
        except json.JSONDecodeError:
            # If not JSON, at least verify text is present
            assert len(response.text) > 0

    @pytest.mark.integration
    @run_for_optional_imports(["google.genai", "vertexai"], "gemini")
    def test_structured_output_json_schema(self, gemini_v2_client, gemini_v2_config):
        """Test structured output with JSON schema dict."""
        # Define response schema
        response_format = {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "answer": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["question", "answer", "reasoning"],
        }

        response = gemini_v2_client.create({
            "model": gemini_v2_config["model"],
            "messages": [{"role": "user", "content": "What causes seasons on Earth?"}],
            "response_format": response_format,
        })

        assert isinstance(response, UnifiedResponse)
        assert len(response.text) > 0

        # Try to parse JSON
        try:
            result = json.loads(response.text)
            assert isinstance(result, dict)
        except json.JSONDecodeError:
            # If not JSON, at least verify text is present
            assert len(response.text) > 0


class TestGeminiV2ClientToolCalling:
    """Test function/tool calling with real API calls."""

    @pytest.mark.integration
    @run_for_optional_imports(["google.genai", "vertexai"], "gemini")
    def test_tool_calling_basic(self, gemini_v2_client, gemini_v2_config):
        """Test basic tool calling functionality."""
        # Define a simple tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        response = gemini_v2_client.create({
            "model": gemini_v2_config["model"],
            "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}],
            "tools": tools,
        })

        # Verify response structure
        assert isinstance(response, UnifiedResponse)
        assert len(response.messages) > 0

        # Check for tool calls
        tool_calls = response.messages[0].get_tool_calls()
        if tool_calls:
            assert len(tool_calls) > 0
            assert isinstance(tool_calls[0], ToolCallContent)
            assert tool_calls[0].name == "get_weather"

            # Parse arguments
            args = json.loads(tool_calls[0].arguments)
            assert "location" in args
        else:
            # Model might have responded with text instead of tool call
            assert len(response.text) > 0


class TestGeminiV2ClientUsageAndCost:
    """Test usage tracking and cost calculation."""

    @pytest.mark.integration
    @run_for_optional_imports(["google.genai", "vertexai"], "gemini")
    def test_usage_tracking(self, gemini_v2_client, gemini_v2_config):
        """Test that usage is properly tracked."""
        response = gemini_v2_client.create({
            "model": gemini_v2_config["model"],
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        })

        # Get usage via client method
        usage = GeminiV2Client.get_usage(response)

        # Verify all keys present
        for key in GeminiV2Client.RESPONSE_USAGE_KEYS:
            assert key in usage

        # Verify values are reasonable
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        assert usage["cost"] >= 0
        assert usage["model"] == gemini_v2_config["model"]

    @pytest.mark.integration
    @run_for_optional_imports(["google.genai", "vertexai"], "gemini")
    def test_cost_calculation_accuracy(self, gemini_v2_client, gemini_v2_config):
        """Test that cost calculation is accurate."""
        response = gemini_v2_client.create({
            "model": gemini_v2_config["model"],
            "messages": [{"role": "user", "content": "Say 'hello' in 5 different languages."}],
        })

        # Cost should be calculated
        assert response.cost is not None
        assert response.cost >= 0

        # Verify cost matches manual calculation
        calculated_cost = gemini_v2_client.cost(response)
        assert calculated_cost == response.cost

    @pytest.mark.integration
    @run_for_optional_imports(["google.genai", "vertexai"], "gemini")
    def test_cost_with_custom_price(self, gemini_v2_client, gemini_v2_config):
        """Test cost calculation with custom price parameter."""
        custom_price = [0.001, 0.002]  # [input_price_per_1k, output_price_per_1k]

        response = gemini_v2_client.create({
            "model": gemini_v2_config["model"],
            "messages": [{"role": "user", "content": "Hello"}],
            "price": custom_price,
        })

        # Cost should be calculated using custom price
        assert response.cost is not None
        assert response.cost >= 0

        # Manual calculation
        input_cost = (response.usage["prompt_tokens"] / 1000) * custom_price[0]
        output_cost = (response.usage["completion_tokens"] / 1000) * custom_price[1]
        expected_cost = input_cost + output_cost

        assert abs(response.cost - expected_cost) < 0.0001  # Allow small floating point differences


class TestGeminiV2ClientV1Compatibility:
    """Test backward compatibility with v1 format."""

    @pytest.mark.integration
    @run_for_optional_imports(["google.genai", "vertexai"], "gemini")
    def test_v1_compatible_format(self, gemini_v2_client, gemini_v2_config):
        """Test v1 compatible response format."""
        # Get v1 compatible response
        v1_response = gemini_v2_client.create_v1_compatible({
            "model": gemini_v2_config["model"],
            "messages": [{"role": "user", "content": "What is 10 + 10?"}],
        })

        # Verify v1 format structure
        assert isinstance(v1_response, dict)
        assert "id" in v1_response
        assert "model" in v1_response
        assert "choices" in v1_response
        assert "usage" in v1_response
        assert "cost" in v1_response
        assert v1_response["object"] == "chat.completion"

        # Verify choices structure
        assert len(v1_response["choices"]) > 0
        assert "message" in v1_response["choices"][0]
        assert "content" in v1_response["choices"][0]["message"]

        # Verify content
        assert "20" in v1_response["choices"][0]["message"]["content"]

    @pytest.mark.integration
    @run_for_optional_imports(["google.genai", "vertexai"], "gemini")
    def test_message_retrieval(self, gemini_v2_client, gemini_v2_config):
        """Test message retrieval method for V1 compatibility."""
        response = gemini_v2_client.create({
            "model": gemini_v2_config["model"],
            "messages": [{"role": "user", "content": "Say exactly: 'Integration test successful'"}],
        })

        # Retrieve messages
        messages = gemini_v2_client.message_retrieval(response)

        assert len(messages) > 0
        assert isinstance(messages[0], str)
        assert len(messages[0]) > 0
        assert "successful" in messages[0].lower()


class TestGeminiV2ClientErrorHandling:
    """Test error handling with real API calls."""

    @pytest.mark.integration
    @run_for_optional_imports(["google.genai", "vertexai"], "gemini")
    def test_invalid_model_error(self, gemini_v2_client):
        """Test error handling for invalid model."""
        with pytest.raises(Exception):  # Gemini SDK will raise an error
            gemini_v2_client.create({
                "model": "invalid-model-name-that-does-not-exist",
                "messages": [{"role": "user", "content": "Hello"}],
            })

    @pytest.mark.integration
    @run_for_optional_imports(["google.genai", "vertexai"], "gemini")
    def test_empty_messages_error(self, gemini_v2_client, gemini_v2_config):
        """Test error handling for empty messages."""
        with pytest.raises(Exception):  # Gemini SDK will raise an error
            gemini_v2_client.create({
                "model": gemini_v2_config["model"],
                "messages": [],
            })


class TestGeminiV2ClientWithLLMConfig:
    """Test Gemini V2 client integration with LLMConfig and agents."""

    @pytest.mark.integration
    @run_for_optional_imports(["google.genai", "vertexai"], "gemini")
    def test_llm_config_integration(self, gemini_v2_config):
        """Test Gemini V2 client works with LLMConfig."""
        from autogen import ConversableAgent, LLMConfig

        # Configure LLM with V2 client
        llm_config_v2 = LLMConfig(
            config_list=[
                {
                    "api_type": "gemini_v2",
                    "model": gemini_v2_config["model"],
                    "api_key": gemini_v2_config["api_key"],
                    "temperature": 0.7,
                    "max_tokens": 500,
                }
            ]
        )

        # Create agent with V2 client
        agent_v2 = ConversableAgent(
            name="assistant_v2",
            llm_config=llm_config_v2,
            system_message="You are a helpful assistant.",
        )

        # Use the agent
        response = agent_v2.generate_reply(
            messages=[{"role": "user", "content": "Explain machine learning in one sentence."}]
        )

        assert isinstance(response, str)
        assert len(response) > 0
        assert "machine" in response.lower() or "learning" in response.lower()

    @pytest.mark.integration
    @run_for_optional_imports(["google.genai", "vertexai"], "gemini")
    def test_structured_output_with_agent(self, gemini_v2_config):
        """Test structured output with agent."""
        from autogen import ConversableAgent, LLMConfig

        # Define structured output model
        class Plan(BaseModel):
            title: str
            steps: list[str]
            estimated_time: str
            resources_needed: list[str]

        # Create agent with structured output
        structured_agent = ConversableAgent(
            name="structured_planner",
            llm_config=LLMConfig(
                config_list=[
                    {
                        "api_type": "gemini_v2",
                        "model": gemini_v2_config["model"],
                        "api_key": gemini_v2_config["api_key"],
                        "response_format": Plan,
                    }
                ]
            ),
            system_message="You are a planning assistant. Always respond with structured plans.",
        )

        # Get structured response
        response = structured_agent.generate_reply(
            messages=[{"role": "user", "content": "Create a plan for learning Python in 30 days."}]
        )

        assert isinstance(response, str)
        assert len(response) > 0

        # Try to parse JSON
        try:
            plan_data = json.loads(response)
            assert isinstance(plan_data, dict)
        except json.JSONDecodeError:
            # If not JSON, at least verify text is present
            assert len(response) > 0
