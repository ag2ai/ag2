# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for OpenAICompletionsClient with real API calls.

These tests require:
- OPENAI_API_KEY environment variable set
- OpenAI account with access to Chat Completions API models
- pytest markers: @pytest.mark.openai
- @run_for_optional_imports decorator to handle optional dependencies

Run with:
    bash scripts/test-core-llm.sh test/llm_clients/test_openai_completions_client_integration.py
"""

import os

import pytest

from autogen.import_utils import run_for_optional_imports
from autogen.llm_clients import OpenAICompletionsClient
from test.credentials import Credentials


@pytest.fixture
def openai_completions_client(credentials_gpt_4o_mini: Credentials) -> OpenAICompletionsClient:
    """Create OpenAICompletionsClient with credentials from AG2 test framework."""
    return OpenAICompletionsClient(api_key=credentials_gpt_4o_mini.api_key)


class TestOpenAICompletionsClientBasicChat:
    """Test basic chat functionality with real API calls."""

    @pytest.mark.openai
    @run_for_optional_imports("openai", "openai")
    def test_simple_chat_gpt4(self, openai_completions_client):
        """Test simple chat with GPT-4."""
        response = openai_completions_client.create({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            "temperature": 0,
        })

        # Verify response structure
        assert response.provider == "openai"
        assert response.model.startswith("gpt-4")
        assert len(response.messages) > 0

        # Verify text content
        assert "4" in response.text

        # Verify usage tracking
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0
        assert response.usage["total_tokens"] > 0

        # Verify cost calculation
        assert response.cost is not None
        assert response.cost > 0

    @pytest.mark.openai
    @run_for_optional_imports("openai", "openai")
    def test_chat_with_system_message(self, openai_completions_client):
        """Test chat with system message."""
        response = openai_completions_client.create({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful math tutor. Be concise."},
                {"role": "user", "content": "Explain what a prime number is in one sentence."},
            ],
            "temperature": 0.7,
        })

        assert response.provider == "openai"
        assert len(response.text) > 0
        assert "prime" in response.text.lower()


class TestOpenAICompletionsClientReasoningModels:
    """Test reasoning models (o1, o3 series) with real API calls."""

    @pytest.mark.openai
    @run_for_optional_imports("openai", "openai")
    @pytest.mark.skipif(
        os.getenv("ENABLE_O1_TESTS") != "1",
        reason="o1 models require special access and are expensive. Set ENABLE_O1_TESTS=1 to run this test.",
    )
    def test_o1_model_with_reasoning(self, openai_completions_client):
        """Test o1 model extracts reasoning blocks.

        Note: This test is skipped by default because:
        - o1-preview requires special API access
        - o1 models are significantly more expensive than standard models
        - Not all OpenAI accounts have access to o1 models

        To enable: export ENABLE_O1_TESTS=1
        """
        response = openai_completions_client.create({
            "model": "o1-preview",
            "messages": [{"role": "user", "content": "What is 15 factorial? Show your reasoning."}],
        })

        # Verify response structure
        assert response.provider == "openai"
        assert response.model.startswith("o1")

        # Verify reasoning blocks are present (o1 models should provide reasoning)
        # Note: This depends on OpenAI's implementation
        # If reasoning is available, it should be extracted
        if len(response.reasoning) > 0:
            assert response.reasoning[0].reasoning is not None
            assert len(response.reasoning[0].reasoning) > 0

        # Verify text answer
        assert len(response.text) > 0

        # Verify cost (o1 models are more expensive)
        assert response.cost > 0


class TestOpenAICompletionsClientToolCalling:
    """Test function/tool calling with real API calls (from agentchat_oai_responses_api_tool_call.ipynb)."""

    @pytest.mark.openai
    @run_for_optional_imports("openai", "openai")
    def test_tool_calling_basic(self, openai_completions_client):
        """Test basic tool calling functionality."""
        # Define a simple tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add_numbers",
                    "description": "Add two numbers together",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]

        response = openai_completions_client.create({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Please add 42 and 58 using the add_numbers function."}],
            "tools": tools,
            "temperature": 0,
        })

        # Verify response has tool calls
        assert len(response.messages) > 0
        tool_calls = response.messages[0].get_tool_calls()

        # Should have requested to call the tool
        assert len(tool_calls) > 0
        assert tool_calls[0].name == "add_numbers"

        # Parse arguments to verify correct values
        import json

        args = json.loads(tool_calls[0].arguments)
        assert args.get("a") == 42 or args.get("a") == 58
        assert args.get("b") == 58 or args.get("b") == 42

    @pytest.mark.openai
    @run_for_optional_imports("openai", "openai")
    def test_tool_calling_with_result(self, openai_completions_client):
        """Test tool calling with result returned."""
        # First request: Ask to use tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        response = openai_completions_client.create({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What's the weather like in San Francisco?"}],
            "tools": tools,
            "temperature": 0,
        })

        # Should have tool call
        tool_calls = response.messages[0].get_tool_calls()
        assert len(tool_calls) > 0
        assert tool_calls[0].name == "get_weather"


class TestOpenAICompletionsClientStructuredOutput:
    """Test structured output with real API calls (from agentchat_oai_responses_api_structured_output.ipynb)."""

    @pytest.mark.openai
    @run_for_optional_imports("openai", "openai")
    def test_structured_output_json_schema(self, openai_completions_client):
        """Test structured output with JSON schema."""
        # Define response schema
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "qa_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["question", "answer", "reasoning"],
                    "additionalProperties": False,
                },
            },
        }

        response = openai_completions_client.create({
            "model": "gpt-4o-mini",  # gpt-4o models support JSON schema
            "messages": [
                {
                    "role": "system",
                    "content": "You are a Q&A bot. Always return a JSON object with question, answer, and reasoning fields.",
                },
                {"role": "user", "content": "What causes seasons on Earth?"},
            ],
            "response_format": response_format,
            "temperature": 0,
        })

        # Parse JSON response
        import json

        result = json.loads(response.text)

        # Verify structure
        assert "question" in result
        assert "answer" in result
        assert "reasoning" in result

        # Verify content
        assert "season" in result["answer"].lower() or "season" in result["reasoning"].lower()

    @pytest.mark.openai
    @run_for_optional_imports("openai", "openai")
    def test_structured_output_simple_json(self, openai_completions_client):
        """Test structured output with simple JSON mode."""
        response = openai_completions_client.create({
            "model": "gpt-4o-mini",  # gpt-4o models support JSON mode
            "messages": [
                {
                    "role": "system",
                    "content": "Return your response as a JSON object with 'topic' and 'explanation' fields.",
                },
                {"role": "user", "content": "Explain photosynthesis briefly."},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        })

        # Should be valid JSON
        import json

        result = json.loads(response.text)

        # Should have some structure
        assert isinstance(result, dict)
        assert len(result) > 0


class TestOpenAICompletionsClientImageInput:
    """Test image input/vision capabilities (from agentchat_oai_responses_image.ipynb)."""

    @pytest.mark.openai
    @run_for_optional_imports("openai", "openai")
    @pytest.mark.skipif(
        os.getenv("SKIP_VISION_TESTS") == "1", reason="Vision tests may not be available to all accounts"
    )
    def test_image_url_input(self, openai_completions_client):
        """Test image input with URL."""
        # Use a public domain image URL
        image_url = "https://upload.wikimedia.org/wikipedia/commons/3/3b/BlkStdSchnauzer2.jpg"

        response = openai_completions_client.create({
            "model": "gpt-4o-mini",  # gpt-4o models support vision
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What animal is in this image? Answer in one word."},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "temperature": 0,
        })

        # Verify response
        assert len(response.text) > 0
        assert "dog" in response.text.lower() or "schnauzer" in response.text.lower()

    @pytest.mark.openai
    @run_for_optional_imports("openai", "openai")
    @pytest.mark.skipif(
        os.getenv("SKIP_VISION_TESTS") == "1", reason="Vision tests may not be available to all accounts"
    )
    def test_image_description(self, openai_completions_client):
        """Test detailed image description."""
        image_url = "https://upload.wikimedia.org/wikipedia/commons/3/3b/BlkStdSchnauzer2.jpg"

        response = openai_completions_client.create({
            "model": "gpt-4o-mini",  # gpt-4o models support vision
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail."},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "temperature": 0.3,
        })

        # Should provide detailed description
        assert len(response.text) > 50
        assert response.cost > 0  # Vision tokens cost more


class TestOpenAICompletionsClientUsageAndCost:
    """Test usage tracking and cost calculation."""

    @pytest.mark.openai
    @run_for_optional_imports("openai", "openai")
    def test_usage_tracking(self, openai_completions_client):
        """Test that usage is properly tracked."""
        response = openai_completions_client.create({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
            "temperature": 0,
        })

        # Get usage via client method
        usage = openai_completions_client.get_usage(response)

        # Verify all keys present
        for key in openai_completions_client.RESPONSE_USAGE_KEYS:
            assert key in usage

        # Verify values are reasonable
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        assert usage["cost"] > 0
        assert usage["model"].startswith("gpt")

    @pytest.mark.openai
    @run_for_optional_imports("openai", "openai")
    def test_cost_calculation_accuracy(self, openai_completions_client):
        """Test that cost calculation is accurate."""
        response = openai_completions_client.create({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Say 'hello' in 5 different languages."}],
            "temperature": 0,
        })

        # Cost should be calculated
        assert response.cost is not None
        assert response.cost > 0

        # Verify cost matches manual calculation
        calculated_cost = openai_completions_client.cost(response)
        assert calculated_cost == response.cost

    @pytest.mark.openai
    @run_for_optional_imports("openai", "openai")
    def test_message_retrieval(self, openai_completions_client):
        """Test message retrieval method."""
        response = openai_completions_client.create({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Say exactly: 'Integration test successful'"}],
            "temperature": 0,
        })

        # Retrieve messages
        messages = openai_completions_client.message_retrieval(response)

        assert len(messages) > 0
        assert isinstance(messages[0], str)
        assert len(messages[0]) > 0


class TestOpenAICompletionsClientV1Compatibility:
    """Test backward compatibility with v1 format."""

    @pytest.mark.openai
    @run_for_optional_imports("openai", "openai")
    def test_v1_compatible_format(self, openai_completions_client):
        """Test v1 compatible response format."""
        # Get v1 compatible response
        v1_response = openai_completions_client.create_v1_compatible({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What is 10 + 10?"}],
            "temperature": 0,
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


class TestOpenAICompletionsClientErrorHandling:
    """Test error handling with real API calls."""

    @pytest.mark.openai
    @run_for_optional_imports("openai", "openai")
    def test_invalid_model_error(self, openai_completions_client):
        """Test error handling for invalid model."""
        with pytest.raises(Exception):  # OpenAI SDK will raise an error
            openai_completions_client.create({
                "model": "invalid-model-name-that-does-not-exist",
                "messages": [{"role": "user", "content": "Hello"}],
            })

    @pytest.mark.openai
    @run_for_optional_imports("openai", "openai")
    def test_empty_messages_error(self, openai_completions_client):
        """Test error handling for empty messages."""
        with pytest.raises(Exception):  # OpenAI SDK will raise an error
            openai_completions_client.create({
                "model": "gpt-4",
                "messages": [],
            })


class TestOpenAICompletionsClientMultiTurnConversation:
    """Test multi-turn conversations."""

    @pytest.mark.openai
    @run_for_optional_imports("openai", "openai")
    def test_multi_turn_conversation(self, openai_completions_client):
        """Test multi-turn conversation maintains context."""
        # First turn
        response1 = openai_completions_client.create({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "My favorite color is blue."}],
            "temperature": 0,
        })

        # Second turn - reference first turn
        response2 = openai_completions_client.create({
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "My favorite color is blue."},
                {"role": "assistant", "content": response1.text},
                {"role": "user", "content": "What is my favorite color?"},
            ],
            "temperature": 0,
        })

        # Should remember the color
        assert "blue" in response2.text.lower()

    @pytest.mark.openai
    @run_for_optional_imports("openai", "openai")
    def test_conversation_with_system_message(self, openai_completions_client):
        """Test conversation with persistent system message."""
        system_msg = "You are a pirate. Always respond in pirate speak."

        response = openai_completions_client.create({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": "What's your favorite thing about the ocean?"},
            ],
            "temperature": 0.7,
        })

        # Response should be in pirate speak (this is probabilistic but likely)
        text_lower = response.text.lower()
        pirate_words = ["arr", "ahoy", "matey", "ye", "aye", "sea", "ship"]
        has_pirate_speak = any(word in text_lower for word in pirate_words)

        # At least should mention ocean/sea
        assert has_pirate_speak or "ocean" in text_lower or "sea" in text_lower
