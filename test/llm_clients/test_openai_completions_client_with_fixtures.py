# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for OpenAICompletionsClient using real captured API response fixtures.

These tests use actual OpenAI API responses that were captured and saved as JSON fixtures.
This provides more realistic testing than mocked responses while still not requiring API keys.

Run with:
    bash scripts/test-skip-llm.sh test/llm_clients/test_openai_completions_client_with_fixtures.py
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from autogen.llm_clients import OpenAICompletionsClient
from autogen.llm_clients.models import ToolCallContent, UnifiedResponse


def _load_fixture(fixture_name: str) -> dict[str, Any]:
    """Load a captured OpenAI response fixture from JSON file."""
    fixture_path = Path(__file__).parent / "fixtures" / "openai_responses" / f"{fixture_name}.json"
    with open(fixture_path) as f:
        return json.load(f)


def _create_mock_response_from_fixture(fixture_data: dict[str, Any]) -> Any:
    """Convert fixture dict to mock OpenAI response object with proper structure."""

    # Create mock usage
    usage_data = fixture_data.get("usage", {})
    mock_usage = Mock()
    mock_usage.prompt_tokens = usage_data.get("prompt_tokens", 0)
    mock_usage.completion_tokens = usage_data.get("completion_tokens", 0)
    mock_usage.total_tokens = usage_data.get("total_tokens", 0)

    # Create mock messages
    mock_choices = []
    for choice_data in fixture_data.get("choices", []):
        message_data = choice_data.get("message", {})

        # Create mock message - only include attributes that exist in fixture
        # Use spec to limit which attributes can be accessed
        available_attrs = ["role", "content", "refusal", "tool_calls"]

        mock_message = Mock(spec=available_attrs)
        mock_message.role = message_data.get("role", "assistant")
        mock_message.content = message_data.get("content")
        mock_message.refusal = message_data.get("refusal")

        # Handle tool calls
        tool_calls_data = message_data.get("tool_calls")
        if tool_calls_data:
            mock_tool_calls = []
            for tc in tool_calls_data:
                mock_tc = Mock()
                mock_tc.id = tc.get("id")
                mock_tc.type = tc.get("type")

                # Create function mock
                mock_function = Mock()
                mock_function.name = tc["function"]["name"]
                mock_function.arguments = tc["function"]["arguments"]
                mock_tc.function = mock_function

                mock_tool_calls.append(mock_tc)
            mock_message.tool_calls = mock_tool_calls
        else:
            mock_message.tool_calls = None

        # Create mock choice
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = choice_data.get("finish_reason", "stop")
        mock_choice.index = choice_data.get("index", 0)

        mock_choices.append(mock_choice)

    # Create mock response
    mock_response = Mock()
    mock_response.id = fixture_data.get("id")
    mock_response.model = fixture_data.get("model")
    mock_response.created = fixture_data.get("created")
    mock_response.object = fixture_data.get("object")
    mock_response.choices = mock_choices
    mock_response.usage = mock_usage
    mock_response.system_fingerprint = fixture_data.get("system_fingerprint")
    mock_response.service_tier = fixture_data.get("service_tier")

    return mock_response


@pytest.fixture
def mock_openai_client():
    """Create mock OpenAI client."""
    with patch("autogen.llm_clients.openai_completions_client.OpenAI") as mock_openai_class:
        mock_client_instance = Mock()
        mock_openai_class.return_value = mock_client_instance
        yield mock_client_instance


class TestOpenAICompletionsClientWithFixtures:
    """Test OpenAICompletionsClient using real captured API response fixtures."""

    def test_simple_text_response(self, mock_openai_client):
        """Test handling simple text response from real API fixture."""
        # Load real API response fixture
        fixture = _load_fixture("simple_text_response")
        mock_response = _create_mock_response_from_fixture(fixture)

        # Setup client
        client = OpenAICompletionsClient(api_key="test-key")
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Execute
        response = client.create({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "What is 2 + 2?"}]})

        # Verify response structure
        assert isinstance(response, UnifiedResponse)
        assert response.id == fixture["id"]
        assert response.model == fixture["model"]
        assert response.provider == "openai"

        # Verify content
        assert len(response.messages) == 1
        assert response.text == "4"

        # Verify usage
        assert response.usage["prompt_tokens"] == fixture["usage"]["prompt_tokens"]
        assert response.usage["completion_tokens"] == fixture["usage"]["completion_tokens"]
        assert response.usage["total_tokens"] == fixture["usage"]["total_tokens"]

        # Verify cost was calculated
        assert response.cost > 0

    def test_multimodal_vision_response(self, mock_openai_client):
        """Test handling multimodal vision response from real API fixture."""
        fixture = _load_fixture("multimodal_vision_response")
        mock_response = _create_mock_response_from_fixture(fixture)

        client = OpenAICompletionsClient(api_key="test-key")
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Execute
        response = client.create({
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What animal is in this image?"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                    ],
                }
            ],
        })

        # Verify
        assert isinstance(response, UnifiedResponse)
        assert response.id == fixture["id"]
        assert len(response.messages) == 1

        # Verify text content was extracted
        assert len(response.text) > 0
        assert response.text.lower() in ["dog", "schnauzer", "the image shows a dog."]

    def test_tool_call_response(self, mock_openai_client):
        """Test handling tool call response from real API fixture."""
        fixture = _load_fixture("tool_call_response")
        mock_response = _create_mock_response_from_fixture(fixture)

        client = OpenAICompletionsClient(api_key="test-key")
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Execute
        response = client.create({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        })

        # Verify response structure
        assert isinstance(response, UnifiedResponse)
        assert response.finish_reason == "tool_calls"

        # Verify tool calls were extracted
        tool_calls = response.messages[0].get_tool_calls()
        assert len(tool_calls) == 1
        assert isinstance(tool_calls[0], ToolCallContent)

        # Verify tool call details from real fixture
        tool_call = tool_calls[0]
        assert tool_call.name == "get_weather"
        assert "location" in tool_call.arguments or "San Francisco" in tool_call.arguments

    def test_multi_turn_context_response(self, mock_openai_client):
        """Test handling multi-turn conversation from real API fixture."""
        fixture = _load_fixture("multi_turn_context_response")
        mock_response = _create_mock_response_from_fixture(fixture)

        client = OpenAICompletionsClient(api_key="test-key")
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Execute
        response = client.create({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "My favorite color is blue."},
                {"role": "assistant", "content": "I'll remember that."},
                {"role": "user", "content": "What is my favorite color?"},
            ],
        })

        # Verify
        assert isinstance(response, UnifiedResponse)
        assert len(response.messages) == 1

        # Verify context was maintained (response should mention "blue")
        response_text = response.text.lower()
        assert "blue" in response_text

    def test_system_message_response(self, mock_openai_client):
        """Test handling system message from real API fixture."""
        fixture = _load_fixture("system_message_response")
        mock_response = _create_mock_response_from_fixture(fixture)

        client = OpenAICompletionsClient(api_key="test-key")
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Execute
        response = client.create({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a math tutor. Show your work."},
                {"role": "user", "content": "What is 15 + 27?"},
            ],
        })

        # Verify
        assert isinstance(response, UnifiedResponse)
        assert len(response.messages) == 1

        # Verify answer contains 42
        assert "42" in response.text

    def test_multiple_images_response(self, mock_openai_client):
        """Test handling multiple images from real API fixture."""
        fixture = _load_fixture("multiple_images_response")
        mock_response = _create_mock_response_from_fixture(fixture)

        client = OpenAICompletionsClient(api_key="test-key")
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Execute
        response = client.create({
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these two dogs."},
                        {"type": "image_url", "image_url": {"url": "https://example.com/dog1.jpg"}},
                        {"type": "image_url", "image_url": {"url": "https://example.com/dog2.jpg"}},
                    ],
                }
            ],
        })

        # Verify
        assert isinstance(response, UnifiedResponse)
        assert len(response.messages) == 1
        assert len(response.text) > 0


class TestOpenAICompletionsClientCostWithFixtures:
    """Test cost calculation with real fixture data."""

    def test_cost_calculation_with_real_usage(self, mock_openai_client):
        """Test that cost is calculated correctly from real usage data."""
        fixture = _load_fixture("simple_text_response")
        mock_response = _create_mock_response_from_fixture(fixture)

        client = OpenAICompletionsClient(api_key="test-key")
        client.client.chat.completions.create = Mock(return_value=mock_response)

        response = client.create({"model": "gpt-4o-mini", "messages": []})

        # Verify cost was calculated
        assert response.cost is not None
        assert response.cost > 0

        # Verify usage stats match fixture
        usage = client.get_usage(response)
        assert usage["prompt_tokens"] == fixture["usage"]["prompt_tokens"]
        assert usage["completion_tokens"] == fixture["usage"]["completion_tokens"]
        assert usage["total_tokens"] == fixture["usage"]["total_tokens"]
        assert usage["model"] == fixture["model"]


class TestOpenAICompletionsClientMessageRetrievalWithFixtures:
    """Test message retrieval with real fixtures."""

    def test_message_retrieval_from_text_fixture(self, mock_openai_client):
        """Test message retrieval from real text response."""
        fixture = _load_fixture("simple_text_response")
        mock_response = _create_mock_response_from_fixture(fixture)

        client = OpenAICompletionsClient(api_key="test-key")
        client.client.chat.completions.create = Mock(return_value=mock_response)

        response = client.create({"model": "gpt-4o-mini", "messages": []})
        messages = client.message_retrieval(response)

        assert len(messages) == 1
        assert messages[0] == fixture["choices"][0]["message"]["content"]

    def test_message_retrieval_from_tool_call_fixture(self, mock_openai_client):
        """Test message retrieval from tool call response."""
        fixture = _load_fixture("tool_call_response")
        mock_response = _create_mock_response_from_fixture(fixture)

        client = OpenAICompletionsClient(api_key="test-key")
        client.client.chat.completions.create = Mock(return_value=mock_response)

        response = client.create({"model": "gpt-4o-mini", "messages": []})
        messages = client.message_retrieval(response)

        # Tool call responses have no text content, only tool calls
        assert len(messages) == 1
        # Message should be empty string or representation of tool calls
        assert isinstance(messages[0], str)


class TestOpenAICompletionsClientV1CompatibleWithFixtures:
    """Test V1 compatibility with real fixtures."""

    def test_v1_compatible_with_real_response(self, mock_openai_client):
        """Test V1 compatible format with real fixture."""
        fixture = _load_fixture("simple_text_response")
        mock_response = _create_mock_response_from_fixture(fixture)

        client = OpenAICompletionsClient(api_key="test-key")
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Get V1 compatible response
        v1_response = client.create_v1_compatible({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Test"}],
        })

        # Verify V1 format structure
        assert isinstance(v1_response, dict)
        assert v1_response["id"] == fixture["id"]
        assert v1_response["model"] == fixture["model"]
        assert v1_response["object"] == "chat.completion"
        assert "choices" in v1_response
        assert "usage" in v1_response
        assert "cost" in v1_response

        # Verify choices structure
        assert len(v1_response["choices"]) == 1
        assert "message" in v1_response["choices"][0]
        assert "role" in v1_response["choices"][0]["message"]
        assert "content" in v1_response["choices"][0]["message"]
