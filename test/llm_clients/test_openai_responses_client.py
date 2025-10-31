# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenAIResponsesClient."""

from unittest.mock import Mock, patch

import pytest

from autogen.llm_clients import OpenAIResponsesClient
from autogen.llm_clients.models import ReasoningContent, TextContent, ToolCallContent, UnifiedResponse


class MockOpenAIResponse:
    """Mock OpenAI API response."""

    def __init__(
        self,
        response_id="chatcmpl-test123",
        model="o1-preview",
        choices=None,
        usage=None,
        created=1234567890,
    ):
        self.id = response_id
        self.model = model
        self.choices = choices or []
        self.usage = usage
        self.created = created
        self.system_fingerprint = "fp_test"
        self.service_tier = None


class MockChoice:
    """Mock choice in OpenAI response."""

    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason
        self.index = 0


class MockMessage:
    """Mock message in OpenAI response."""

    def __init__(self, role="assistant", content=None, reasoning=None, tool_calls=None, name=None):
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.tool_calls = tool_calls
        self.name = name


class MockUsage:
    """Mock usage stats."""

    def __init__(self, prompt_tokens=50, completion_tokens=100):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class MockToolCall:
    """Mock tool call."""

    def __init__(self, call_id="call_123", name="get_weather", arguments='{"city":"SF"}'):
        self.id = call_id
        # Create a function object with name and arguments as attributes
        self.function = Mock()
        self.function.name = name
        self.function.arguments = arguments


@pytest.fixture
def mock_openai_client():
    """Create mock OpenAI client."""
    # Mock the OpenAI import
    with patch("autogen.llm_clients.openai_responses_client.OpenAI") as mock_openai_class:
        # Create a mock instance that will be returned when OpenAI() is called
        mock_client_instance = Mock()
        mock_openai_class.return_value = mock_client_instance
        yield mock_client_instance


class TestOpenAIResponsesClientCreation:
    """Test client initialization."""

    def test_create_client_with_api_key(self, mock_openai_client):
        """Test creating client with API key."""
        client = OpenAIResponsesClient(api_key="test-key")
        assert client is not None
        assert client.client is not None

    def test_create_client_with_base_url(self, mock_openai_client):
        """Test creating client with custom base URL."""
        client = OpenAIResponsesClient(api_key="test-key", base_url="https://custom.api.com")
        assert client is not None

    def test_client_has_required_methods(self, mock_openai_client):
        """Test that client has all ModelClientV2 methods."""
        client = OpenAIResponsesClient(api_key="test-key")
        assert hasattr(client, "create")
        assert hasattr(client, "create_v1_compatible")
        assert hasattr(client, "cost")
        assert hasattr(client, "get_usage")
        assert hasattr(client, "message_retrieval")
        assert hasattr(client, "RESPONSE_USAGE_KEYS")


class TestOpenAIResponsesClientCreate:
    """Test create() method."""

    def test_create_simple_response(self, mock_openai_client):
        """Test creating a simple text response."""
        # Setup mock
        mock_message = MockMessage(role="assistant", content="The answer is 42")
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())

        client = OpenAIResponsesClient(api_key="test-key")
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Test
        response = client.create({"model": "gpt-4", "messages": [{"role": "user", "content": "What is 40+2?"}]})

        # Verify
        assert isinstance(response, UnifiedResponse)
        assert response.id == "chatcmpl-test123"
        assert response.model == "o1-preview"
        assert response.provider == "openai"
        assert len(response.messages) == 1
        assert response.text == "The answer is 42"

    def test_create_response_with_reasoning(self, mock_openai_client):
        """Test creating response with reasoning blocks (o1/o3 models)."""
        # Setup mock with reasoning
        mock_message = MockMessage(
            role="assistant",
            content="The answer is 42",
            reasoning="Step 1: I need to add 40 and 2\nStep 2: 40 + 2 = 42",
        )
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(model="o1-preview", choices=[mock_choice], usage=MockUsage())

        client = OpenAIResponsesClient(api_key="test-key")
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Test
        response = client.create({"model": "o1-preview", "messages": [{"role": "user", "content": "What is 40+2?"}]})

        # Verify reasoning blocks are extracted
        assert len(response.reasoning) == 1
        assert isinstance(response.reasoning[0], ReasoningContent)
        assert "Step 1" in response.reasoning[0].reasoning
        assert "Step 2" in response.reasoning[0].reasoning

        # Verify text is also preserved
        assert len(response.messages[0].content) == 2  # reasoning + text
        text_blocks = [b for b in response.messages[0].content if isinstance(b, TextContent)]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "The answer is 42"

    def test_create_response_with_tool_calls(self, mock_openai_client):
        """Test creating response with tool calls."""
        # Setup mock with tool calls
        mock_tool_call = MockToolCall()
        mock_message = MockMessage(role="assistant", content=None, tool_calls=[mock_tool_call])
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())

        client = OpenAIResponsesClient(api_key="test-key")
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Test
        response = client.create({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Get weather"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        })

        # Verify tool calls are extracted
        tool_calls = response.messages[0].get_tool_calls()
        assert len(tool_calls) == 1
        assert isinstance(tool_calls[0], ToolCallContent)
        assert tool_calls[0].id == "call_123"
        assert tool_calls[0].name == "get_weather"

    def test_create_response_with_usage(self, mock_openai_client):
        """Test that usage information is properly extracted."""
        # Setup mock
        mock_message = MockMessage(content="Test response")
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(
            choices=[mock_choice], usage=MockUsage(prompt_tokens=100, completion_tokens=200)
        )

        client = OpenAIResponsesClient(api_key="test-key")
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Test
        response = client.create({"model": "gpt-4", "messages": []})

        # Verify usage
        assert response.usage["prompt_tokens"] == 100
        assert response.usage["completion_tokens"] == 200
        assert response.usage["total_tokens"] == 300


class TestOpenAIResponsesClientCost:
    """Test cost() method."""

    def test_cost_calculation_o1_preview(self, mock_openai_client):
        """Test cost calculation for o1-preview model."""
        client = OpenAIResponsesClient(api_key="test-key")

        # Create response with known usage
        mock_message = MockMessage(content="Test")
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(
            model="o1-preview", choices=[mock_choice], usage=MockUsage(prompt_tokens=1000, completion_tokens=500)
        )
        client.client.chat.completions.create = Mock(return_value=mock_response)

        response = client.create({"model": "o1-preview", "messages": []})

        # Verify cost calculation
        assert response.cost is not None
        assert response.cost > 0
        # o1-preview: $0.015/1K prompt, $0.060/1K completion
        expected = (1000 * 0.015 / 1000) + (500 * 0.060 / 1000)
        assert abs(response.cost - expected) < 0.001

    def test_cost_calculation_unknown_model(self, mock_openai_client):
        """Test cost calculation falls back for unknown models."""
        client = OpenAIResponsesClient(api_key="test-key")

        mock_message = MockMessage(content="Test")
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(
            model="unknown-model", choices=[mock_choice], usage=MockUsage(prompt_tokens=100, completion_tokens=100)
        )
        client.client.chat.completions.create = Mock(return_value=mock_response)

        response = client.create({"model": "unknown-model", "messages": []})

        # Should use default pricing (gpt-4 level)
        assert response.cost > 0


class TestOpenAIResponsesClientGetUsage:
    """Test get_usage() method."""

    def test_get_usage_returns_all_keys(self, mock_openai_client):
        """Test that get_usage() returns all required keys."""
        client = OpenAIResponsesClient(api_key="test-key")

        mock_message = MockMessage(content="Test")
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(
            choices=[mock_choice], usage=MockUsage(prompt_tokens=50, completion_tokens=75)
        )
        client.client.chat.completions.create = Mock(return_value=mock_response)

        response = client.create({"model": "gpt-4", "messages": []})
        usage = client.get_usage(response)

        # Verify all required keys
        for key in client.RESPONSE_USAGE_KEYS:
            assert key in usage

        assert usage["prompt_tokens"] == 50
        assert usage["completion_tokens"] == 75
        assert usage["total_tokens"] == 125
        assert usage["model"] == "o1-preview"
        assert usage["cost"] > 0


class TestOpenAIResponsesClientMessageRetrieval:
    """Test message_retrieval() method."""

    def test_message_retrieval_simple_text(self, mock_openai_client):
        """Test retrieving text from simple response."""
        client = OpenAIResponsesClient(api_key="test-key")

        mock_message = MockMessage(content="Hello world")
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())
        client.client.chat.completions.create = Mock(return_value=mock_response)

        response = client.create({"model": "gpt-4", "messages": []})
        messages = client.message_retrieval(response)

        assert len(messages) == 1
        assert messages[0] == "Hello world"

    def test_message_retrieval_with_reasoning(self, mock_openai_client):
        """Test retrieving text from response with reasoning."""
        client = OpenAIResponsesClient(api_key="test-key")

        mock_message = MockMessage(content="Answer: 42", reasoning="Let me think... 40 + 2 = 42")
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())
        client.client.chat.completions.create = Mock(return_value=mock_response)

        response = client.create({"model": "o1-preview", "messages": []})
        messages = client.message_retrieval(response)

        # Should concatenate reasoning and text
        assert len(messages) == 1
        assert "Let me think" in messages[0]
        assert "Answer: 42" in messages[0]


class TestOpenAIResponsesClientV1Compatible:
    """Test create_v1_compatible() method."""

    def test_create_v1_compatible_format(self, mock_openai_client):
        """Test backward compatible response format."""
        client = OpenAIResponsesClient(api_key="test-key")

        mock_message = MockMessage(content="Test response")
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Get v1 compatible response
        response = client.create_v1_compatible({"model": "gpt-4", "messages": []})

        # Verify it's a dict with expected structure
        assert isinstance(response, dict)
        assert "id" in response
        assert "model" in response
        assert "choices" in response
        assert "usage" in response
        assert "cost" in response
        assert response["object"] == "chat.completion"

    def test_v1_compatible_loses_reasoning(self, mock_openai_client):
        """Test that v1 compatible format loses reasoning blocks."""
        client = OpenAIResponsesClient(api_key="test-key")

        mock_message = MockMessage(content="Answer: 42", reasoning="Step 1: ... Step 2: ...")
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Get v1 response
        v1_response = client.create_v1_compatible({"model": "o1-preview", "messages": []})

        # V1 format should flatten to just content
        # Note: In v1 format, reasoning is lost (this is the limitation)
        assert "choices" in v1_response
        assert len(v1_response["choices"]) == 1
        assert "message" in v1_response["choices"][0]


class TestOpenAIResponsesClientIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_with_reasoning(self, mock_openai_client):
        """Test complete workflow with reasoning blocks."""
        client = OpenAIResponsesClient(api_key="test-key")

        # Mock o1 model response with reasoning
        mock_message = MockMessage(
            role="assistant",
            content="Based on my analysis, quantum computing uses qubits.",
            reasoning="Step 1: Analyze quantum computing fundamentals\nStep 2: Consider qubit superposition\nStep 3: Formulate clear explanation",
        )
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(
            model="o1-preview", choices=[mock_choice], usage=MockUsage(prompt_tokens=100, completion_tokens=200)
        )
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Create request
        response = client.create({
            "model": "o1-preview",
            "messages": [{"role": "user", "content": "Explain quantum computing"}],
        })

        # Verify all aspects
        assert isinstance(response, UnifiedResponse)
        assert response.model == "o1-preview"
        assert response.provider == "openai"

        # Check reasoning
        assert len(response.reasoning) == 1
        assert "Step 1" in response.reasoning[0].reasoning

        # Check text
        assert "quantum computing" in response.text.lower()

        # Check usage
        usage = client.get_usage(response)
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 200

        # Check cost
        assert response.cost > 0

    def test_protocol_compliance(self, mock_openai_client):
        """Test that client implements ModelClientV2 protocol."""
        client = OpenAIResponsesClient(api_key="test-key")

        # Check protocol compliance
        assert hasattr(client, "RESPONSE_USAGE_KEYS")
        assert callable(client.create)
        assert callable(client.create_v1_compatible)
        assert callable(client.cost)
        assert callable(client.get_usage)
        assert callable(client.message_retrieval)
