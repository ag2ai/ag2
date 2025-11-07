# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GeminiStatelessClient (mocked, no API calls).

These tests use mocking and do not require the google.genai package.
They should run in both skip-llm and core-llm test modes.
"""

from unittest.mock import Mock, patch

import pytest

from autogen.llm_clients import GeminiStatelessClient
from autogen.llm_clients.models import ThinkingContent, ToolCallContent, UnifiedResponse


class MockGeminiPart:
    """Mock Gemini Part object."""

    def __init__(self, text=None, thought=False, function_call=None, inline_data=None):
        self.text = text
        self.thought = thought
        self.function_call = function_call
        self.inline_data = inline_data


class MockFunctionCall:
    """Mock Gemini FunctionCall."""

    def __init__(self, name, args):
        self.name = name
        self.args = args


class MockUsageMetadata:
    """Mock Gemini UsageMetadata."""

    def __init__(self, prompt_tokens=50, completion_tokens=100, thinking_tokens=0):
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = completion_tokens
        self.total_token_count = prompt_tokens + completion_tokens
        self.thoughts_token_count = thinking_tokens


class MockCandidate:
    """Mock Gemini Candidate."""

    def __init__(self, content_parts):
        self.content = Mock()
        self.content.parts = content_parts
        self.finish_reason = "STOP"


class MockGenerateContentResponse:
    """Mock Gemini GenerateContentResponse."""

    def __init__(self, candidates, usage_metadata=None, model_version="gemini-2.0-flash"):
        self.candidates = candidates
        self.usage_metadata = usage_metadata or MockUsageMetadata()
        self.model_version = model_version


@pytest.fixture
def mock_genai_client():
    """Create mock Google GenAI client."""
    # Patch at the import location, not the module attribute
    with patch("google.genai.Client") as mock_client_class:
        # Create mock client instance
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock the models.generate_content method
        mock_client.models = Mock()
        mock_client.models.generate_content = Mock()

        yield mock_client


class TestGeminiStatelessClientCreation:
    """Test client initialization."""

    def test_create_client_with_api_key(self, mock_genai_client):
        """Test creating client with API key."""
        client = GeminiStatelessClient(api_key="test-key")
        assert client is not None
        assert client.use_vertexai is False

    def test_create_client_vertexai_mode(self):
        """Test creating client in Vertex AI mode."""
        with patch("autogen.llm_clients.gemini_stateless_client.vertexai") as mock_vertex:
            mock_vertex.init = Mock()
            client = GeminiStatelessClient(vertexai=True, project="test-project", location="us-central1")
            assert client is not None
            assert client.use_vertexai is True
            mock_vertex.init.assert_called_once()

    def test_client_has_required_methods(self, mock_genai_client):
        """Test that client has all ModelClientV2 methods."""
        client = GeminiStatelessClient(api_key="test-key")
        assert hasattr(client, "create")
        assert hasattr(client, "cost")
        assert hasattr(client, "get_usage")
        assert hasattr(client, "message_retrieval")
        assert hasattr(client, "RESPONSE_USAGE_KEYS")


class TestGeminiStatelessClientCreate:
    """Test create() method."""

    def test_create_simple_text_response(self, mock_genai_client):
        """Test creating a simple text response."""
        # Setup mock response
        mock_part = MockGeminiPart(text="Hello, World!")
        mock_candidate = MockCandidate([mock_part])
        mock_response = MockGenerateContentResponse([mock_candidate])

        mock_genai_client.models.generate_content.return_value = mock_response

        # Create client and test
        client = GeminiStatelessClient(api_key="test-key")
        response = client.create({"model": "gemini-2.0-flash", "messages": [{"role": "user", "content": "Say hello"}]})

        # Verify
        assert isinstance(response, UnifiedResponse)
        assert response.provider == "gemini"
        assert response.model == "gemini-2.0-flash"
        assert len(response.messages) == 1
        assert response.text == "Hello, World!"

    def test_create_response_with_thinking(self, mock_genai_client):
        """Test creating response with thinking blocks (Gemini 2.5+)."""
        # Setup mock with thinking
        thinking_part = MockGeminiPart(text="Let me think step by step...", thought=True)
        text_part = MockGeminiPart(text="The answer is 42")
        mock_candidate = MockCandidate([thinking_part, text_part])
        mock_response = MockGenerateContentResponse(
            [mock_candidate],
            usage_metadata=MockUsageMetadata(prompt_tokens=50, completion_tokens=100, thinking_tokens=200),
        )

        mock_genai_client.models.generate_content.return_value = mock_response

        # Test
        client = GeminiStatelessClient(api_key="test-key")
        response = client.create({
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "Solve this"}],
            "thinking_config": {"include_thoughts": True},
        })

        # Verify thinking blocks
        thinking_blocks = [b for b in response.messages[0].content if b.type == "thinking"]
        assert len(thinking_blocks) == 1
        assert isinstance(thinking_blocks[0], ThinkingContent)
        assert "step by step" in thinking_blocks[0].thinking

        # Verify text is also present
        text_blocks = [b for b in response.messages[0].content if b.type == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "The answer is 42"

        # Verify thinking tokens tracked
        assert response.usage["thinking_tokens"] == 200

    def test_create_response_with_tool_calls(self, mock_genai_client):
        """Test creating response with tool calls."""
        # Setup mock with function call
        mock_func_call = MockFunctionCall("get_weather", {"location": "Tokyo"})
        function_part = MockGeminiPart(function_call=mock_func_call)
        mock_candidate = MockCandidate([function_part])
        mock_response = MockGenerateContentResponse([mock_candidate])

        mock_genai_client.models.generate_content.return_value = mock_response

        # Test
        client = GeminiStatelessClient(api_key="test-key")
        response = client.create({
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        })

        # Verify tool calls
        tool_calls = response.messages[0].get_tool_calls()
        assert len(tool_calls) == 1
        assert isinstance(tool_calls[0], ToolCallContent)
        assert tool_calls[0].name == "get_weather"
        assert "Tokyo" in tool_calls[0].arguments

    def test_create_response_with_usage(self, mock_genai_client):
        """Test that usage information is properly extracted."""
        # Setup mock
        mock_part = MockGeminiPart(text="Test response")
        mock_candidate = MockCandidate([mock_part])
        mock_response = MockGenerateContentResponse(
            [mock_candidate], usage_metadata=MockUsageMetadata(prompt_tokens=100, completion_tokens=200)
        )

        mock_genai_client.models.generate_content.return_value = mock_response

        # Test
        client = GeminiStatelessClient(api_key="test-key")
        response = client.create({"model": "gemini-2.0-flash", "messages": [{"role": "user", "content": "Test"}]})

        # Verify usage
        assert response.usage["prompt_tokens"] == 100
        assert response.usage["completion_tokens"] == 200
        assert response.usage["total_tokens"] == 300
        assert response.usage["thinking_tokens"] == 0  # No thinking in this test


class TestGeminiStatelessClientCost:
    """Test cost() method."""

    def test_cost_calculation_gemini_flash(self, mock_genai_client):
        """Test cost calculation for gemini-2.0-flash model."""
        client = GeminiStatelessClient(api_key="test-key")

        # Create response with known usage
        mock_part = MockGeminiPart(text="Test")
        mock_candidate = MockCandidate([mock_part])
        mock_response = MockGenerateContentResponse(
            [mock_candidate], usage_metadata=MockUsageMetadata(prompt_tokens=1000, completion_tokens=500)
        )
        mock_genai_client.models.generate_content.return_value = mock_response

        response = client.create({"model": "gemini-2.0-flash", "messages": [{"role": "user", "content": "Test"}]})

        # Verify cost calculation
        assert response.cost is not None
        assert response.cost > 0

    def test_cost_calculation_with_thinking_tokens(self, mock_genai_client):
        """Test cost calculation includes thinking tokens (Gemini 2.5 preview)."""
        client = GeminiStatelessClient(api_key="test-key")

        # Create response with thinking tokens
        mock_part = MockGeminiPart(text="Answer")
        mock_candidate = MockCandidate([mock_part])
        mock_response = MockGenerateContentResponse(
            [mock_candidate],
            usage_metadata=MockUsageMetadata(prompt_tokens=1000, completion_tokens=500, thinking_tokens=2000),
            model_version="gemini-2.5-flash-preview-04-17",
        )
        mock_genai_client.models.generate_content.return_value = mock_response

        response = client.create({
            "model": "gemini-2.5-flash-preview-04-17",
            "messages": [{"role": "user", "content": "Think"}],
        })

        # Verify thinking tokens are tracked
        assert response.usage["thinking_tokens"] == 2000

        # Verify cost includes thinking tokens ($3/million for thinking)
        # Expected: base_cost + (3.0 * 2000 / 1e6) = base_cost + 0.006
        assert response.cost > 0.006


class TestGeminiStatelessClientGetUsage:
    """Test get_usage() method."""

    def test_get_usage_returns_all_keys(self, mock_genai_client):
        """Test that get_usage() returns all required keys."""
        client = GeminiStatelessClient(api_key="test-key")

        mock_part = MockGeminiPart(text="Test")
        mock_candidate = MockCandidate([mock_part])
        mock_response = MockGenerateContentResponse(
            [mock_candidate], usage_metadata=MockUsageMetadata(prompt_tokens=50, completion_tokens=75)
        )
        mock_genai_client.models.generate_content.return_value = mock_response

        response = client.create({"model": "gemini-2.0-flash", "messages": [{"role": "user", "content": "Test"}]})
        usage = client.get_usage(response)

        # Verify all required keys
        for key in client.RESPONSE_USAGE_KEYS:
            assert key in usage

        assert usage["prompt_tokens"] == 50
        assert usage["completion_tokens"] == 75
        assert usage["total_tokens"] == 125
        assert usage["model"] == "gemini-2.0-flash"
        assert usage["cost"] > 0


class TestGeminiStatelessClientMessageRetrieval:
    """Test message_retrieval() method."""

    def test_message_retrieval_simple_text(self, mock_genai_client):
        """Test retrieving text from simple response."""
        client = GeminiStatelessClient(api_key="test-key")

        mock_part = MockGeminiPart(text="Hello world")
        mock_candidate = MockCandidate([mock_part])
        mock_response = MockGenerateContentResponse([mock_candidate])
        mock_genai_client.models.generate_content.return_value = mock_response

        response = client.create({"model": "gemini-2.0-flash", "messages": [{"role": "user", "content": "Hi"}]})
        messages = client.message_retrieval(response)

        assert len(messages) == 1
        assert messages[0] == "Hello world"

    def test_message_retrieval_with_thinking(self, mock_genai_client):
        """Test retrieving text from response with thinking."""
        client = GeminiStatelessClient(api_key="test-key")

        thinking_part = MockGeminiPart(text="Thinking...", thought=True)
        text_part = MockGeminiPart(text="Answer: 42")
        mock_candidate = MockCandidate([thinking_part, text_part])
        mock_response = MockGenerateContentResponse([mock_candidate])
        mock_genai_client.models.generate_content.return_value = mock_response

        response = client.create({"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "Think"}]})
        messages = client.message_retrieval(response)

        # message_retrieval returns separate entries for thinking and text
        assert len(messages) == 2
        assert "Thinking" in messages[0]
        assert "Answer: 42" in messages[1]


class TestGeminiStatelessClientIntegration:
    """Integration tests for complete workflows (mocked)."""

    def test_full_workflow_with_thinking(self, mock_genai_client):
        """Test complete workflow with thinking blocks."""
        client = GeminiStatelessClient(api_key="test-key")

        # Mock Gemini 2.5 response with thinking
        thinking_part = MockGeminiPart(
            text="Step 1: Consider quantum fundamentals\nStep 2: Analyze superposition", thought=True
        )
        text_part = MockGeminiPart(text="Quantum computing uses qubits in superposition.")
        mock_candidate = MockCandidate([thinking_part, text_part])
        mock_response = MockGenerateContentResponse(
            [mock_candidate],
            usage_metadata=MockUsageMetadata(prompt_tokens=100, completion_tokens=200, thinking_tokens=150),
            model_version="gemini-2.5-flash",
        )
        mock_genai_client.models.generate_content.return_value = mock_response

        # Create request
        response = client.create({
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "Explain quantum computing"}],
            "thinking_config": {"include_thoughts": True},
        })

        # Verify all aspects
        assert isinstance(response, UnifiedResponse)
        assert response.model == "gemini-2.5-flash"
        assert response.provider == "gemini"

        # Check thinking
        thinking = response.thinking
        assert len(thinking) == 1
        assert "Step 1" in thinking[0].thinking

        # Check text
        assert "quantum computing" in response.text.lower()

        # Check usage
        usage = client.get_usage(response)
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 200

        # Thinking tokens are in response.usage, not in get_usage() return
        assert response.usage["thinking_tokens"] == 150

        # Check cost
        assert response.cost > 0

    def test_protocol_compliance(self, mock_genai_client):
        """Test that client implements ModelClientV2 protocol."""
        client = GeminiStatelessClient(api_key="test-key")

        # Check protocol compliance
        assert hasattr(client, "RESPONSE_USAGE_KEYS")
        assert callable(client.create)
        assert callable(client.cost)
        assert callable(client.get_usage)
        assert callable(client.message_retrieval)

    def test_multimodal_content_handling(self, mock_genai_client):
        """Test handling of multimodal content."""
        client = GeminiStatelessClient(api_key="test-key")

        # Mock response with text
        mock_part = MockGeminiPart(text="This is a red square")
        mock_candidate = MockCandidate([mock_part])
        mock_response = MockGenerateContentResponse([mock_candidate])
        mock_genai_client.models.generate_content.return_value = mock_response

        # Test with image input
        response = client.create({
            "model": "gemini-2.0-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                    ],
                }
            ],
        })

        # Verify response
        assert isinstance(response, UnifiedResponse)
        assert "red square" in response.text.lower()

    def test_tool_calling_workflow(self, mock_genai_client):
        """Test complete tool calling workflow."""
        client = GeminiStatelessClient(api_key="test-key")

        # Mock tool call
        mock_func_call = MockFunctionCall("get_weather", {"location": "San Francisco", "unit": "fahrenheit"})
        function_part = MockGeminiPart(function_call=mock_func_call)
        mock_candidate = MockCandidate([function_part])
        mock_response = MockGenerateContentResponse([mock_candidate])
        mock_genai_client.models.generate_content.return_value = mock_response

        # Test
        response = client.create({
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                            },
                        },
                    },
                }
            ],
        })

        # Verify tool call
        tool_calls = response.messages[0].get_tool_calls()
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].id  # Should have generated UUID
        assert "San Francisco" in tool_calls[0].arguments
