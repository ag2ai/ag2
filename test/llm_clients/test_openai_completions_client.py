# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenAICompletionsClient."""

from unittest.mock import Mock, patch

import pytest

from autogen.llm_clients import OpenAICompletionsClient
from autogen.llm_clients.models import GenericContent, ReasoningContent, TextContent, ToolCallContent, UnifiedResponse


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
    with patch("autogen.llm_clients.openai_completions_client.OpenAI") as mock_openai_class:
        # Create a mock instance that will be returned when OpenAI() is called
        mock_client_instance = Mock()
        mock_openai_class.return_value = mock_client_instance
        yield mock_client_instance


class TestOpenAICompletionsClientCreation:
    """Test client initialization."""

    def test_create_client_with_api_key(self, mock_openai_client):
        """Test creating client with API key."""
        client = OpenAICompletionsClient(api_key="test-key")
        assert client is not None
        assert client.client is not None

    def test_create_client_with_base_url(self, mock_openai_client):
        """Test creating client with custom base URL."""
        client = OpenAICompletionsClient(api_key="test-key", base_url="https://custom.api.com")
        assert client is not None

    def test_client_has_required_methods(self, mock_openai_client):
        """Test that client has all ModelClientV2 methods."""
        client = OpenAICompletionsClient(api_key="test-key")
        assert hasattr(client, "create")
        assert hasattr(client, "create_v1_compatible")
        assert hasattr(client, "cost")
        assert hasattr(client, "get_usage")
        assert hasattr(client, "message_retrieval")
        assert hasattr(client, "RESPONSE_USAGE_KEYS")


class TestOpenAICompletionsClientCreate:
    """Test create() method."""

    def test_create_simple_response(self, mock_openai_client):
        """Test creating a simple text response."""
        # Setup mock
        mock_message = MockMessage(role="assistant", content="The answer is 42")
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())

        client = OpenAICompletionsClient(api_key="test-key")
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

        client = OpenAICompletionsClient(api_key="test-key")
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

        client = OpenAICompletionsClient(api_key="test-key")
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

        client = OpenAICompletionsClient(api_key="test-key")
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Test
        response = client.create({"model": "gpt-4", "messages": []})

        # Verify usage
        assert response.usage["prompt_tokens"] == 100
        assert response.usage["completion_tokens"] == 200
        assert response.usage["total_tokens"] == 300


class TestOpenAICompletionsClientCost:
    """Test cost() method."""

    def test_cost_calculation_o1_preview(self, mock_openai_client):
        """Test cost calculation for o1-preview model."""
        client = OpenAICompletionsClient(api_key="test-key")

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
        client = OpenAICompletionsClient(api_key="test-key")

        mock_message = MockMessage(content="Test")
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(
            model="unknown-model", choices=[mock_choice], usage=MockUsage(prompt_tokens=100, completion_tokens=100)
        )
        client.client.chat.completions.create = Mock(return_value=mock_response)

        response = client.create({"model": "unknown-model", "messages": []})

        # Should use default pricing (gpt-4 level)
        assert response.cost > 0


class TestOpenAICompletionsClientGetUsage:
    """Test get_usage() method."""

    def test_get_usage_returns_all_keys(self, mock_openai_client):
        """Test that get_usage() returns all required keys."""
        client = OpenAICompletionsClient(api_key="test-key")

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


class TestOpenAICompletionsClientMessageRetrieval:
    """Test message_retrieval() method."""

    def test_message_retrieval_simple_text(self, mock_openai_client):
        """Test retrieving text from simple response."""
        client = OpenAICompletionsClient(api_key="test-key")

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
        client = OpenAICompletionsClient(api_key="test-key")

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


class TestOpenAICompletionsClientV1Compatible:
    """Test create_v1_compatible() method."""

    def test_create_v1_compatible_format(self, mock_openai_client):
        """Test backward compatible response format."""
        client = OpenAICompletionsClient(api_key="test-key")

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
        client = OpenAICompletionsClient(api_key="test-key")

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


class TestOpenAICompletionsClientIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_with_reasoning(self, mock_openai_client):
        """Test complete workflow with reasoning blocks."""
        client = OpenAICompletionsClient(api_key="test-key")

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
        client = OpenAICompletionsClient(api_key="test-key")

        # Check protocol compliance
        assert hasattr(client, "RESPONSE_USAGE_KEYS")
        assert callable(client.create)
        assert callable(client.create_v1_compatible)
        assert callable(client.cost)
        assert callable(client.get_usage)
        assert callable(client.message_retrieval)


class TestOpenAICompletionsClientGenericContent:
    """Test GenericContent handling for unknown OpenAI response types."""

    def test_multimodal_content_with_unknown_type(self, mock_openai_client):
        """Test that unknown content types in multimodal content are handled as GenericContent."""
        client = OpenAICompletionsClient(api_key="test-key")

        # Mock message with unknown content type in multimodal list
        mock_message = MockMessage(
            role="assistant",
            content=[
                {"type": "text", "text": "Here's the result:"},
                {"type": "reflection", "reflection": "Upon reviewing...", "confidence": 0.87},  # Unknown type
            ],
        )
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Test
        response = client.create({"model": "gpt-4", "messages": []})

        # Verify both content blocks are preserved
        assert len(response.messages[0].content) == 2

        # First should be TextContent
        assert isinstance(response.messages[0].content[0], TextContent)
        assert response.messages[0].content[0].text == "Here's the result:"

        # Second should be GenericContent (unknown type)
        assert isinstance(response.messages[0].content[1], GenericContent)
        assert response.messages[0].content[1].type == "reflection"
        assert response.messages[0].content[1].reflection == "Upon reviewing..."
        assert response.messages[0].content[1].confidence == 0.87

    def test_unknown_message_field_as_generic_content(self, mock_openai_client):
        """Test that unknown fields in message object are captured as GenericContent."""
        client = OpenAICompletionsClient(api_key="test-key")

        # Create a mock message with unknown field
        mock_message = MockMessage(role="assistant", content="Test response")
        # Add unknown field via model_dump simulation
        mock_message.model_dump = Mock(
            return_value={
                "role": "assistant",
                "content": "Test response",
                "thinking": "Let me analyze this step by step...",  # Unknown field
                "confidence_score": 0.92,  # Another unknown field
            }
        )

        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Test
        response = client.create({"model": "gpt-4", "messages": []})

        # Should have text + 2 generic content blocks for unknown fields
        content_blocks = response.messages[0].content
        generic_blocks = [b for b in content_blocks if isinstance(b, GenericContent)]

        assert len(generic_blocks) == 2

        # Find thinking block
        thinking_blocks = [b for b in generic_blocks if b.type == "thinking"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].thinking == "Let me analyze this step by step..."

        # Find confidence_score block
        confidence_blocks = [b for b in generic_blocks if b.type == "confidence_score"]
        assert len(confidence_blocks) == 1
        assert confidence_blocks[0].confidence_score == 0.92

    def test_generic_content_serialization(self, mock_openai_client):
        """Test that GenericContent can be serialized properly."""
        client = OpenAICompletionsClient(api_key="test-key")

        # Mock message with unknown content type
        mock_message = MockMessage(
            role="assistant", content=[{"type": "custom_analysis", "analysis": "Deep dive...", "score": 9.5}]
        )
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Test
        response = client.create({"model": "gpt-4", "messages": []})

        # Get GenericContent block
        generic_block = response.messages[0].content[0]
        assert isinstance(generic_block, GenericContent)

        # Test serialization
        serialized = generic_block.model_dump()
        assert serialized["type"] == "custom_analysis"
        assert serialized["analysis"] == "Deep dive..."
        assert serialized["score"] == 9.5

    def test_generic_content_attribute_access(self, mock_openai_client):
        """Test that GenericContent supports attribute access for unknown fields."""
        client = OpenAICompletionsClient(api_key="test-key")

        # Mock message with unknown content type
        mock_message = MockMessage(
            role="assistant",
            content=[
                {
                    "type": "advanced_reasoning",
                    "reasoning_steps": ["Step 1", "Step 2"],
                    "confidence": 0.95,
                    "citations": [{"url": "https://example.com"}],
                }
            ],
        )
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Test
        response = client.create({"model": "gpt-4", "messages": []})

        # Get GenericContent block
        generic_block = response.messages[0].content[0]
        assert isinstance(generic_block, GenericContent)

        # Test attribute access (Pydantic native extra='allow')
        assert generic_block.type == "advanced_reasoning"
        assert generic_block.reasoning_steps == ["Step 1", "Step 2"]
        assert generic_block.confidence == 0.95
        assert generic_block.citations == [{"url": "https://example.com"}]

    def test_generic_content_helper_methods(self, mock_openai_client):
        """Test that GenericContent helper methods work correctly."""
        client = OpenAICompletionsClient(api_key="test-key")

        # Mock message with unknown content type
        mock_message = MockMessage(
            role="assistant",
            content=[{"type": "evaluation", "score": 8.5, "feedback": "Good work", "tags": ["A", "B"]}],
        )
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Test
        response = client.create({"model": "gpt-4", "messages": []})

        # Get GenericContent block
        generic_block = response.messages[0].content[0]

        # Test get() method
        assert generic_block.get("score") == 8.5
        assert generic_block.get("feedback") == "Good work"
        assert generic_block.get("missing", "default") == "default"

        # Test get_all_fields()
        all_fields = generic_block.get_all_fields()
        assert "type" in all_fields
        assert "score" in all_fields
        assert "feedback" in all_fields
        assert "tags" in all_fields

        # Test get_extra_fields()
        extra_fields = generic_block.get_extra_fields()
        assert "score" in extra_fields
        assert "feedback" in extra_fields
        # Should not include base field 'type'
        assert "type" not in extra_fields or extra_fields["type"] == "evaluation"

        # Test has_field()
        assert generic_block.has_field("score") is True
        assert generic_block.has_field("type") is True
        assert generic_block.has_field("missing") is False

    def test_mixed_known_and_unknown_content(self, mock_openai_client):
        """Test response with both known content types and GenericContent."""
        client = OpenAICompletionsClient(api_key="test-key")

        # Mock message with mixed content
        mock_message = MockMessage(
            role="assistant",
            content=[
                {"type": "text", "text": "Answer is 42"},
                {"type": "reflection", "reflection": "I calculated...", "confidence": 0.9},  # Unknown
                {"type": "text", "text": "Hope this helps!"},
            ],
        )
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())
        client.client.chat.completions.create = Mock(return_value=mock_response)

        # Test
        response = client.create({"model": "gpt-4", "messages": []})

        # Verify content blocks
        content_blocks = response.messages[0].content
        assert len(content_blocks) == 3

        # First text block
        assert isinstance(content_blocks[0], TextContent)
        assert content_blocks[0].text == "Answer is 42"

        # Unknown reflection block
        assert isinstance(content_blocks[1], GenericContent)
        assert content_blocks[1].type == "reflection"
        assert content_blocks[1].reflection == "I calculated..."

        # Second text block
        assert isinstance(content_blocks[2], TextContent)
        assert content_blocks[2].text == "Hope this helps!"

        # Test message retrieval combines all text
        messages = client.message_retrieval(response)
        assert len(messages) == 1
        # Should include text from TextContent blocks and GenericContent
        assert "Answer is 42" in messages[0]
        assert "Hope this helps!" in messages[0]
