# test/llm_clients/test_anthropic_v2.py
# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for AnthropicCompletionsClient (v2 client).

These tests use mocked Anthropic API responses to test the client's
functionality without requiring API keys.
"""

import json
from typing import Any
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from autogen.llm_clients.anthropic_v2 import AnthropicV2Client
from autogen.llm_clients.models import (
    GenericContent,
    ReasoningContent,
    TextContent,
    ToolCallContent,
    UnifiedMessage,
    UnifiedResponse,
)


# Mock Anthropic response classes
class MockUsage:
    """Mock Anthropic usage object."""

    def __init__(self, input_tokens: int = 10, output_tokens: int = 20):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockTextBlock:
    """Mock Anthropic TextBlock."""

    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class MockThinkingBlock:
    """Mock Anthropic ThinkingBlock."""

    def __init__(self, thinking: str):
        self.type = "thinking"
        self.thinking = thinking


class MockToolUseBlock:
    """Mock Anthropic ToolUseBlock."""

    def __init__(self, tool_id: str, name: str, input_data: dict):
        self.type = "tool_use"
        self.id = tool_id
        self.name = name
        self.input = input_data


class MockAnthropicMessage:
    """Mock Anthropic Message response."""

    def __init__(
        self,
        content: list[Any],
        id: str = "msg_123",
        model: str = "claude-3-5-sonnet-20241022",
        stop_reason: str = "end_turn",
        usage: MockUsage | None = None,
        parsed_output: Any = None,
    ):
        self.id = id
        self.model = model
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or MockUsage()
        if parsed_output is not None:
            self.parsed_output = parsed_output


@pytest.fixture
def mock_anthropic_client():
    """Create mock Anthropic client."""
    with patch("autogen.llm_clients.anthropic_v2.Anthropic") as mock_anthropic_class:
        mock_client_instance = Mock()
        mock_messages = Mock()
        mock_beta = Mock()
        mock_beta.messages = Mock()
        mock_client_instance.messages = mock_messages
        mock_client_instance.beta = mock_beta
        mock_anthropic_class.return_value = mock_client_instance
        yield mock_client_instance


@pytest.fixture
def anthropic_v2_client(mock_anthropic_client):
    """Create AnthropicV2Client instance with mocked Anthropic SDK."""
    return AnthropicV2Client(api_key="test-key")


class TestAnthropicV2ClientCreation:
    """Test client initialization."""

    def test_create_client_with_api_key(self, mock_anthropic_client):
        """Test creating client with API key."""
        client = AnthropicV2Client(api_key="test-key")
        assert client._api_key == "test-key"
        assert client._client is not None

    def test_create_client_with_env_var(self, mock_anthropic_client):
        """Test creating client with environment variable."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            client = AnthropicV2Client()
            assert client._api_key == "env-key"

    def test_create_client_with_response_format(self, mock_anthropic_client):
        """Test creating client with response format."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        client = AnthropicV2Client(api_key="test-key", response_format=TestModel)
        assert client._response_format == TestModel

    def test_create_client_missing_api_key(self, mock_anthropic_client):
        """Test creating client without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key is required"):
                AnthropicV2Client()


class TestStandardCompletion:
    """Test standard completion without structured outputs."""

    def test_simple_text_response(self, anthropic_v2_client, mock_anthropic_client):
        """Test simple text response."""
        # Setup mock response
        text_block = MockTextBlock("Hello, world!")
        mock_response = MockAnthropicMessage(content=[text_block])
        mock_anthropic_client.messages.create.return_value = mock_response

        # Make request
        params = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = anthropic_v2_client.create(params)

        # Verify response
        assert isinstance(response, UnifiedResponse)
        assert response.model == "claude-3-5-sonnet-20241022"
        assert response.provider == "anthropic"
        assert len(response.messages) == 1
        assert len(response.messages[0].content) == 1
        assert isinstance(response.messages[0].content[0], TextContent)
        assert response.messages[0].content[0].text == "Hello, world!"

    def test_response_with_thinking_block(self, anthropic_v2_client, mock_anthropic_client):
        """Test response with thinking block."""
        # Setup mock response with thinking
        thinking_block = MockThinkingBlock("Let me think about this...")
        text_block = MockTextBlock("The answer is 42")
        mock_response = MockAnthropicMessage(content=[thinking_block, text_block])
        mock_anthropic_client.messages.create.return_value = mock_response

        # Make request
        params = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "What is 6 * 7?"}],
        }
        response = anthropic_v2_client.create(params)

        # Verify response has both thinking and text
        assert len(response.messages[0].content) == 2
        assert isinstance(response.messages[0].content[0], ReasoningContent)
        assert response.messages[0].content[0].reasoning == "Let me think about this..."
        assert isinstance(response.messages[0].content[1], TextContent)
        assert response.messages[0].content[1].text == "The answer is 42"

    def test_response_with_tool_calls(self, anthropic_v2_client, mock_anthropic_client):
        """Test response with tool calls."""
        # Setup mock response with tool use
        tool_block = MockToolUseBlock("tool_123", "get_weather", {"city": "San Francisco"})
        mock_response = MockAnthropicMessage(
            content=[tool_block], stop_reason="tool_use", usage=MockUsage(10, 15)
        )
        mock_anthropic_client.messages.create.return_value = mock_response

        # Make request with tools
        params = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "What's the weather in SF?"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            ],
        }
        response = anthropic_v2_client.create(params)

        # Verify tool call
        assert len(response.messages[0].content) == 1
        assert isinstance(response.messages[0].content[0], ToolCallContent)
        assert response.messages[0].content[0].id == "tool_123"
        assert response.messages[0].content[0].name == "get_weather"
        assert response.finish_reason == "tool_calls"


class TestNativeStructuredOutputs:
    """Test native structured outputs (beta API)."""

    def test_structured_output_with_parse_method(self, anthropic_v2_client, mock_anthropic_client):
        """Test structured output using .parse() method (Pydantic model, no tools)."""
        from pydantic import BaseModel

        class ContactInfo(BaseModel):
            name: str
            email: str

        # Setup mock parsed output
        parsed_model = ContactInfo(name="John Doe", email="john@example.com")
        mock_response = MockAnthropicMessage(
            content=[MockTextBlock('{"name": "John Doe", "email": "john@example.com"}')],
            parsed_output=parsed_model,
        )
        mock_anthropic_client.beta.messages.parse.return_value = mock_response

        # Make request
        params = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Extract contact info"}],
            "response_format": ContactInfo,
        }
        response = anthropic_v2_client.create(params)

        # Verify parsed output is stored
        assert len(response.messages[0].content) == 2  # GenericContent + TextContent
        parsed_block = response.messages[0].content[0]
        assert isinstance(parsed_block, GenericContent)
        assert parsed_block.type == "parsed"
        assert parsed_block.parsed == {"name": "John Doe", "email": "john@example.com"}

    def test_structured_output_with_create_method(self, anthropic_v2_client, mock_anthropic_client):
        """Test structured output using .create() method (with tools or dict schema)."""
        from pydantic import BaseModel

        class ContactInfo(BaseModel):
            name: str
            email: str

        # Setup mock response with JSON text (no parsed_output)
        json_text = '{"name": "Jane Doe", "email": "jane@example.com"}'
        mock_response = MockAnthropicMessage(content=[MockTextBlock(json_text)])
        mock_anthropic_client.beta.messages.create.return_value = mock_response

        # Make request with tools (forces .create() method)
        params = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Extract contact info"}],
            "response_format": ContactInfo,
            "tools": [
                {
                    "name": "search",
                    "description": "Search",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        }
        response = anthropic_v2_client.create(params)

        # Verify JSON was parsed into Pydantic model
        assert len(response.messages[0].content) == 2  # GenericContent + TextContent
        parsed_block = response.messages[0].content[0]
        assert isinstance(parsed_block, GenericContent)
        assert parsed_block.type == "parsed"
        assert parsed_block.parsed == {"name": "Jane Doe", "email": "jane@example.com"}

    def test_structured_output_dict_schema(self, anthropic_v2_client, mock_anthropic_client):
        """Test structured output with dict schema (always uses .create())."""
        # Setup mock response
        json_text = '{"name": "Test", "value": 42}'
        mock_response = MockAnthropicMessage(content=[MockTextBlock(json_text)])
        mock_anthropic_client.beta.messages.create.return_value = mock_response

        # Make request with dict schema
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "value": {"type": "integer"}},
        }
        params = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": schema,
        }
        response = anthropic_v2_client.create(params)

        # Verify text content (no parsing for dict schemas)
        assert len(response.messages[0].content) == 1
        assert isinstance(response.messages[0].content[0], TextContent)
        assert response.messages[0].content[0].text == json_text

    def test_structured_output_fallback_to_json_mode(self, anthropic_v2_client, mock_anthropic_client):
        """Test fallback to JSON Mode when native SO fails."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            value: str

        # Mock native SO to fail
        mock_anthropic_client.beta.messages.create.side_effect = Exception("API Error")
        # Mock JSON Mode to succeed
        json_mode_response = MockAnthropicMessage(
            content=[MockTextBlock("<json_response>{\"value\": \"test\"}</json_response>")]
        )
        mock_anthropic_client.messages.create.return_value = json_mode_response

        params = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": TestModel,
        }
        response = anthropic_v2_client.create(params)

        # Should fallback to JSON Mode
        assert mock_anthropic_client.messages.create.called


class TestJSONMode:
    """Test JSON Mode structured outputs (fallback for older models)."""

    def test_json_mode_extraction(self, anthropic_v2_client, mock_anthropic_client):
        """Test JSON extraction from <json_response> tags."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            age: int

        # Setup mock response with JSON in tags
        json_text = '<json_response>{"name": "Alice", "age": 30}</json_response>'
        mock_response = MockAnthropicMessage(content=[MockTextBlock(json_text)])
        mock_anthropic_client.messages.create.return_value = mock_response

        params = {
            "model": "claude-3-5-sonnet-20241022",  # Older model
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": TestModel,
        }
        response = anthropic_v2_client.create(params)

        # Verify JSON was extracted and parsed
        text_content = response.messages[0].content[0]
        assert isinstance(text_content, TextContent)
        # Should contain parsed JSON (not the tags)
        assert "name" in text_content.text
        assert "Alice" in text_content.text


class TestCostAndUsage:
    """Test cost calculation and usage extraction."""

    def test_cost_calculation(self, anthropic_v2_client):
        """Test cost calculation from usage."""
        usage = {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}
        response = UnifiedResponse(
            id="test",
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            messages=[UnifiedMessage(role="assistant", content=[TextContent(type="text", text="test")])],
            usage=usage,
        )
        cost = anthropic_v2_client.cost(response)
        assert isinstance(cost, float)
        assert cost >= 0

    def test_get_usage(self, anthropic_v2_client):
        """Test get_usage static method."""
        usage = {"prompt_tokens": 50, "completion_tokens": 75, "total_tokens": 125}
        response = UnifiedResponse(
            id="test",
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            messages=[UnifiedMessage(role="assistant", content=[TextContent(type="text", text="test")])],
            usage=usage,
            cost=0.001,
        )
        usage_dict = AnthropicV2Client.get_usage(response)
        assert usage_dict["prompt_tokens"] == 50
        assert usage_dict["completion_tokens"] == 75
        assert usage_dict["total_tokens"] == 125
        assert usage_dict["cost"] == 0.001
        assert usage_dict["model"] == "claude-3-5-sonnet-20241022"


class TestMessageRetrieval:
    """Test message_retrieval method."""

    def test_message_retrieval_text_only(self, anthropic_v2_client):
        """Test message retrieval for text-only response."""
        response = UnifiedResponse(
            id="test",
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            messages=[
                UnifiedMessage(
                    role="assistant",
                    content=[TextContent(type="text", text="Hello, world!")],
                )
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
        messages = anthropic_v2_client.message_retrieval(response)
        assert isinstance(messages, list)
        assert len(messages) == 1
        assert messages[0] == "Hello, world!"

    def test_message_retrieval_with_tool_calls(self, anthropic_v2_client):
        """Test message retrieval with tool calls."""
        response = UnifiedResponse(
            id="test",
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            messages=[
                UnifiedMessage(
                    role="assistant",
                    content=[
                        ToolCallContent(
                            type="tool_call",
                            id="call_123",
                            name="get_weather",
                            arguments='{"city": "SF"}',
                        )
                    ],
                )
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
        messages = anthropic_v2_client.message_retrieval(response)
        assert isinstance(messages, list)
        assert len(messages) == 1
        # Should return ChatCompletionMessage dict for tool calls
        assert hasattr(messages[0], "tool_calls") or isinstance(messages[0], dict)


class TestV1Compatibility:
    """Test create_v1_compatible method."""

    def test_create_v1_compatible(self, anthropic_v2_client, mock_anthropic_client):
        """Test create_v1_compatible returns ChatCompletion format."""
        # Setup mock response
        text_block = MockTextBlock("Test response")
        mock_response = MockAnthropicMessage(content=[text_block])
        mock_anthropic_client.messages.create.return_value = mock_response

        params = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        v1_response = anthropic_v2_client.create_v1_compatible(params)

        # Verify ChatCompletion format
        from autogen.oai.oai_models import ChatCompletion

        assert isinstance(v1_response, ChatCompletion)
        assert v1_response.id is not None
        assert v1_response.model == "claude-3-5-sonnet-20241022"
        assert len(v1_response.choices) == 1
        assert v1_response.choices[0].message.content == "Test response"

    def test_create_v1_compatible_with_thinking(self, anthropic_v2_client, mock_anthropic_client):
        """Test create_v1_compatible preserves thinking blocks."""
        # Setup mock response with thinking
        thinking_block = MockThinkingBlock("Thinking...")
        text_block = MockTextBlock("Answer")
        mock_response = MockAnthropicMessage(content=[thinking_block, text_block])
        mock_anthropic_client.messages.create.return_value = mock_response

        params = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Question"}],
        }
        v1_response = anthropic_v2_client.create_v1_compatible(params)

        # Verify thinking is preserved in [Thinking] tags
        content = v1_response.choices[0].message.content
        assert "[Thinking]" in content
        assert "Thinking..." in content
        assert "Answer" in content


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_json_in_structured_output(self, anthropic_v2_client, mock_anthropic_client):
        """Test handling of invalid JSON in structured output."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            value: str

        # Setup mock response with invalid JSON
        mock_response = MockAnthropicMessage(content=[MockTextBlock("not valid json")])
        mock_anthropic_client.beta.messages.create.return_value = mock_response

        params = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": TestModel,
            "tools": [{"name": "test", "input_schema": {}}],  # Force .create() method
        }
        # Should not raise, but log warning
        response = anthropic_v2_client.create(params)
        # Should fallback to raw text
        assert isinstance(response.messages[0].content[0], TextContent)

    def test_missing_api_key_error(self):
        """Test error when API key is missing."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key is required"):
                AnthropicV2Client()