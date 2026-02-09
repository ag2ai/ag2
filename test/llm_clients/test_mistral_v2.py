# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for MistralAIClientV2."""

from unittest.mock import Mock, patch

import pytest

from autogen.import_utils import run_for_optional_imports
from autogen.llm_clients import MistralAIClientV2
from autogen.llm_clients.models import (
    TextContent,
    ToolCallContent,
    UnifiedMessage,
    UnifiedResponse,
    UserRoleEnum,
)


class MockMistralResponse:
    """Mock Mistral API response."""

    def __init__(
        self,
        response_id="mistral-test123",
        model="mistral-small-latest",
        choices=None,
        usage=None,
    ):
        self.id = response_id
        self.model = model
        self.choices = choices or []
        self.usage = usage


class MockMistralChoice:
    """Mock choice in Mistral response."""

    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason


class MockMistralMessage:
    """Mock message in Mistral response."""

    def __init__(self, role="assistant", content=None, tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class MockMistralUsage:
    """Mock usage stats."""

    def __init__(self, prompt_tokens=50, completion_tokens=100):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class MockMistralToolCall:
    """Mock tool call."""

    def __init__(self, call_id="call_123", name="get_weather", arguments=None):
        self.id = call_id
        self.function = Mock()
        self.function.name = name
        # Mistral returns arguments as dict
        self.function.arguments = arguments or {"city": "San Francisco"}


@pytest.fixture
def mock_mistral_client():
    """Create mock Mistral client."""
    with patch("autogen.llm_clients.mistral_v2.Mistral") as mock_mistral_class:
        mock_client_instance = Mock()
        mock_mistral_class.return_value = mock_client_instance
        yield mock_client_instance


@pytest.fixture
def mistral_v2_client(mock_mistral_client):
    """Create MistralAIClientV2 instance."""
    return MistralAIClientV2(api_key="fake_api_key")


class TestMistralV2ClientCreation:
    """Test client initialization."""

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_create_client_with_api_key(self, mock_mistral_client):
        """Test creating client with API key."""
        client = MistralAIClientV2(api_key="test-key")
        assert client is not None
        assert client._client is not None

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_create_client_missing_api_key(self, mock_mistral_client):
        """Test creating client without API key raises error."""
        with pytest.raises(AssertionError) as exc_info:
            MistralAIClientV2()
        assert "api_key" in str(exc_info.value).lower() or "MISTRAL_API_KEY" in str(exc_info.value)

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_client_has_required_methods(self, mock_mistral_client):
        """Test that client has all ModelClientV2 methods."""
        client = MistralAIClientV2(api_key="test-key")
        assert hasattr(client, "create")
        assert hasattr(client, "create_v1_compatible")
        assert hasattr(client, "cost")
        assert hasattr(client, "get_usage")
        assert hasattr(client, "message_retrieval")
        assert hasattr(client, "RESPONSE_USAGE_KEYS")


class TestMistralV2ClientCreate:
    """Test create() method."""

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_create_simple_response(self, mistral_v2_client, mock_mistral_client):
        """Test creating a simple text response."""
        # Setup mock
        mock_message = MockMistralMessage(role="assistant", content="The answer is 42")
        mock_choice = MockMistralChoice(message=mock_message)
        mock_response = MockMistralResponse(
            response_id="mistral-123",
            model="mistral-small-latest",
            choices=[mock_choice],
            usage=MockMistralUsage(prompt_tokens=10, completion_tokens=20),
        )

        mock_mistral_client.chat.complete = Mock(return_value=mock_response)

        # Test
        response = mistral_v2_client.create({
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "What is 40+2?"}],
        })

        # Verify
        assert isinstance(response, UnifiedResponse)
        assert response.id == "mistral-123"
        assert response.model == "mistral-small-latest"
        assert response.provider == "mistral"
        assert len(response.messages) == 1
        assert response.text == "The answer is 42"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20
        assert response.usage["total_tokens"] == 30

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_create_response_with_tool_calls(self, mistral_v2_client, mock_mistral_client):
        """Test creating response with tool calls."""
        # Setup mock with tool calls
        mock_tool_call = MockMistralToolCall(call_id="call_abc", name="get_weather", arguments={"city": "NYC"})
        mock_message = MockMistralMessage(
            role="assistant", content="I'll check the weather", tool_calls=[mock_tool_call]
        )
        mock_choice = MockMistralChoice(message=mock_message, finish_reason="tool_calls")
        mock_response = MockMistralResponse(
            response_id="mistral-456",
            model="mistral-small-latest",
            choices=[mock_choice],
            usage=MockMistralUsage(),
        )

        mock_mistral_client.chat.complete = Mock(return_value=mock_response)

        # Test
        response = mistral_v2_client.create({
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "Get weather"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "description": "Get weather", "parameters": {}},
                }
            ],
        })

        # Verify tool calls are extracted
        tool_calls = response.messages[0].get_tool_calls()
        assert len(tool_calls) == 1
        assert isinstance(tool_calls[0], ToolCallContent)
        assert tool_calls[0].id == "call_abc"
        assert tool_calls[0].name == "get_weather"
        # Arguments should be JSON string
        assert "city" in tool_calls[0].arguments
        assert "NYC" in tool_calls[0].arguments

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_create_response_with_usage(self, mistral_v2_client, mock_mistral_client):
        """Test that usage information is properly extracted."""
        # Setup mock
        mock_message = MockMistralMessage(content="Test response")
        mock_choice = MockMistralChoice(message=mock_message)
        mock_response = MockMistralResponse(
            choices=[mock_choice], usage=MockMistralUsage(prompt_tokens=100, completion_tokens=200)
        )

        mock_mistral_client.chat.complete = Mock(return_value=mock_response)

        # Test
        response = mistral_v2_client.create({
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "Test"}],
        })

        # Verify usage
        assert response.usage["prompt_tokens"] == 100
        assert response.usage["completion_tokens"] == 200
        assert response.usage["total_tokens"] == 300


class TestMistralV2ClientTransformResponse:
    """Test _transform_response() method."""

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_transform_response_text_content(self, mistral_v2_client):
        """Test transforming response with text content."""
        mock_message = MockMistralMessage(role="assistant", content="Hello world")
        mock_choice = MockMistralChoice(message=mock_message)
        mock_response = MockMistralResponse(
            response_id="test-123",
            model="mistral-small-latest",
            choices=[mock_choice],
            usage=MockMistralUsage(),
        )

        unified_response = mistral_v2_client._transform_response(mock_response, "mistral-small-latest")

        assert isinstance(unified_response, UnifiedResponse)
        assert unified_response.id == "test-123"
        assert unified_response.model == "mistral-small-latest"
        assert unified_response.provider == "mistral"
        assert len(unified_response.messages) == 1
        assert len(unified_response.messages[0].content) == 1
        assert isinstance(unified_response.messages[0].content[0], TextContent)
        assert unified_response.messages[0].content[0].text == "Hello world"

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_transform_response_tool_calls(self, mistral_v2_client):
        """Test transforming response with tool calls."""
        mock_tool_call = MockMistralToolCall(call_id="call_xyz", name="calculate", arguments={"a": 1, "b": 2})
        mock_message = MockMistralMessage(role="assistant", content="", tool_calls=[mock_tool_call])
        mock_choice = MockMistralChoice(message=mock_message, finish_reason="tool_calls")
        mock_response = MockMistralResponse(
            response_id="test-456",
            model="mistral-small-latest",
            choices=[mock_choice],
            usage=MockMistralUsage(),
        )

        unified_response = mistral_v2_client._transform_response(mock_response, "mistral-small-latest")

        assert len(unified_response.messages) == 1
        tool_calls = unified_response.messages[0].get_tool_calls()
        assert len(tool_calls) == 1
        assert isinstance(tool_calls[0], ToolCallContent)
        assert tool_calls[0].id == "call_xyz"
        assert tool_calls[0].name == "calculate"
        # Arguments should be JSON string
        import json

        args_dict = json.loads(tool_calls[0].arguments)
        assert args_dict["a"] == 1
        assert args_dict["b"] == 2

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_transform_response_role_normalization(self, mistral_v2_client):
        """Test that roles are normalized correctly."""
        mock_message = MockMistralMessage(role="assistant", content="Test")
        mock_choice = MockMistralChoice(message=mock_message)
        mock_response = MockMistralResponse(choices=[mock_choice], usage=MockMistralUsage())

        unified_response = mistral_v2_client._transform_response(mock_response, "mistral-small-latest")

        # Role should be normalized
        assert isinstance(unified_response.messages[0].role, UserRoleEnum)
        assert unified_response.messages[0].role == UserRoleEnum.ASSISTANT

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_unified_message_creation(self):
        """Test UnifiedMessage creation and methods."""
        # Create UnifiedMessage with text content
        message = UnifiedMessage(
            role=UserRoleEnum.ASSISTANT,
            content=[TextContent(text="Hello world")],
        )

        assert message.role == UserRoleEnum.ASSISTANT
        assert len(message.content) == 1
        assert isinstance(message.content[0], TextContent)
        assert message.get_text() == "Hello world"
        assert len(message.get_tool_calls()) == 0

        # Create UnifiedMessage with tool calls
        tool_call = ToolCallContent(id="call_123", name="get_weather", arguments='{"city": "SF"}')
        message_with_tools = UnifiedMessage(
            role=UserRoleEnum.ASSISTANT,
            content=[
                TextContent(text="Checking weather"),
                tool_call,
            ],
        )

        # get_text() includes text from all content blocks, including ToolCallContent
        full_text = message_with_tools.get_text()
        assert "Checking weather" in full_text
        assert "get_weather" in full_text  # Tool call text is included

        # Extract only TextContent blocks separately
        text_blocks = [block.text for block in message_with_tools.content if isinstance(block, TextContent)]
        assert text_blocks == ["Checking weather"]

        tool_calls = message_with_tools.get_tool_calls()
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_123"
        assert tool_calls[0].name == "get_weather"


class TestMistralV2ClientCost:
    """Test cost() method."""

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_cost_calculation_mistral_small(self, mistral_v2_client, mock_mistral_client):
        """Test cost calculation for mistral-small-latest model."""
        mock_message = MockMistralMessage(content="Test")
        mock_choice = MockMistralChoice(message=mock_message)
        mock_response = MockMistralResponse(
            model="mistral-small-latest",
            choices=[mock_choice],
            usage=MockMistralUsage(prompt_tokens=1000, completion_tokens=500),
        )
        mock_mistral_client.chat.complete = Mock(return_value=mock_response)

        response = mistral_v2_client.create({
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "Test"}],
        })

        # Verify cost calculation
        assert response.cost is not None
        assert response.cost > 0
        # mistral-small-latest: $0.001/1K input, $0.003/1K output
        expected = (1000 * 0.001 / 1000) + (500 * 0.003 / 1000)
        assert abs(response.cost - expected) < 0.0001

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_cost_calculation_empty_usage(self, mistral_v2_client):
        """Test cost calculation with empty usage."""
        response = UnifiedResponse(
            id="test",
            model="mistral-small-latest",
            provider="mistral",
            messages=[],
            usage={},
        )

        cost = mistral_v2_client.cost(response)
        assert cost == 0.0


class TestMistralV2ClientGetUsage:
    """Test get_usage() method."""

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_get_usage_returns_all_keys(self, mistral_v2_client, mock_mistral_client):
        """Test that get_usage() returns all required keys."""
        mock_message = MockMistralMessage(content="Test")
        mock_choice = MockMistralChoice(message=mock_message)
        mock_response = MockMistralResponse(
            choices=[mock_choice], usage=MockMistralUsage(prompt_tokens=50, completion_tokens=75)
        )
        mock_mistral_client.chat.complete = Mock(return_value=mock_response)

        response = mistral_v2_client.create({
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "Test"}],
        })
        usage = mistral_v2_client.get_usage(response)

        # Verify all required keys
        for key in mistral_v2_client.RESPONSE_USAGE_KEYS:
            assert key in usage

        assert usage["prompt_tokens"] == 50
        assert usage["completion_tokens"] == 75
        assert usage["total_tokens"] == 125
        assert usage["model"] == "mistral-small-latest"
        assert usage["cost"] > 0


class TestMistralV2ClientMessageRetrieval:
    """Test message_retrieval() method."""

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_message_retrieval_simple_text(self, mistral_v2_client, mock_mistral_client):
        """Test retrieving text from simple response."""
        mock_message = MockMistralMessage(content="Hello world")
        mock_choice = MockMistralChoice(message=mock_message)
        mock_response = MockMistralResponse(choices=[mock_choice], usage=MockMistralUsage())
        mock_mistral_client.chat.complete = Mock(return_value=mock_response)

        response = mistral_v2_client.create({
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        messages = mistral_v2_client.message_retrieval(response)

        assert len(messages) == 1
        assert messages[0] == "Hello world"
        assert isinstance(messages[0], str)

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_message_retrieval_with_tool_calls(self, mistral_v2_client, mock_mistral_client):
        """Test retrieving messages with tool calls."""
        mock_tool_call = MockMistralToolCall(call_id="call_123", name="get_weather", arguments={"city": "SF"})
        mock_message = MockMistralMessage(role="assistant", content="Checking weather", tool_calls=[mock_tool_call])
        mock_choice = MockMistralChoice(message=mock_message, finish_reason="tool_calls")
        mock_response = MockMistralResponse(choices=[mock_choice], usage=MockMistralUsage())
        mock_mistral_client.chat.complete = Mock(return_value=mock_response)

        response = mistral_v2_client.create({
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "Get weather"}],
        })
        messages = mistral_v2_client.message_retrieval(response)

        assert len(messages) == 1
        assert isinstance(messages[0], dict)
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Checking weather"
        assert "tool_calls" in messages[0]
        assert len(messages[0]["tool_calls"]) == 1
        assert messages[0]["tool_calls"][0]["id"] == "call_123"
        assert messages[0]["tool_calls"][0]["function"]["name"] == "get_weather"


class TestMistralV2ClientV1Compatible:
    """Test create_v1_compatible() method."""

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_create_v1_compatible_format(self, mistral_v2_client, mock_mistral_client):
        """Test backward compatible response format."""
        mock_message = MockMistralMessage(content="Test response")
        mock_choice = MockMistralChoice(message=mock_message)
        mock_response = MockMistralResponse(choices=[mock_choice], usage=MockMistralUsage())
        mock_mistral_client.chat.complete = Mock(return_value=mock_response)

        # Get v1 compatible response
        response = mistral_v2_client.create_v1_compatible({
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "Test"}],
        })

        # Verify it's a dict with expected structure
        assert isinstance(response, dict)
        assert "id" in response
        assert "model" in response
        assert "choices" in response
        assert "usage" in response
        assert "cost" in response
        assert response["object"] == "chat.completion"
        assert len(response["choices"]) == 1
        assert "message" in response["choices"][0]
        assert response["choices"][0]["message"]["content"] == "Test response"

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_v1_compatible_preserves_tool_calls(self, mistral_v2_client, mock_mistral_client):
        """Test that v1 compatible format preserves tool calls."""
        mock_tool_call = MockMistralToolCall(call_id="call_abc", name="calculate", arguments={"x": 10})
        mock_message = MockMistralMessage(role="assistant", content="", tool_calls=[mock_tool_call])
        mock_choice = MockMistralChoice(message=mock_message, finish_reason="tool_calls")
        mock_response = MockMistralResponse(choices=[mock_choice], usage=MockMistralUsage())
        mock_mistral_client.chat.complete = Mock(return_value=mock_response)

        # Get v1 response
        v1_response = mistral_v2_client.create_v1_compatible({
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "Test"}],
        })

        # V1 format should preserve tool calls
        assert "choices" in v1_response
        assert len(v1_response["choices"]) == 1
        message = v1_response["choices"][0]["message"]
        assert "tool_calls" in message
        assert len(message["tool_calls"]) == 1
        assert message["tool_calls"][0]["id"] == "call_abc"
        assert message["tool_calls"][0]["function"]["name"] == "calculate"
        assert v1_response["choices"][0]["finish_reason"] == "tool_calls"


class TestMistralV2ClientParameterValidation:
    """Test parameter validation."""

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_parse_params_validates_model(self, mistral_v2_client):
        """Test that parse_params requires model."""
        with pytest.raises(AssertionError) as exc_info:
            mistral_v2_client.parse_params({"messages": []})
        assert "model" in str(exc_info.value).lower()

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_parse_params_sets_defaults(self, mistral_v2_client):
        """Test that parse_params sets default values."""
        params = mistral_v2_client.parse_params({
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        assert params["model"] == "mistral-small-latest"
        assert "temperature" in params
        assert "messages" in params


class TestMistralV2ClientBackwardCompatibility:
    """Test backward compatibility with V1 interface."""

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_message_retrieval_v1_compatible(self, mistral_v2_client, mock_mistral_client):
        """Test message_retrieval works with UnifiedResponse (V1 compatibility)."""
        mock_message = MockMistralMessage(content="Test message")
        mock_choice = MockMistralChoice(message=mock_message)
        mock_response = MockMistralResponse(choices=[mock_choice], usage=MockMistralUsage())
        mock_mistral_client.chat.complete = Mock(return_value=mock_response)

        response = mistral_v2_client.create({
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "Test"}],
        })
        messages = mistral_v2_client.message_retrieval(response)

        # Should work with UnifiedResponse (duck typing)
        assert isinstance(messages, list)
        assert len(messages) > 0

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_cost_v1_compatible(self, mistral_v2_client):
        """Test cost() works with UnifiedResponse (V1 compatibility)."""
        response = UnifiedResponse(
            id="test",
            model="mistral-small-latest",
            provider="mistral",
            messages=[],
            usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        )

        cost = mistral_v2_client.cost(response)
        assert isinstance(cost, float)
        assert cost >= 0

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_get_usage_v1_compatible(self, mistral_v2_client):
        """Test get_usage() works with UnifiedResponse (V1 compatibility)."""
        response = UnifiedResponse(
            id="test",
            model="mistral-small-latest",
            provider="mistral",
            messages=[],
            usage={"prompt_tokens": 50, "completion_tokens": 75, "total_tokens": 125},
            cost=0.001,
        )

        usage = mistral_v2_client.get_usage(response)
        assert isinstance(usage, dict)
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert "cost" in usage
        assert "model" in usage


class TestMistralV2ClientIntegration:
    """Integration tests for complete workflows."""

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_full_workflow_with_tool_calls(self, mistral_v2_client, mock_mistral_client):
        """Test complete workflow with tool calls."""
        mock_tool_call = MockMistralToolCall(call_id="call_xyz", name="get_weather", arguments={"city": "Paris"})
        mock_message = MockMistralMessage(
            role="assistant", content="I'll check the weather for you", tool_calls=[mock_tool_call]
        )
        mock_choice = MockMistralChoice(message=mock_message, finish_reason="tool_calls")
        mock_response = MockMistralResponse(
            model="mistral-small-latest",
            choices=[mock_choice],
            usage=MockMistralUsage(prompt_tokens=100, completion_tokens=200),
        )
        mock_mistral_client.chat.complete = Mock(return_value=mock_response)

        # Create request
        response = mistral_v2_client.create({
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "description": "Get weather", "parameters": {}},
                }
            ],
        })

        # Verify all aspects
        assert isinstance(response, UnifiedResponse)
        assert response.model == "mistral-small-latest"
        assert response.provider == "mistral"

        # Check tool calls
        tool_calls = response.messages[0].get_tool_calls()
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"

        # Check text
        assert "weather" in response.text.lower()

        # Check usage
        usage = mistral_v2_client.get_usage(response)
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 200

        # Check cost
        assert response.cost > 0

    @run_for_optional_imports(["mistralai"], "mistral")
    def test_protocol_compliance(self, mistral_v2_client):
        """Test that client implements ModelClientV2 protocol."""
        # Check protocol compliance
        assert hasattr(mistral_v2_client, "RESPONSE_USAGE_KEYS")
        assert callable(mistral_v2_client.create)
        assert callable(mistral_v2_client.create_v1_compatible)
        assert callable(mistral_v2_client.cost)
        assert callable(mistral_v2_client.get_usage)
        assert callable(mistral_v2_client.message_retrieval)
