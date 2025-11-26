# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenAICompletionsClient."""

from unittest.mock import Mock, patch

import pytest

from autogen.llm_clients import OpenAICompletionsClient
from autogen.llm_clients.models import (
    GenericContent,
    ReasoningContent,
    TextContent,
    ToolCallContent,
    UnifiedResponse,
    UserRoleEnum,
)


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

    def test_role_normalization_to_enum(self, mock_openai_client):
        """Test that known roles are normalized to UserRoleEnum."""
        client = OpenAICompletionsClient(api_key="test-key")

        # Create response with known role
        mock_message = MockMessage(role="assistant", content="Test")
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())
        client.client.chat.completions.create = Mock(return_value=mock_response)

        response = client.create({"model": "gpt-4", "messages": []})

        # Verify role is UserRoleEnum
        assert isinstance(response.messages[0].role, UserRoleEnum)
        assert response.messages[0].role == UserRoleEnum.ASSISTANT
        assert response.messages[0].role.value == "assistant"

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

    # Note: test_multimodal_content_with_unknown_type removed
    # OpenAI Chat Completions API always returns content as str, never list
    # List content only exists in REQUEST messages for multimodal inputs,
    # not in RESPONSE messages from the API

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


class TestOpenAICompletionsClientStructuredOutputs:
    """Test structured outputs with Pydantic models and chat.completions.parse()."""

    def test_pydantic_model_detection(self, mock_openai_client):
        """Test that Pydantic BaseModel classes are correctly detected."""
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("Pydantic not installed")

        class TestModel(BaseModel):
            name: str
            age: int

        client = OpenAICompletionsClient(api_key="test-key")

        # Should detect Pydantic model
        assert client._is_pydantic_model(TestModel) is True

        # Should not detect regular classes
        assert client._is_pydantic_model(str) is False
        assert client._is_pydantic_model(dict) is False
        assert client._is_pydantic_model({"type": "json_schema"}) is False
        assert client._is_pydantic_model(None) is False

    def test_structured_output_with_pydantic_model(self, mock_openai_client):
        """Test that response_format with Pydantic model uses parse() method."""
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("Pydantic not installed")

        class QueryAnswer(BaseModel):
            answer: str
            confidence: float

        client = OpenAICompletionsClient(api_key="test-key", response_format=QueryAnswer)

        # Create mock parsed response
        parsed_obj = QueryAnswer(answer="42", confidence=0.95)
        mock_message = MockMessage(role="assistant", content="The answer is 42")
        mock_message.parsed = parsed_obj
        mock_message.refusal = None

        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())

        # Mock parse() method
        client.client.chat.completions.parse = Mock(return_value=mock_response)
        client.client.chat.completions.create = Mock()

        # Test
        response = client.create({"model": "gpt-4o", "messages": [{"role": "user", "content": "What is the answer?"}]})

        # Should call parse() not create()
        assert client.client.chat.completions.parse.called
        assert not client.client.chat.completions.create.called

        # Should have parsed content
        content_blocks = response.messages[0].content
        parsed_blocks = [b for b in content_blocks if b.type == "parsed"]
        assert len(parsed_blocks) == 1
        assert parsed_blocks[0].parsed["answer"] == "42"
        assert parsed_blocks[0].parsed["confidence"] == 0.95

    def test_structured_output_with_json_schema(self, mock_openai_client):
        """Test that JSON schema dict still uses create() method."""
        client = OpenAICompletionsClient(api_key="test-key")

        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        }

        mock_message = MockMessage(role="assistant", content='{"answer": "42"}')
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())

        # Mock create() method
        client.client.chat.completions.create = Mock(return_value=mock_response)
        client.client.chat.completions.parse = Mock()

        # Test
        response = client.create({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "What?"}],
            "response_format": json_schema,
        })

        # Should call create() not parse()
        assert client.client.chat.completions.create.called
        assert not client.client.chat.completions.parse.called

        # Should have text content
        assert response.messages[0].content[0].text == '{"answer": "42"}'

    def test_structured_output_with_refusal(self, mock_openai_client):
        """Test handling of refusals in structured outputs."""
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("Pydantic not installed")

        class SensitiveQuery(BaseModel):
            contains_pii: bool
            explanation: str

        client = OpenAICompletionsClient(api_key="test-key", response_format=SensitiveQuery)

        # Create mock response with refusal
        mock_message = MockMessage(role="assistant", content=None)
        mock_message.parsed = None
        mock_message.refusal = "I cannot process this request as it may contain personally identifiable information."

        mock_choice = MockChoice(message=mock_message, finish_reason="stop")
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())

        client.client.chat.completions.parse = Mock(return_value=mock_response)

        # Test
        response = client.create({"model": "gpt-4o", "messages": [{"role": "user", "content": "Process this PII"}]})

        # Should have refusal content
        content_blocks = response.messages[0].content
        refusal_blocks = [b for b in content_blocks if b.type == "refusal"]
        assert len(refusal_blocks) == 1
        assert "personally identifiable information" in refusal_blocks[0].refusal

    def test_default_response_format_merged_into_params(self, mock_openai_client):
        """Test that default response_format is merged into params."""
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("Pydantic not installed")

        class DefaultModel(BaseModel):
            result: str

        client = OpenAICompletionsClient(api_key="test-key", response_format=DefaultModel)

        parsed_obj = DefaultModel(result="success")
        mock_message = MockMessage(role="assistant", content="Success")
        mock_message.parsed = parsed_obj
        mock_message.refusal = None

        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())

        client.client.chat.completions.parse = Mock(return_value=mock_response)

        # Test without response_format in params
        client.create({"model": "gpt-4o", "messages": [{"role": "user", "content": "Test"}]})

        # Should still call parse() because of default response_format
        assert client.client.chat.completions.parse.called

        # Verify response_format was passed
        call_args = client.client.chat.completions.parse.call_args
        assert call_args[1]["response_format"] == DefaultModel

    def test_explicit_response_format_overrides_default(self, mock_openai_client):
        """Test that explicit response_format in params overrides default."""
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("Pydantic not installed")

        class DefaultModel(BaseModel):
            default_field: str

        class OverrideModel(BaseModel):
            override_field: str

        client = OpenAICompletionsClient(api_key="test-key", response_format=DefaultModel)

        parsed_obj = OverrideModel(override_field="overridden")
        mock_message = MockMessage(role="assistant", content="Overridden")
        mock_message.parsed = parsed_obj
        mock_message.refusal = None

        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())

        client.client.chat.completions.parse = Mock(return_value=mock_response)

        # Test with explicit response_format
        response = client.create({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Test"}],
            "response_format": OverrideModel,
        })

        # Should use OverrideModel, not DefaultModel
        call_args = client.client.chat.completions.parse.call_args
        assert call_args[1]["response_format"] == OverrideModel

        # Check parsed content uses override model
        parsed_blocks = [b for b in response.messages[0].content if b.type == "parsed"]
        assert "override_field" in parsed_blocks[0].parsed

    def test_no_response_format_uses_create(self, mock_openai_client):
        """Test that requests without response_format use create() method."""
        client = OpenAICompletionsClient(api_key="test-key")

        mock_message = MockMessage(role="assistant", content="Regular response")
        mock_choice = MockChoice(message=mock_message)
        mock_response = MockOpenAIResponse(choices=[mock_choice], usage=MockUsage())

        client.client.chat.completions.create = Mock(return_value=mock_response)
        client.client.chat.completions.parse = Mock()

        # Test without response_format
        response = client.create({"model": "gpt-4o", "messages": [{"role": "user", "content": "Test"}]})

        # Should call create() not parse()
        assert client.client.chat.completions.create.called
        assert not client.client.chat.completions.parse.called

        # Should have text content
        assert response.messages[0].content[0].text == "Regular response"
