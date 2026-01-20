# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for BedrockV2Client."""

import base64
import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from autogen.import_utils import run_for_optional_imports
from autogen.llm_clients.bedrock_v2 import BedrockV2Client, BedrockV2EntryDict, BedrockV2LLMConfigEntry
from autogen.llm_clients.models import (
    GenericContent,
    ImageContent,
    ToolCallContent,
    UnifiedResponse,
    UserRoleEnum,
)


@pytest.fixture
def mock_bedrock_runtime():
    """Create mock Bedrock runtime client."""
    with patch("autogen.llm_clients.bedrock_v2.boto3") as mock_boto3:
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_boto3.Session.return_value = mock_session
        mock_boto3.client.return_value = mock_client
        yield mock_client


@pytest.fixture
def bedrock_v2_client(mock_bedrock_runtime):
    """Create BedrockV2Client instance for testing."""
    return BedrockV2Client(
        aws_region="us-east-1",
        aws_access_key="test_key",
        aws_secret_key="test_secret",
    )


class TestBedrockV2ClientInitialization:
    """Test client initialization."""

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_client_with_credentials(self, mock_bedrock_runtime):
        """Test creating client with credentials."""
        client = BedrockV2Client(
            aws_region="us-east-1",
            aws_access_key="test_key",
            aws_secret_key="test_secret",
        )
        assert client is not None
        assert client.bedrock_runtime is not None

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_client_without_credentials(self, mock_bedrock_runtime):
        """Test creating client without credentials (uses IAM role)."""
        client = BedrockV2Client(aws_region="us-east-1")
        assert client is not None

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_client_missing_region(self, mock_bedrock_runtime):
        """Test creating client without region raises error."""
        with patch.dict("os.environ", {}, clear=True), pytest.raises(ValueError, match="Region is required"):
            BedrockV2Client()

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_client_has_required_methods(self, bedrock_v2_client):
        """Test that client has all ModelClientV2 methods."""
        assert hasattr(bedrock_v2_client, "create")
        assert hasattr(bedrock_v2_client, "create_v1_compatible")
        assert hasattr(bedrock_v2_client, "cost")
        assert hasattr(bedrock_v2_client, "get_usage")
        assert hasattr(bedrock_v2_client, "message_retrieval")
        assert hasattr(bedrock_v2_client, "RESPONSE_USAGE_KEYS")


class TestBedrockV2ClientCreate:
    """Test create() method."""

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_simple_text_response(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test creating a simple text response."""
        mock_response = {
            "stopReason": "stop",
            "output": {
                "message": {
                    "content": [{"text": "Hello from Bedrock"}],
                },
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        assert isinstance(response, UnifiedResponse)
        assert response.model == "anthropic.claude-sonnet-4-5-20250929-v1:0"
        assert response.provider == "bedrock"
        assert response.text == "Hello from Bedrock"
        assert response.finish_reason == "stop"
        assert len(response.messages) == 1
        assert isinstance(response.messages[0].role, UserRoleEnum)

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_response_with_tool_calls(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test creating response with tool calls."""
        mock_response = {
            "stopReason": "tool_use",
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "call-123",
                                "name": "get_weather",
                                "input": {"city": "San Francisco"},
                            },
                        },
                    ],
                },
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "Get weather"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                    },
                }
            ],
        })

        tool_calls = response.messages[0].get_tool_calls()
        assert len(tool_calls) == 1
        assert isinstance(tool_calls[0], ToolCallContent)
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].id == "call-123"

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_response_with_image(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test creating response with image content."""
        image_bytes = b"fake_image_data"
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        mock_response = {
            "stopReason": "stop",
            "output": {
                "message": {
                    "content": [
                        {"text": "Here's an image"},
                        {
                            "image": {
                                "format": "jpeg",
                                "source": {"bytes": image_bytes},
                            },
                        },
                    ],
                },
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "Show me an image"}],
        })

        image_blocks = response.get_content_by_type("image")
        assert len(image_blocks) == 1
        assert isinstance(image_blocks[0], ImageContent)
        assert image_blocks[0].data_uri.startswith("data:image/jpeg;base64,")

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_response_with_unknown_content(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test creating response with unknown content type."""
        mock_response = {
            "stopReason": "stop",
            "output": {
                "message": {
                    "content": [
                        {"text": "Hello"},
                        {"unknownType": {"customField": "value"}},
                    ],
                },
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        generic_blocks = [b for b in response.messages[0].content if isinstance(b, GenericContent)]
        assert len(generic_blocks) == 1
        assert generic_blocks[0].type == "unknown"

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_with_system_messages(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test create() with system messages when supports_system_prompts is True."""
        mock_response = {
            "stopReason": "stop",
            "output": {
                "message": {
                    "content": [{"text": "Hello"}],
                },
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
            "supports_system_prompts": True,
        })

        # Verify system was passed to converse
        call_args = mock_bedrock_runtime.converse.call_args
        assert call_args is not None
        assert "system" in call_args.kwargs
        assert isinstance(response, UnifiedResponse)

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_without_system_prompts_support(self, mock_bedrock_runtime):
        """Test create() when supports_system_prompts is False."""
        client = BedrockV2Client(
            aws_region="us-east-1",
            aws_access_key="test_key",
            aws_secret_key="test_secret",
        )

        mock_response = {
            "stopReason": "stop",
            "output": {
                "message": {
                    "content": [{"text": "Hello"}],
                },
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
            "supports_system_prompts": False,
        })

        # Verify system was NOT passed to converse
        call_args = mock_bedrock_runtime.converse.call_args
        assert call_args is not None
        assert "system" not in call_args.kwargs
        assert isinstance(response, UnifiedResponse)

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_with_inference_config(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test create() with inference config parameters."""
        mock_response = {
            "stopReason": "stop",
            "output": {
                "message": {
                    "content": [{"text": "Hello"}],
                },
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100,
        })

        # Verify inferenceConfig was passed
        call_args = mock_bedrock_runtime.converse.call_args
        assert call_args is not None
        assert "inferenceConfig" in call_args.kwargs
        inference_config = call_args.kwargs["inferenceConfig"]
        assert "temperature" in inference_config or "maxTokens" in inference_config
        assert isinstance(response, UnifiedResponse)

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_with_additional_model_request_fields(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test create() with additionalModelRequestFields."""
        mock_response = {
            "stopReason": "stop",
            "output": {
                "message": {
                    "content": [{"text": "Hello"}],
                },
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "Hello"}],
            "additional_model_request_fields": {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 1024,
                },
            },
        })

        # Verify additionalModelRequestFields was passed
        call_args = mock_bedrock_runtime.converse.call_args
        assert call_args is not None
        assert "additionalModelRequestFields" in call_args.kwargs
        additional_fields = call_args.kwargs["additionalModelRequestFields"]
        assert "thinking" in additional_fields
        assert isinstance(response, UnifiedResponse)

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_with_tools_and_response_format(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test create() with both user tools and response format."""
        class Answer(BaseModel):
            answer: str

        mock_response = {
            "stopReason": "tool_use",
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "call-123",
                                "name": "__structured_output",
                                "input": {"answer": "42"},
                            },
                        },
                    ],
                },
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        client = BedrockV2Client(
            aws_region="us-east-1",
            aws_access_key="test_key",
            aws_secret_key="test_secret",
            response_format=Answer,
        )

        response = client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "What is the answer?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                    },
                }
            ],
        })

        # Verify toolConfig was passed with both tools
        call_args = mock_bedrock_runtime.converse.call_args
        assert call_args is not None
        assert "toolConfig" in call_args.kwargs
        tool_config = call_args.kwargs["toolConfig"]
        assert len(tool_config["tools"]) == 2  # user tool + structured output tool
        assert isinstance(response, UnifiedResponse)

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_with_price_parameter(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test create() with price parameter for cost calculation."""
        mock_response = {
            "stopReason": "stop",
            "output": {
                "message": {
                    "content": [{"text": "Test"}],
                },
            },
            "usage": {"inputTokens": 1000, "outputTokens": 500, "totalTokens": 1500},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "price": [0.01, 0.02],
        })

        # Verify price was stored for cost calculation
        assert response.cost > 0
        expected_cost = (1000 / 1000) * 0.01 + (500 / 1000) * 0.02
        assert abs(response.cost - expected_cost) < 0.001

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_with_empty_base_params(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test create() when base_params is empty (no inferenceConfig)."""
        mock_response = {
            "stopReason": "stop",
            "output": {
                "message": {
                    "content": [{"text": "Hello"}],
                },
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        # Verify inferenceConfig was NOT passed when empty
        call_args = mock_bedrock_runtime.converse.call_args
        assert call_args is not None
        assert "inferenceConfig" not in call_args.kwargs
        assert isinstance(response, UnifiedResponse)

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_with_empty_additional_params(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test create() when additional_params is empty."""
        mock_response = {
            "stopReason": "stop",
            "output": {
                "message": {
                    "content": [{"text": "Hello"}],
                },
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        # Verify additionalModelRequestFields was NOT passed when empty
        call_args = mock_bedrock_runtime.converse.call_args
        assert call_args is not None
        assert "additionalModelRequestFields" not in call_args.kwargs
        assert isinstance(response, UnifiedResponse)

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_with_no_tools(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test create() when no tools are provided."""
        mock_response = {
            "stopReason": "stop",
            "output": {
                "message": {
                    "content": [{"text": "Hello"}],
                },
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        # Verify toolConfig was NOT passed when no tools
        call_args = mock_bedrock_runtime.converse.call_args
        assert call_args is not None
        assert "toolConfig" not in call_args.kwargs
        assert isinstance(response, UnifiedResponse)

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_response_is_none(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test create() when response is None raises RuntimeError."""
        mock_bedrock_runtime.converse.return_value = None

        with pytest.raises(RuntimeError, match="Failed to get response from Bedrock"):
            bedrock_v2_client.create({
                "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
                "messages": [{"role": "user", "content": "Hello"}],
            })

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_with_response_format_no_user_tools(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test create() with response format but no user tools."""
        class Answer(BaseModel):
            answer: str

        mock_response = {
            "stopReason": "tool_use",
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "call-123",
                                "name": "__structured_output",
                                "input": {"answer": "42"},
                            },
                        },
                    ],
                },
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        client = BedrockV2Client(
            aws_region="us-east-1",
            aws_access_key="test_key",
            aws_secret_key="test_secret",
            response_format=Answer,
        )

        response = client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "What is the answer?"}],
        })

        # Verify toolConfig was passed with only structured output tool
        call_args = mock_bedrock_runtime.converse.call_args
        assert call_args is not None
        assert "toolConfig" in call_args.kwargs
        tool_config = call_args.kwargs["toolConfig"]
        assert len(tool_config["tools"]) == 1  # only structured output tool
        assert tool_config["tools"][0]["toolSpec"]["name"] == "__structured_output"
        assert isinstance(response, UnifiedResponse)

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_with_all_parameters(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test create() with all optional parameters set."""
        mock_response = {
            "stopReason": "stop",
            "output": {
                "message": {
                    "content": [{"text": "Hello"}],
                },
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            "temperature": 0.7,
            "max_tokens": 100,
            "supports_system_prompts": True,
            "price": [0.01, 0.02],
            "additional_model_request_fields": {"custom": "value"},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "description": "Test",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        })

        # Verify all parameters were passed correctly
        call_args = mock_bedrock_runtime.converse.call_args
        assert call_args is not None
        assert "inferenceConfig" in call_args.kwargs
        assert "additionalModelRequestFields" in call_args.kwargs
        assert "system" in call_args.kwargs
        assert "toolConfig" in call_args.kwargs
        assert isinstance(response, UnifiedResponse)


class TestBedrockV2ClientStructuredOutputs:
    """Test structured outputs."""

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_structured_output_with_pydantic_model(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test structured output with Pydantic model."""

        class Answer(BaseModel):
            answer: str
            confidence: float

        mock_response = {
            "stopReason": "tool_use",
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "call-123",
                                "name": "__structured_output",
                                "input": {"answer": "42", "confidence": 0.95},
                            },
                        },
                    ],
                },
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "What is the answer?"}],
            "response_format": Answer,
        })

        assert response.text
        assert "42" in response.text
        assert "0.95" in response.text

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_structured_output_with_json_schema(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test structured output with JSON schema dict."""
        schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        }

        mock_response = {
            "stopReason": "tool_use",
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "call-123",
                                "name": "__structured_output",
                                "input": {"result": "success"},
                            },
                        },
                    ],
                },
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "Test"}],
            "response_format": schema,
        })

        assert response.text
        assert "success" in response.text


class TestBedrockV2ClientV1Compatible:
    """Test create_v1_compatible() method."""

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_v1_compatible_format(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test backward compatible response format."""
        mock_response = {
            "stopReason": "stop",
            "output": {
                "message": {
                    "content": [{"text": "Test response"}],
                },
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create_v1_compatible({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        assert isinstance(response, dict)
        assert "id" in response
        assert "model" in response
        assert "choices" in response
        assert "usage" in response
        assert "cost" in response
        assert response["object"] == "chat.completion"

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_v1_compatible_with_tool_calls(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test v1 compatible format with tool calls."""
        mock_response = {
            "stopReason": "tool_use",
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "call-123",
                                "name": "get_weather",
                                "input": {"city": "SF"},
                            },
                        },
                    ],
                },
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create_v1_compatible({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "Get weather"}],
        })

        assert "tool_calls" in response["choices"][0]["message"]
        assert len(response["choices"][0]["message"]["tool_calls"]) == 1


class TestBedrockV2ClientCost:
    """Test cost() method."""

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_cost_calculation_with_custom_price(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test cost calculation with custom price."""
        mock_response = {
            "stopReason": "stop",
            "output": {"message": {"content": [{"text": "Test"}]}},
            "usage": {"inputTokens": 1000, "outputTokens": 500, "totalTokens": 1500},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "price": [0.01, 0.02],  # $0.01/1K input, $0.02/1K output
        })

        expected_cost = (1000 / 1000) * 0.01 + (500 / 1000) * 0.02
        assert abs(response.cost - expected_cost) < 0.001

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_cost_calculation_without_price(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test cost calculation falls back to default pricing."""
        mock_response = {
            "stopReason": "stop",
            "output": {"message": {"content": [{"text": "Test"}]}},
            "usage": {"inputTokens": 100, "outputTokens": 100, "totalTokens": 200},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        assert response.cost >= 0


class TestBedrockV2ClientGetUsage:
    """Test get_usage() method."""

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_get_usage_returns_all_keys(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test that get_usage() returns all required keys."""
        mock_response = {
            "stopReason": "stop",
            "output": {"message": {"content": [{"text": "Test"}]}},
            "usage": {"inputTokens": 50, "outputTokens": 75, "totalTokens": 125},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        usage = bedrock_v2_client.get_usage(response)

        for key in bedrock_v2_client.RESPONSE_USAGE_KEYS:
            assert key in usage

        assert usage["prompt_tokens"] == 50
        assert usage["completion_tokens"] == 75
        assert usage["total_tokens"] == 125
        assert usage["model"] == "anthropic.claude-sonnet-4-5-20250929-v1:0"


class TestBedrockV2ClientMessageRetrieval:
    """Test message_retrieval() method."""

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_message_retrieval_simple_text(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test retrieving text from simple response."""
        mock_response = {
            "stopReason": "stop",
            "output": {"message": {"content": [{"text": "Hello world"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        messages = bedrock_v2_client.message_retrieval(response)

        assert len(messages) == 1
        assert messages[0] == "Hello world"

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_message_retrieval_with_tool_calls(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test retrieving messages with tool calls."""
        mock_response = {
            "stopReason": "tool_use",
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "call-123",
                                "name": "get_weather",
                                "input": {"city": "SF"},
                            },
                        },
                    ],
                },
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "Get weather"}],
        })
        messages = bedrock_v2_client.message_retrieval(response)

        assert len(messages) == 1
        assert isinstance(messages[0], dict)
        assert "tool_calls" in messages[0]


class TestBedrockV2ClientStructuredOutputHelpers:
    """Test structured output helper methods."""

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_get_response_format_schema_pydantic(self, bedrock_v2_client):
        """Test schema extraction from Pydantic model."""

        class TestModel(BaseModel):
            field1: str
            field2: int

        schema = bedrock_v2_client._get_response_format_schema(TestModel)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "field1" in schema["properties"]
        assert "field2" in schema["properties"]

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_get_response_format_schema_dict(self, bedrock_v2_client):
        """Test schema extraction from dict."""
        schema_dict = {"type": "object", "properties": {"test": {"type": "string"}}}
        schema = bedrock_v2_client._get_response_format_schema(schema_dict)
        # The method adds 'required' field if missing
        assert schema["type"] == "object"
        assert schema["properties"] == {"test": {"type": "string"}}
        assert "required" in schema
        assert schema["required"] == []

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_structured_output_tool(self, bedrock_v2_client):
        """Test creating structured output tool."""
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        tool = bedrock_v2_client._create_structured_output_tool(schema)
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "__structured_output"
        assert "parameters" in tool["function"]

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_extract_structured_output_from_tool_call(self, bedrock_v2_client):
        """Test extracting structured output from tool call."""
        from autogen.oai.oai_models import ChatCompletionMessageToolCall

        tool_calls = [
            ChatCompletionMessageToolCall(
                id="call-123",
                function={
                    "name": "__structured_output",
                    "arguments": json.dumps({"answer": "42", "confidence": 0.95}),
                },
                type="function",
            ),
        ]
        result = bedrock_v2_client._extract_structured_output_from_tool_call(tool_calls)
        assert result == {"answer": "42", "confidence": 0.95}

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_extract_structured_output_not_found(self, bedrock_v2_client):
        """Test extracting structured output when not found."""
        tool_calls = [Mock(function=Mock(name="other_tool", arguments="{}"))]
        result = bedrock_v2_client._extract_structured_output_from_tool_call(tool_calls)
        assert result is None


class TestBedrockV2EntryDict:
    """Test BedrockV2EntryDict."""

    def test_bedrock_v2_entry_dict_creation(self):
        """Test creating BedrockV2EntryDict."""
        entry = BedrockV2EntryDict(
            api_type="bedrock_v2",
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            aws_region="us-east-1",
        )
        assert entry["api_type"] == "bedrock_v2"
        assert entry["aws_region"] == "us-east-1"


class TestBedrockV2LLMConfigEntry:
    """Test BedrockV2LLMConfigEntry."""

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_bedrock_v2_llm_config_entry_creation(self):
        """Test creating BedrockV2LLMConfigEntry."""
        entry = BedrockV2LLMConfigEntry(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            aws_region="us-east-1",
        )
        assert entry.api_type == "bedrock_v2"
        assert entry.aws_region == "us-east-1"

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_bedrock_v2_llm_config_entry_create_client(self):
        """Test create_client() method."""
        entry = BedrockV2LLMConfigEntry(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            aws_region="us-east-1",
            aws_access_key="test_key",
            aws_secret_key="test_secret",
        )
        client = entry.create_client()
        assert isinstance(client, BedrockV2Client)


class TestBedrockV2ClientErrorHandling:
    """Test error handling."""

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_missing_model(self, bedrock_v2_client):
        """Test create() with missing model raises error."""
        with pytest.raises(ValueError, match="Please provide the 'model'"):
            bedrock_v2_client.create({"messages": []})

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_create_api_error(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test handling API errors."""
        mock_bedrock_runtime.converse.side_effect = Exception("API Error")
        with pytest.raises(Exception, match="API Error"):
            bedrock_v2_client.create({
                "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
                "messages": [{"role": "user", "content": "Hello"}],
            })


class TestBedrockV2ClientProtocolCompliance:
    """Test ModelClientV2 protocol compliance."""

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_protocol_compliance(self, bedrock_v2_client):
        """Test that client implements ModelClientV2 protocol."""
        assert hasattr(bedrock_v2_client, "RESPONSE_USAGE_KEYS")
        assert callable(bedrock_v2_client.create)
        assert callable(bedrock_v2_client.create_v1_compatible)
        assert callable(bedrock_v2_client.cost)
        assert callable(bedrock_v2_client.get_usage)
        assert callable(bedrock_v2_client.message_retrieval)

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_full_workflow(self, bedrock_v2_client, mock_bedrock_runtime):
        """Test complete workflow from create to usage extraction."""
        mock_response = {
            "stopReason": "stop",
            "output": {"message": {"content": [{"text": "Test response"}]}},
            "usage": {"inputTokens": 100, "outputTokens": 200, "totalTokens": 300},
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        response = bedrock_v2_client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        assert isinstance(response, UnifiedResponse)
        usage = bedrock_v2_client.get_usage(response)
        assert usage["total_tokens"] == 300
        cost = bedrock_v2_client.cost(response)
        assert cost >= 0
        messages = bedrock_v2_client.message_retrieval(response)
        assert len(messages) == 1
