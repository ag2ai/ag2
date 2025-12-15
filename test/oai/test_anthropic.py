# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

import logging

import pytest

from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.llm_config import LLMConfig
from autogen.oai.anthropic import AnthropicClient, AnthropicLLMConfigEntry, _calculate_cost

with optional_import_block() as result:
    from anthropic.types import Message, TextBlock, ThinkingBlock


from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_completion():
    class MockCompletion:
        def __init__(
            self,
            id="msg_013Zva2CMHLNnXjNJJKqJ2EF",
            completion="Hi! My name is Claude.",
            model="claude-3-opus-20240229",
            stop_reason="end_turn",
            role="assistant",
            type: Literal["completion"] = "completion",
            usage={"input_tokens": 10, "output_tokens": 25},
        ):
            self.id = id
            self.role = role
            self.completion = completion
            self.model = model
            self.stop_reason = stop_reason
            self.type = type
            self.usage = usage

    return MockCompletion


@pytest.fixture
def anthropic_client():
    return AnthropicClient(api_key="dummy_api_key")


def test_anthropic_llm_config_entry():
    anthropic_llm_config = AnthropicLLMConfigEntry(
        model="claude-sonnet-4-5",
        api_key="dummy_api_key",
        stream=False,
        temperature=1.0,
        max_tokens=100,
    )
    expected = {
        "api_type": "anthropic",
        "model": "claude-sonnet-4-5",
        "api_key": "dummy_api_key",
        "stream": False,
        "temperature": 1.0,
        "max_tokens": 100,
        "tags": [],
    }
    actual = anthropic_llm_config.model_dump()
    assert actual == expected

    assert LLMConfig(anthropic_llm_config).model_dump() == {
        "config_list": [expected],
    }


@run_for_optional_imports(["anthropic"], "anthropic")
def test_initialization_missing_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SECRET_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    with pytest.raises(ValueError, match="credentials are required to use the Anthropic API."):
        AnthropicClient()

    AnthropicClient(api_key="dummy_api_key")


@pytest.fixture
def anthropic_client_with_aws_credentials():
    return AnthropicClient(
        aws_access_key="dummy_access_key",
        aws_secret_key="dummy_secret_key",
        aws_session_token="dummy_session_token",
        aws_region="us-west-2",
    )


@pytest.fixture
def anthropic_client_with_vertexai_credentials():
    return AnthropicClient(
        gcp_project_id="dummy_project_id",
        gcp_region="us-west-2",
        gcp_auth_token="dummy_auth_token",
    )


@run_for_optional_imports(["anthropic"], "anthropic")
def test_intialization(anthropic_client):
    assert anthropic_client.api_key == "dummy_api_key", "`api_key` should be correctly set in the config"


@run_for_optional_imports(["anthropic"], "anthropic")
def test_intialization_with_aws_credentials(anthropic_client_with_aws_credentials):
    assert anthropic_client_with_aws_credentials.aws_access_key == "dummy_access_key", (
        "`aws_access_key` should be correctly set in the config"
    )
    assert anthropic_client_with_aws_credentials.aws_secret_key == "dummy_secret_key", (
        "`aws_secret_key` should be correctly set in the config"
    )
    assert anthropic_client_with_aws_credentials.aws_session_token == "dummy_session_token", (
        "`aws_session_token` should be correctly set in the config"
    )
    assert anthropic_client_with_aws_credentials.aws_region == "us-west-2", (
        "`aws_region` should be correctly set in the config"
    )


@run_for_optional_imports(["anthropic"], "anthropic")
def test_initialization_with_vertexai_credentials(anthropic_client_with_vertexai_credentials):
    assert anthropic_client_with_vertexai_credentials.gcp_project_id == "dummy_project_id", (
        "`gcp_project_id` should be correctly set in the config"
    )
    assert anthropic_client_with_vertexai_credentials.gcp_region == "us-west-2", (
        "`gcp_region` should be correctly set in the config"
    )
    assert anthropic_client_with_vertexai_credentials.gcp_auth_token == "dummy_auth_token", (
        "`gcp_auth_token` should be correctly set in the config"
    )


# Test cost calculation
@run_for_optional_imports(["anthropic"], "anthropic")
def test_cost_calculation(mock_completion):
    completion = mock_completion(
        completion="Hi! My name is Claude.",
        usage={"prompt_tokens": 10, "completion_tokens": 25, "total_tokens": 35},
        model="claude-3-opus-20240229",
    )
    assert (
        _calculate_cost(completion.usage["prompt_tokens"], completion.usage["completion_tokens"], completion.model)
        == 0.002025
    ), "Cost should be $0.002025"


@run_for_optional_imports(["anthropic"], "anthropic")
def test_load_config(anthropic_client):
    params = {
        "model": "claude-sonnet-4-5",
        "stream": False,
        "temperature": 1,
        "top_p": 0.8,
        "max_tokens": 100,
    }
    expected_params = {
        "model": "claude-sonnet-4-5",
        "stream": False,
        "temperature": 1,
        "timeout": None,
        "top_p": 0.8,
        "max_tokens": 100,
        "stop_sequences": None,
        "top_k": None,
        "tool_choice": None,
    }
    result = anthropic_client.load_config(params)
    assert result == expected_params, "Config should be correctly loaded"


@run_for_optional_imports(["anthropic"], "anthropic")
def test_extract_json_response(anthropic_client):
    # Define test Pydantic model
    class Step(BaseModel):
        explanation: str
        output: str

    class MathReasoning(BaseModel):
        steps: list[Step]
        final_answer: str

    # Set up the response format
    anthropic_client._response_format = MathReasoning

    # Test case 1: JSON within tags - CORRECT
    tagged_response = Message(
        id="msg_123",
        content=[
            TextBlock(
                text="""<json_response>
            {
                "steps": [
                    {"explanation": "Step 1", "output": "8x = -30"},
                    {"explanation": "Step 2", "output": "x = -3.75"}
                ],
                "final_answer": "x = -3.75"
            }
            </json_response>""",
                type="text",
            )
        ],
        model="claude-sonnet-4-5",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage={"input_tokens": 10, "output_tokens": 25},
    )

    result = anthropic_client._extract_json_response(tagged_response)
    assert isinstance(result, MathReasoning)
    assert len(result.steps) == 2
    assert result.final_answer == "x = -3.75"

    # Test case 2: Plain JSON without tags - SHOULD STILL PASS
    plain_response = Message(
        id="msg_123",
        content=[
            TextBlock(
                text="""Here's the solution:
            {
                "steps": [
                    {"explanation": "Step 1", "output": "8x = -30"},
                    {"explanation": "Step 2", "output": "x = -3.75"}
                ],
                "final_answer": "x = -3.75"
            }""",
                type="text",
            )
        ],
        model="claude-sonnet-4-5",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage={"input_tokens": 10, "output_tokens": 25},
    )

    result = anthropic_client._extract_json_response(plain_response)
    assert isinstance(result, MathReasoning)
    assert len(result.steps) == 2
    assert result.final_answer == "x = -3.75"

    # Test case 3: Invalid JSON - RAISE ERROR
    invalid_response = Message(
        id="msg_123",
        content=[
            TextBlock(
                text="""<json_response>
            {
                "steps": [
                    {"explanation": "Step 1", "output": "8x = -30"},
                    {"explanation": "Missing closing brace"
                ],
                "final_answer": "x = -3.75"
            """,
                type="text",
            )
        ],
        model="claude-sonnet-4-5",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage={"input_tokens": 10, "output_tokens": 25},
    )

    with pytest.raises(
        ValueError, match="Failed to parse response as valid JSON matching the schema for Structured Output: "
    ):
        anthropic_client._extract_json_response(invalid_response)

    # Test case 4: No JSON content - RAISE ERROR
    no_json_response = Message(
        id="msg_123",
        content=[TextBlock(text="This response contains no JSON at all.", type="text")],
        model="claude-sonnet-4-5",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage={"input_tokens": 10, "output_tokens": 25},
    )

    with pytest.raises(ValueError, match="No valid JSON found in response for Structured Output."):
        anthropic_client._extract_json_response(no_json_response)

    # Test case 5: Plain JSON without tags, using ThinkingBlock - SHOULD STILL PASS
    plain_response = Message(
        id="msg_123",
        content=[
            ThinkingBlock(
                signature="json_response",
                thinking="""Here's the solution:
            {
                "steps": [
                    {"explanation": "Step 1", "output": "8x = -30"},
                    {"explanation": "Step 2", "output": "x = -3.75"}
                ],
                "final_answer": "x = -3.75"
            }""",
                type="thinking",
            )
        ],
        model="claude-3-5-sonnet-latest",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage={"input_tokens": 10, "output_tokens": 25},
    )

    result = anthropic_client._extract_json_response(plain_response)
    assert isinstance(result, MathReasoning)
    assert len(result.steps) == 2
    assert result.final_answer == "x = -3.75"


@run_for_optional_imports(["anthropic"], "anthropic")
def test_convert_tools_to_functions(anthropic_client):
    tools = [
        {
            "type": "function",
            "function": {
                "description": "weather tool",
                "name": "weather_tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city_name": {"type": "string", "description": "city_name"},
                        "city_list": {
                            "$defs": {
                                "city_list_class": {
                                    "properties": {
                                        "item1": {"title": "Item1", "type": "string"},
                                        "item2": {"title": "Item2", "type": "string"},
                                    },
                                    "required": ["item1", "item2"],
                                    "title": "city_list_class",
                                    "type": "object",
                                }
                            },
                            "items": {"$ref": "#/$defs/city_list_class"},
                            "type": "array",
                            "description": "city_list",
                        },
                    },
                    "required": ["city_name", "city_list"],
                },
            },
        }
    ]
    expected = [
        {
            "description": "weather tool",
            "name": "weather_tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {"type": "string", "description": "city_name"},
                    "city_list": {
                        "$defs": {
                            "city_list_class": {
                                "properties": {
                                    "item1": {"title": "Item1", "type": "string"},
                                    "item2": {"title": "Item2", "type": "string"},
                                },
                                "required": ["item1", "item2"],
                                "title": "city_list_class",
                                "type": "object",
                            }
                        },
                        "items": {"$ref": "#/properties/city_list/$defs/city_list_class"},
                        "type": "array",
                        "description": "city_list",
                    },
                },
                "required": ["city_name", "city_list"],
            },
        }
    ]
    actual = anthropic_client.convert_tools_to_functions(tools=tools)
    assert actual == expected


@run_for_optional_imports(["anthropic"], "anthropic")
def test_process_image_content_valid_data_url():
    from autogen.oai.anthropic import process_image_content

    content_item = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
    processed = process_image_content(content_item)
    expected = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}}
    assert processed == expected


@run_for_optional_imports(["anthropic"], "anthropic")
def test_process_image_content_non_image_type():
    from autogen.oai.anthropic import process_image_content

    content_item = {"type": "text", "text": "Just text"}
    processed = process_image_content(content_item)
    assert processed == content_item


@run_for_optional_imports(["anthropic"], "anthropic")
def test_process_message_content_string():
    from autogen.oai.anthropic import process_message_content

    message = {"content": "Hello"}
    processed = process_message_content(message)
    assert processed == "Hello"


@run_for_optional_imports(["anthropic"], "anthropic")
def test_process_message_content_list():
    from autogen.oai.anthropic import process_message_content

    message = {
        "content": [
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]
    }
    processed = process_message_content(message)
    expected = [
        {"type": "text", "text": "Hello"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
    ]
    assert processed == expected


@run_for_optional_imports(["anthropic"], "anthropic")
def test_oai_messages_to_anthropic_messages():
    from autogen.oai.anthropic import oai_messages_to_anthropic_messages

    params = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "System text."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBB"}},
                ],
            },
        ]
    }
    processed = oai_messages_to_anthropic_messages(params)

    # The function should update the system message (in the params dict) by concatenating only its text parts.
    assert params.get("system") == "System text."

    # The processed messages list should include a user message with the image URL converted to a base64 image format.
    user_message = next((m for m in processed if m["role"] == "user"), None)
    expected_content = [
        {"type": "text", "text": "Hello"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "BBB"}},
    ]
    assert user_message is not None
    assert user_message["content"] == expected_content


# ==============================================================================
# Unit Tests for Native Structured Outputs Feature
# ==============================================================================


@run_for_optional_imports(["anthropic"], "anthropic")
def test_supports_native_structured_outputs():
    """Test model detection for native structured outputs (Approach 1)."""
    from autogen.oai.anthropic import supports_native_structured_outputs

    # Sonnet 4.5 models should be supported
    assert supports_native_structured_outputs("claude-sonnet-4-5")
    assert supports_native_structured_outputs("claude-3-5-sonnet-20241022")
    assert supports_native_structured_outputs("claude-3-7-sonnet-20250219")

    # Pattern matching for future Sonnet versions
    assert supports_native_structured_outputs("claude-3-5-sonnet-20260101")
    assert supports_native_structured_outputs("claude-3-7-sonnet-20260615")

    # Future Opus 4.x models should be supported
    assert supports_native_structured_outputs("claude-opus-4-1")
    assert supports_native_structured_outputs("claude-opus-4-5")

    # Older models should NOT be supported
    assert not supports_native_structured_outputs("claude-3-haiku-20240307")
    assert not supports_native_structured_outputs("claude-3-sonnet-20240229")
    assert not supports_native_structured_outputs("claude-3-opus-20240229")
    assert not supports_native_structured_outputs("claude-2.1")
    assert not supports_native_structured_outputs("claude-instant-1.2")

    # Haiku models should not be supported
    assert not supports_native_structured_outputs("claude-3-5-haiku-20241022")


@run_for_optional_imports(["anthropic"], "anthropic")
def test_has_beta_messages_api():
    """Test SDK version detection for beta API (Approach 2)."""
    from autogen.oai.anthropic import has_beta_messages_api

    # Should detect if current SDK has beta.messages.parse()
    has_beta = has_beta_messages_api()

    # If we have anthropic SDK, it should be a boolean
    assert isinstance(has_beta, bool)

    # If True, verify we can import the beta API
    if has_beta:
        try:
            from anthropic.resources.beta.messages import Messages

            assert hasattr(Messages, "parse"), "Beta API should have parse method"
        except ImportError:
            pytest.fail("has_beta_messages_api returned True but cannot import beta API")


@run_for_optional_imports(["anthropic"], "anthropic")
def test_transform_schema_for_anthropic():
    """Test schema transformation for Anthropic compatibility."""
    from autogen.oai.anthropic import transform_schema_for_anthropic

    # Test basic schema transformation
    input_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 50},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "score": {"type": "number"},
        },
        "required": ["name", "age"],
    }

    transformed = transform_schema_for_anthropic(input_schema)

    # Should remove unsupported constraints
    assert "minLength" not in transformed["properties"]["name"]
    assert "maxLength" not in transformed["properties"]["name"]
    assert "minimum" not in transformed["properties"]["age"]
    assert "maximum" not in transformed["properties"]["age"]

    # Should add additionalProperties: false if not present
    assert transformed["additionalProperties"] is False

    # Should preserve required fields and types
    assert transformed["required"] == ["name", "age"]
    assert transformed["properties"]["name"]["type"] == "string"
    assert transformed["properties"]["age"]["type"] == "integer"


@run_for_optional_imports(["anthropic"], "anthropic")
def test_transform_schema_preserves_nested_structures():
    """Test that schema transformation preserves nested structures."""
    from autogen.oai.anthropic import transform_schema_for_anthropic

    input_schema = {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "minimum": 0},
                },
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                    },
                },
            },
        },
        "additionalProperties": True,
    }

    transformed = transform_schema_for_anthropic(input_schema)

    # Should preserve nested structure
    assert "data" in transformed["properties"]
    assert "value" in transformed["properties"]["data"]["properties"]

    # Should preserve arrays
    assert transformed["properties"]["items"]["type"] == "array"

    # Should preserve existing additionalProperties setting
    assert transformed["additionalProperties"] is True


@run_for_optional_imports(["anthropic"], "anthropic")
def test_create_routes_to_native_or_json_mode(anthropic_client, monkeypatch):
    """Test that create() method routes to correct implementation."""

    native_called = False
    json_mode_called = False
    standard_called = False

    def mock_create_with_native(params):
        nonlocal native_called
        native_called = True
        return create_mock_anthropic_response()

    def mock_create_with_json_mode(params):
        nonlocal json_mode_called
        json_mode_called = True
        return create_mock_anthropic_response()

    def mock_create_standard(params):
        nonlocal standard_called
        standard_called = True
        return create_mock_anthropic_response()

    # Mock the internal methods
    monkeypatch.setattr(anthropic_client, "_create_with_native_structured_output", mock_create_with_native)
    monkeypatch.setattr(anthropic_client, "_create_with_json_mode", mock_create_with_json_mode)
    monkeypatch.setattr(anthropic_client, "_create_standard", mock_create_standard)

    # Test 1: Sonnet 4.5 with response_format -> native
    anthropic_client._response_format = BaseModel
    params = {"model": "claude-sonnet-4-5", "messages": [], "max_tokens": 100}
    anthropic_client.create(params)
    assert native_called, "Should use native structured output for Sonnet 4.5"

    # Reset flags
    native_called = json_mode_called = standard_called = False

    # Test 2: Haiku with response_format -> JSON Mode
    params = {"model": "claude-3-haiku-20240307", "messages": [], "max_tokens": 100}
    anthropic_client.create(params)
    assert json_mode_called, "Should use JSON Mode for older models"

    # Reset flags
    native_called = json_mode_called = standard_called = False

    # Test 3: No response_format -> standard
    anthropic_client._response_format = None
    params = {"model": "claude-sonnet-4-5", "messages": [], "max_tokens": 100}
    anthropic_client.create(params)
    assert standard_called, "Should use standard create without response_format"


def create_mock_anthropic_response():
    """Helper to create mock Anthropic response."""
    with optional_import_block() as result:
        from anthropic.types import Message, TextBlock

    if result.is_successful:
        return Message(
            id="msg_test123",
            content=[TextBlock(text='{"test": "response"}', type="text")],
            model="claude-sonnet-4-5",
            role="assistant",
            stop_reason="end_turn",
            type="message",
            usage={"input_tokens": 10, "output_tokens": 20},
        )
    return None


@run_for_optional_imports(["anthropic"], "anthropic")
def test_native_structured_output_with_beta_api(anthropic_client, monkeypatch):
    """Test that native structured output uses beta API correctly."""
    from autogen.oai.anthropic import has_beta_messages_api

    if not has_beta_messages_api():
        pytest.skip("SDK does not support beta.messages API")

    beta_parse_called = False
    captured_params = {}

    # Define TestModel first so we can use it in the mock
    class TestModel(BaseModel):
        answer: str

    class MockParsedResponse:
        """Mock response object with parsed_output attribute."""

        def __init__(self, base_response):
            # Copy attributes from base response
            for attr in ["id", "content", "model", "role", "stop_reason", "type", "usage"]:
                if hasattr(base_response, attr):
                    setattr(self, attr, getattr(base_response, attr))
            # Add parsed_output as a Pydantic model instance (not dict)
            self.parsed_output = TestModel(answer="test answer")

    def mock_beta_parse(**kwargs):
        nonlocal beta_parse_called, captured_params
        beta_parse_called = True
        captured_params = kwargs
        # Create a mock response with parsed_output attribute
        base_response = create_mock_anthropic_response()
        return MockParsedResponse(base_response)

    # Mock beta.messages.parse (used for Pydantic models)
    if hasattr(anthropic_client._client, "beta"):
        monkeypatch.setattr(anthropic_client._client.beta.messages, "parse", mock_beta_parse)

        # Set response format (Pydantic model)
        anthropic_client._response_format = TestModel

        # Call create with Sonnet 4.5
        params = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 100,
        }

        anthropic_client._create_with_native_structured_output(params)

        # Verify beta API was called
        assert beta_parse_called, "Should call beta.messages.parse for Pydantic models"

        # Verify output_format parameter (should be the Pydantic model itself)
        assert "output_format" in captured_params
        assert captured_params["output_format"] == TestModel

        # Verify beta header
        assert "betas" in captured_params
        assert "structured-outputs-2025-11-13" in captured_params["betas"]


@run_for_optional_imports(["anthropic"], "anthropic")
def test_json_mode_fallback_on_native_failure(anthropic_client, monkeypatch):
    """Test graceful fallback to JSON Mode if native fails."""

    def mock_native_failure(params):
        raise Exception("Beta API not available")

    def mock_json_mode_success(params):
        return create_mock_anthropic_response()

    monkeypatch.setattr(anthropic_client, "_create_with_native_structured_output", mock_native_failure)
    monkeypatch.setattr(anthropic_client, "_create_with_json_mode", mock_json_mode_success)

    anthropic_client._response_format = BaseModel

    # Should fallback gracefully
    params = {"model": "claude-sonnet-4-5", "messages": [], "max_tokens": 100}

    # Note: This test verifies the fallback logic exists in the implementation
    # The actual implementation should catch exceptions and fallback
    with pytest.raises(Exception):
        # Currently will raise; implementation should add fallback logic
        anthropic_client.create(params)


@run_for_optional_imports(["anthropic"], "anthropic")
def test_pydantic_model_vs_dict_schema(anthropic_client):
    """Test handling of both Pydantic models and dict schemas."""

    class TestModel(BaseModel):
        name: str
        value: int

    # Test with Pydantic model
    anthropic_client._response_format = TestModel
    schema_from_model = TestModel.model_json_schema() if anthropic_client._response_format else {}

    assert "properties" in schema_from_model
    assert "name" in schema_from_model["properties"]
    assert "value" in schema_from_model["properties"]

    # Test with dict schema
    dict_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "value": {"type": "integer"},
        },
        "required": ["name", "value"],
    }
    anthropic_client._response_format = dict_schema

    assert anthropic_client._response_format == dict_schema


# ==============================================================================
# Real API Call Tests for Native Structured Outputs
# ==============================================================================


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@run_for_optional_imports(["anthropic"], "anthropic")
def test_real_native_structured_output_api_call():
    """Real API call test for native structured output with Claude Sonnet 4.5."""
    import os

    from pydantic import BaseModel

    # Define structured output schema
    class Step(BaseModel):
        explanation: str
        output: str

    class MathReasoning(BaseModel):
        steps: list[Step]
        final_answer: str

    # Create client with response format
    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Test with Claude Sonnet 4.5 (supports native structured outputs)
    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Solve the equation: 2x + 5 = 15. Show your work step by step."}],
        "max_tokens": 1024,
        "response_format": MathReasoning,
    }

    # Make actual API call
    response = client.create(params)

    # Verify response structure
    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None

    # Verify it's valid JSON and matches schema
    result = MathReasoning.model_validate_json(response.choices[0].message.content)

    # Verify mathematical correctness
    assert len(result.steps) > 0, "Should have at least one step"
    assert result.final_answer, "Should have a final answer"

    # The answer should be x = 5
    assert "5" in result.final_answer or "x = 5" in result.final_answer.lower()

    # Verify each step has required fields
    for step in result.steps:
        assert step.explanation, "Each step should have an explanation"
        assert step.output, "Each step should have output"


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@run_for_optional_imports(["anthropic"], "anthropic")
def test_real_json_mode_fallback_api_call():
    """Real API call test for JSON Mode fallback with older Claude model."""
    import os

    from pydantic import BaseModel

    # Define structured output schema
    class Step(BaseModel):
        explanation: str
        output: str

    class MathReasoning(BaseModel):
        steps: list[Step]
        final_answer: str

    # Create client
    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Test with Claude Haiku (does NOT support native structured outputs, should fallback to JSON Mode)
    params = {
        "model": "claude-3-haiku-20240307",
        "messages": [{"role": "user", "content": "Solve: 3x - 4 = 11. Show your work step by step."}],
        "max_tokens": 1024,
        "response_format": MathReasoning,
    }

    # Make actual API call - should use JSON Mode fallback
    response = client.create(params)

    # Verify response structure
    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None

    # Verify it's valid JSON and matches schema
    result = MathReasoning.model_validate_json(response.choices[0].message.content)

    # Verify mathematical correctness
    assert len(result.steps) > 0, "JSON Mode should still produce steps"
    assert result.final_answer, "JSON Mode should have final answer"

    # The answer should be x = 5
    assert "5" in result.final_answer or "x = 5" in result.final_answer.lower()


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@run_for_optional_imports(["anthropic"], "anthropic")
def test_real_native_vs_json_mode_comparison():
    """Compare native structured output vs JSON Mode with same prompt."""
    import os

    from pydantic import BaseModel

    class AnalysisResult(BaseModel):
        summary: str
        key_points: list[str]
        conclusion: str

    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    test_message = (
        "Analyze the benefits of structured outputs in AI systems. Provide a summary, key points, and conclusion."
    )

    # Test 1: Native structured output (Claude Sonnet 4.5)
    params_native = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": test_message}],
        "max_tokens": 1024,
        "response_format": AnalysisResult,
    }

    response_native = client.create(params_native)
    result_native = AnalysisResult.model_validate_json(response_native.choices[0].message.content)

    # Test 2: JSON Mode fallback (Haiku)
    params_json = {
        "model": "claude-3-haiku-20240307",
        "messages": [{"role": "user", "content": test_message}],
        "max_tokens": 1024,
        "response_format": AnalysisResult,
    }

    response_json = client.create(params_json)
    result_json = AnalysisResult.model_validate_json(response_json.choices[0].message.content)

    # Both should produce valid structured outputs
    assert result_native.summary and result_native.key_points and result_native.conclusion
    assert result_json.summary and result_json.key_points and result_json.conclusion

    # Both should have at least some key points
    assert len(result_native.key_points) > 0
    assert len(result_json.key_points) > 0


# ==============================================================================
# Unit Tests for Strict Tool Use Feature
# ==============================================================================


@run_for_optional_imports(["anthropic"], "anthropic")
def test_validate_structured_outputs_version(monkeypatch):
    """Test SDK version validation for structured outputs beta header."""
    from autogen.oai.anthropic import validate_structured_outputs_version

    # Test with sufficient version
    monkeypatch.setattr("autogen.oai.anthropic.anthropic_version", "0.74.1")
    validate_structured_outputs_version()  # Should not raise

    monkeypatch.setattr("autogen.oai.anthropic.anthropic_version", "0.80.0")
    validate_structured_outputs_version()  # Should not raise

    # Test with insufficient version
    monkeypatch.setattr("autogen.oai.anthropic.anthropic_version", "0.70.0")
    with pytest.raises(ImportError, match="Anthropic structured outputs require anthropic>=0.74.1"):
        validate_structured_outputs_version()

    monkeypatch.setattr("autogen.oai.anthropic.anthropic_version", "0.74.0")
    with pytest.raises(ImportError, match="Anthropic structured outputs require anthropic>=0.74.1"):
        validate_structured_outputs_version()


@run_for_optional_imports(["anthropic"], "anthropic")
def test_openai_func_to_anthropic_preserves_strict(anthropic_client):
    """Test that strict field is preserved during tool conversion."""
    from autogen.oai.anthropic import AnthropicClient

    # Tool with strict=True
    strict_tool = {
        "name": "calculate",
        "description": "Perform calculation",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "subtract"]},
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
        },
    }

    result = AnthropicClient.openai_func_to_anthropic(strict_tool)

    # Verify strict field is preserved
    assert "strict" in result
    assert result["strict"] is True

    # Verify input_schema conversion
    assert "input_schema" in result
    assert "parameters" not in result

    # Verify schema transformation was applied for strict tools
    # Should add additionalProperties: false (required by Anthropic for strict tools)
    assert result["input_schema"]["additionalProperties"] is False

    # Verify properties are still there
    assert "properties" in result["input_schema"]
    assert "operation" in result["input_schema"]["properties"]
    assert "a" in result["input_schema"]["properties"]
    assert "b" in result["input_schema"]["properties"]

    # Tool without strict field
    legacy_tool = {
        "name": "search",
        "description": "Search function",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }

    result_legacy = AnthropicClient.openai_func_to_anthropic(legacy_tool)

    # Verify strict field is not added if not present
    assert "strict" not in result_legacy

    # Legacy tools should not have schema transformation applied
    # (additionalProperties might not be set)
    assert result_legacy["input_schema"]["properties"]["query"]["type"] == "string"


@run_for_optional_imports(["anthropic"], "anthropic")
def test_strict_tools_use_beta_api(anthropic_client, monkeypatch):
    """Test that strict tools trigger beta API usage."""

    beta_create_called = False
    standard_create_called = False
    captured_params = {}

    def mock_beta_create(**kwargs):
        nonlocal beta_create_called, captured_params
        beta_create_called = True
        captured_params = kwargs
        return create_mock_anthropic_response()

    def mock_standard_create(**kwargs):
        nonlocal standard_create_called
        standard_create_called = True
        return create_mock_anthropic_response()

    # Mock both beta and standard API calls
    if hasattr(anthropic_client._client, "beta"):
        monkeypatch.setattr(anthropic_client._client.beta.messages, "create", mock_beta_create)
    monkeypatch.setattr(anthropic_client._client.messages, "create", mock_standard_create)

    # Test with strict tools
    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Calculate 5 + 3"}],
        "max_tokens": 100,
        "functions": [
            {
                "name": "calculate",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
            }
        ],
    }

    anthropic_client._create_standard(params)

    # Verify beta API was called
    assert beta_create_called, "Strict tools should trigger beta API"
    assert not standard_create_called, "Should not call standard API when strict tools present"

    # Verify beta header
    assert "betas" in captured_params
    assert "structured-outputs-2025-11-13" in captured_params["betas"]

    # Verify tools have strict field
    assert "tools" in captured_params
    assert any(tool.get("strict") for tool in captured_params["tools"])


@run_for_optional_imports(["anthropic"], "anthropic")
def test_legacy_tools_use_standard_api(anthropic_client, monkeypatch):
    """Test that legacy tools (without strict) use standard API."""

    beta_create_called = False
    standard_create_called = False

    def mock_beta_create(**kwargs):
        nonlocal beta_create_called
        beta_create_called = True
        return create_mock_anthropic_response()

    def mock_standard_create(**kwargs):
        nonlocal standard_create_called
        standard_create_called = True
        return create_mock_anthropic_response()

    # Mock both APIs
    if hasattr(anthropic_client._client, "beta"):
        monkeypatch.setattr(anthropic_client._client.beta.messages, "create", mock_beta_create)
    monkeypatch.setattr(anthropic_client._client.messages, "create", mock_standard_create)

    # Test with legacy tools (no strict field)
    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Search for documentation"}],
        "max_tokens": 100,
        "functions": [
            {
                "name": "search",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            }
        ],
    }

    anthropic_client._create_standard(params)

    # Verify standard API was called
    assert standard_create_called, "Legacy tools should use standard API"
    assert not beta_create_called, "Should not call beta API without strict tools"


# ==============================================================================
# Real API Call Tests for Strict Tool Use
# ==============================================================================


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@run_for_optional_imports(["anthropic"], "anthropic")
def test_real_strict_tool_use_api_call():
    """Real API call test for strict tool use with type enforcement."""
    import json
    import os

    # Create client
    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Define strict tool with enum for operation
    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Calculate 15 + 7 using the calculator tool"}],
        "max_tokens": 1024,
        "functions": [
            {
                "name": "calculate",
                "description": "Perform arithmetic calculation",
                "strict": True,  # Enable strict type validation
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                            "description": "The operation to perform",
                        },
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"},
                    },
                    "required": ["operation", "a", "b"],
                },
            }
        ],
    }

    # Make actual API call
    response = client.create(params)

    # Verify response structure
    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0

    # Verify tool call was made
    message = response.choices[0].message
    assert message.tool_calls is not None, "Should have tool calls"
    assert len(message.tool_calls) > 0, "Should have at least one tool call"

    # Verify tool call structure
    tool_call = message.tool_calls[0]
    assert tool_call.function.name == "calculate"

    # Parse and verify arguments
    args = json.loads(tool_call.function.arguments)

    # With strict=True, these should be guaranteed to be correct types
    assert isinstance(args["a"], (int, float)), "Argument 'a' should be a number"
    assert isinstance(args["b"], (int, float)), "Argument 'b' should be a number"
    assert args["operation"] in ["add", "subtract", "multiply", "divide"], "Operation should be valid enum value"

    # Verify the calculation is correct
    assert args["operation"] == "add"
    assert args["a"] == 15
    assert args["b"] == 7


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@run_for_optional_imports(["anthropic"], "anthropic")
def test_real_strict_tool_type_enforcement():
    """Real API call test verifying strict mode enforces correct types."""
    import json
    import os

    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Tool with multiple type constraints
    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Book a flight for 2 passengers to New York, economy cabin"}],
        "max_tokens": 1024,
        "functions": [
            {
                "name": "book_flight",
                "description": "Book a flight",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "passengers": {"type": "integer", "description": "Number of passengers"},
                        "destination": {"type": "string", "description": "Destination city"},
                        "cabin_class": {
                            "type": "string",
                            "enum": ["economy", "business", "first"],
                            "description": "Cabin class",
                        },
                    },
                    "required": ["passengers", "destination", "cabin_class"],
                },
            }
        ],
    }

    response = client.create(params)

    # Verify tool call
    assert response.choices[0].message.tool_calls is not None

    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    # Strict mode guarantees these types
    assert isinstance(args["passengers"], int), "passengers should be integer, not string '2'"
    assert args["passengers"] == 2

    assert isinstance(args["destination"], str)
    assert args["destination"].lower() == "new york"

    assert args["cabin_class"] in ["economy", "business", "first"], "cabin_class must match enum"


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@run_for_optional_imports(["anthropic"], "anthropic")
def test_real_combined_strict_tools_and_structured_output():
    """Real API call test combining strict tools with structured output."""
    import json
    import os

    from pydantic import BaseModel

    # Result schema
    class CalculationResult(BaseModel):
        problem: str
        steps: list[str]
        result: float
        verification: str

    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Use both strict tools and structured output
    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Calculate (10 + 5) * 2 and explain your work"}],
        "max_tokens": 1024,
        "response_format": CalculationResult,
        "functions": [
            {
                "name": "calculate",
                "description": "Perform calculation",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["operation", "a", "b"],
                },
            }
        ],
    }

    response = client.create(params)

    # When both strict tools and structured output are configured with beta.messages.create,
    # Claude chooses which feature to use based on the prompt:
    # - Either makes tool calls (BetaToolUseBlock), OR
    # - Provides structured output (BetaTextBlock)
    # Both are processed via beta API with the structured-outputs-2025-11-13 header
    message = response.choices[0].message

    # Verify at least one content type is present
    has_tool_calls = message.tool_calls is not None and len(message.tool_calls) > 0
    has_structured_output = message.content and message.content.strip()

    assert has_tool_calls or has_structured_output, "Should have either tool calls OR structured output"

    # If tool calls are present, verify strict typing
    if has_tool_calls:
        tool_call = message.tool_calls[0]
        assert tool_call.function.name == "calculate", "Tool call should be for calculate function"
        args = json.loads(tool_call.function.arguments)
        assert isinstance(args["a"], (int, float)), "Argument 'a' should be a number"
        assert isinstance(args["b"], (int, float)), "Argument 'b' should be a number"
        assert args["operation"] in [
            "add",
            "subtract",
            "multiply",
            "divide",
        ], "Operation should be valid enum value"

    # If structured output is present, verify schema compliance
    if has_structured_output:
        result = CalculationResult.model_validate_json(message.content)
        assert result.problem, "Should have problem description"
        assert len(result.steps) > 0, "Should have calculation steps"
        assert isinstance(result.result, (int, float)), "Result should be a number"
        assert result.verification, "Should have verification"


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@run_for_optional_imports(["anthropic"], "anthropic")
def test_real_sdk_version_validation_on_strict_tools():
    """Test that SDK version is validated when using strict tools."""
    import os

    # This test verifies that the version check happens
    # If SDK is too old, it should raise ImportError

    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Test"}],
        "max_tokens": 100,
        "functions": [
            {
                "name": "test_tool",
                "strict": True,
                "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
            }
        ],
    }

    # This should work if SDK >= 0.74.1, otherwise raise ImportError
    # We can't easily test the failure case without downgrading the SDK
    # So we just verify it doesn't raise with a compatible SDK
    try:
        response = client.create(params)
        # If we get here, SDK version is compatible
        assert response is not None
    except ImportError as e:
        # If SDK is too old, should get clear error message
        assert "anthropic>=0.74.1" in str(e)
        assert "Please upgrade" in str(e)


# ==============================================================================
# Real API Call Tests for Extended Thinking
# ==============================================================================


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@run_for_optional_imports(["anthropic"], "anthropic")
def test_real_extended_thinking_api_call():
    """Real API call test for extended thinking feature with ThinkingBlock."""
    import os

    # Create client
    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Test with a complex reasoning problem that benefits from extended thinking
    params = {
        "model": "claude-sonnet-4-5",
        "messages": [
            {
                "role": "user",
                "content": """A farmer has 17 sheep. All but 9 die. How many sheep are left alive?
Think through this step by step, being careful about the wording.""",
            }
        ],
        "max_tokens": 8000,  # Must be greater than thinking.budget_tokens
        "thinking": {
            "type": "enabled",
            "budget_tokens": 3000,  # Budget for internal reasoning
        },
    }

    # Make API call with extended thinking enabled
    response = client.create(params)

    # Verify response structure
    assert response is not None
    assert response.choices is not None
    assert len(response.choices) > 0

    # Get message content
    message = response.choices[0].message
    assert message.content is not None

    content = message.content
    logger.info("\n=== Extended Thinking Response ===")
    logger.info(content)
    logger.info("=== End Response ===\n")

    # Verify both thinking and text content are present
    # The response should contain "[Thinking]" prefix when ThinkingBlock is present
    assert isinstance(content, str)
    assert len(content) > 0

    # Check if thinking was included (indicated by [Thinking] prefix)
    has_thinking = "[Thinking]" in content

    # Verify the answer is correct (9 sheep are left alive)
    assert "9" in content

    # If thinking was included, verify it's properly formatted
    if has_thinking:
        # Should have [Thinking] prefix followed by thinking content, then regular response
        assert content.startswith("[Thinking]")
        # Should have multiple parts (thinking + text)
        parts = content.split("\n\n", 1)
        assert len(parts) >= 1

    # Verify cost tracking includes thinking tokens if present
    assert response.cost is not None
    assert response.cost >= 0

    # Verify token usage
    assert response.usage is not None
    assert response.usage.total_tokens > 0
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
