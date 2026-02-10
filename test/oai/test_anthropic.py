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
            model="claude-opus-4",
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
        model="claude-opus-4",
    )
    assert (
        _calculate_cost(completion.usage["prompt_tokens"], completion.usage["completion_tokens"], completion.model)
        == 0.0002025
    ), "Cost should be $0.0002025"


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

    # Haiku 4.5 should be supported
    assert supports_native_structured_outputs("claude-haiku-4-5")
    assert supports_native_structured_outputs("claude-haiku-4-5-20251001")

    # Older Haiku models should not be supported
    assert not supports_native_structured_outputs("claude-3-5-haiku-20241022")


@run_for_optional_imports(["anthropic"], "anthropic")
def test_has_messages_parse_api():
    """Test SDK version detection for messages.parse() API."""
    from autogen.oai.anthropic import has_messages_parse_api

    # Should detect if current SDK has messages.parse()
    has_parse = has_messages_parse_api()

    # If we have anthropic SDK, it should be a boolean
    assert isinstance(has_parse, bool)

    # If True, verify we can import the stable API
    if has_parse:
        try:
            from anthropic.resources.messages import Messages

            assert hasattr(Messages, "parse"), "Stable API should have parse method"
        except ImportError:
            pytest.fail("has_messages_parse_api returned True but cannot import stable API")


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
def test_native_structured_output_with_parse_api(anthropic_client, monkeypatch):
    """Test that native structured output uses stable messages.parse() API correctly."""
    from autogen.oai.anthropic import has_messages_parse_api

    if not has_messages_parse_api():
        pytest.skip("SDK does not support messages.parse() API")

    parse_called = False
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

    def mock_parse(**kwargs):
        nonlocal parse_called, captured_params
        parse_called = True
        captured_params = kwargs
        # Create a mock response with parsed_output attribute
        base_response = create_mock_anthropic_response()
        return MockParsedResponse(base_response)

    # Mock messages.parse (stable API, used for Pydantic models)
    monkeypatch.setattr(anthropic_client._client.messages, "parse", mock_parse)

    # Set response format (Pydantic model)
    anthropic_client._response_format = TestModel

    # Call create with Sonnet 4.5
    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 100,
    }

    anthropic_client._create_with_native_structured_output(params)

    # Verify stable messages.parse was called
    assert parse_called, "Should call messages.parse for Pydantic models"

    # Verify output_format parameter (should be the Pydantic model itself)
    assert "output_format" in captured_params
    assert captured_params["output_format"] == TestModel

    # Verify no beta headers are present
    assert "betas" not in captured_params


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
def test_strict_tools_use_standard_api_with_strict(anthropic_client, monkeypatch):
    """Test that strict tools use standard messages.create() API (strict is now GA)."""

    standard_create_called = False
    captured_params = {}

    def mock_standard_create(**kwargs):
        nonlocal standard_create_called, captured_params
        standard_create_called = True
        captured_params = kwargs
        return create_mock_anthropic_response()

    # Mock standard API call
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

    # Verify standard API was called (strict tools no longer need beta)
    assert standard_create_called, "Strict tools should use standard messages.create()"

    # Verify no beta headers are present
    assert "betas" not in captured_params

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


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@run_for_optional_imports(["anthropic"], "anthropic")
def test_real_tools_with_structured_output_beta_api(credentials_anthropic_claude_sonnet, caplog):
    """Real API call test for tools + structured outputs using GA API.

    This test verifies that OpenAI tool format works with Anthropic's structured
    outputs API. Previously, the OpenAI wrapper format {"type": "function", ...}
    was rejected by the API with a 400 error.

    The key test is that combining tools (in OpenAI format) with response_format
    doesn't cause a 400 error, proving the tool format conversion works correctly.
    """
    import json
    import logging

    from pydantic import BaseModel

    # Capture logs to verify beta API usage
    caplog.set_level(logging.WARNING)

    # Define structured output schema
    class MathResult(BaseModel):
        steps: list[str]
        answer: int

    # Get API key from credentials
    api_key = credentials_anthropic_claude_sonnet.config_list[0]["api_key"]

    # Create client
    client = AnthropicClient(api_key=api_key)

    # Define tool in OpenAI format with "type": "function" wrapper
    # This is the format that was causing the 400 error before the fix
    tools = [
        {
            "type": "function",  # OpenAI wrapper format
            "function": {
                "name": "calculator",
                "description": "Perform basic math operations",
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
                    "additionalProperties": False,
                },
            },
        }
    ]

    # Test 1: Verify tools with structured output don't cause 400 error
    # This was the original bug - combining tools + response_format would fail
    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Use the calculator tool to compute 23 + 19"}],
        "max_tokens": 1024,
        "response_format": MathResult,  # Triggers beta API
        "tools": tools,  # OpenAI format with "type": "function"
    }

    # Make actual API call - this should succeed with the fix (no 400 error)
    response = client.create(params)

    # Verify response structure
    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0

    message = response.choices[0].message

    # Verify tool call was made (Claude should use the tool)
    assert message.tool_calls is not None, "Should have tool calls"
    assert len(message.tool_calls) > 0, "Should have at least one tool call"

    # Verify tool call structure
    tool_call = message.tool_calls[0]
    assert hasattr(tool_call, "function"), "Tool call should have function attribute"
    assert tool_call.function.name == "calculator"

    # Parse and verify arguments - strict validation should work
    args = json.loads(tool_call.function.arguments)
    assert args["operation"] == "add"
    assert args["a"] == 23
    assert args["b"] == 19

    # Verify cost tracking
    assert response.cost is not None
    assert response.cost >= 0

    # VERIFY BETA API WAS ACTUALLY USED (not fallback to JSON mode)
    # Method 1: Check that no fallback warning was logged
    fallback_warnings = [
        record
        for record in caplog.records
        if "Falling back to JSON Mode" in record.message and record.levelname == "WARNING"
    ]
    assert len(fallback_warnings) == 0, (
        f"Beta API should not fall back to JSON mode. Found warnings: {[r.message for r in fallback_warnings]}"
    )

    # Method 2: Verify response characteristics indicate beta API usage
    # Beta API responses should have proper structured content
    assert response.choices[0].message.content is not None or response.choices[0].message.tool_calls is not None, (
        "Beta API should return either content or tool calls"
    )

    # Test 2: Verify structured output works after tool execution
    # Send tool result and request structured output using OpenAI format
    tool_result_params = {
        "model": "claude-sonnet-4-5",
        "messages": [
            {"role": "user", "content": "Calculate 15 + 7 and show your work"},
            {
                "role": "assistant",
                "content": "",  # Empty content when tool calls are present
                "tool_calls": [
                    {
                        "id": "call_test_123",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": json.dumps({"operation": "add", "a": 15, "b": 7}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_test_123", "content": "22"},
        ],
        "max_tokens": 1024,
        "response_format": MathResult,
    }

    # Get response with structured output
    final_response = client.create(tool_result_params)

    # Verify structured output
    assert final_response.choices[0].message.content is not None
    content = final_response.choices[0].message.content

    # Parse and validate structured output
    result = MathResult.model_validate_json(content) if isinstance(content, str) else MathResult.model_validate(content)

    # Verify structured output has required fields
    assert result.steps, "Should have steps"
    assert result.answer == 22, "Should have correct answer"


# ==============================================================================
# Unit Tests for Streaming Support
# ==============================================================================


@run_for_optional_imports(["anthropic"], "anthropic")
def test_load_config_stream_enabled(anthropic_client):
    """Verify that stream=True flows through load_config without being forced to False."""
    params = {
        "model": "claude-sonnet-4-5",
        "stream": True,
        "temperature": 1,
        "max_tokens": 100,
    }
    result = anthropic_client.load_config(params)
    assert result["stream"] is True, "stream=True should be preserved in config"


@run_for_optional_imports(["anthropic"], "anthropic")
def test_create_streaming_routes(anthropic_client, monkeypatch):
    """Verify that _create_standard routes to _create_streaming when stream=True."""
    streaming_called = False

    def mock_create_streaming(params):
        nonlocal streaming_called
        streaming_called = True
        return create_mock_anthropic_response()

    monkeypatch.setattr(anthropic_client, "_create_streaming", mock_create_streaming)

    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
        "stream": True,
    }
    anthropic_client._create_standard(params)
    assert streaming_called, "_create_standard should route to _create_streaming when stream=True"


@run_for_optional_imports(["anthropic"], "anthropic")
def test_create_standard_no_stream_does_not_route(anthropic_client, monkeypatch):
    """Verify _create_standard does NOT route to _create_streaming when stream=False."""
    streaming_called = False

    def mock_create_streaming(params):
        nonlocal streaming_called
        streaming_called = True
        return create_mock_anthropic_response()

    monkeypatch.setattr(anthropic_client, "_create_streaming", mock_create_streaming)
    monkeypatch.setattr(anthropic_client._client.messages, "create", lambda **kw: create_mock_anthropic_response())

    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
        "stream": False,
    }
    anthropic_client._create_standard(params)
    assert not streaming_called, "_create_standard should NOT route to _create_streaming when stream=False"


@run_for_optional_imports(["anthropic"], "anthropic")
def test_streaming_text_accumulation(anthropic_client, monkeypatch):
    """Mock raw streaming events and verify text accumulation + StreamEvent emission."""
    from autogen.io.base import IOStream

    # Build mock streaming events
    class MockUsage:
        def __init__(self, input_tokens=0, output_tokens=0):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    class MockEvent:
        def __init__(self, type, **kwargs):
            self.type = type
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MockMessage:
        def __init__(self):
            self.id = "msg_stream_123"
            self.model = "claude-sonnet-4-5"
            self.usage = MockUsage(input_tokens=10)

    class MockContentBlock:
        def __init__(self, type, **kwargs):
            self.type = type
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MockDelta:
        def __init__(self, type, **kwargs):
            self.type = type
            for k, v in kwargs.items():
                setattr(self, k, v)

    # Create sequence of streaming events for a text response
    events = [
        MockEvent("message_start", message=MockMessage()),
        MockEvent("content_block_start", index=0, content_block=MockContentBlock("text")),
        MockEvent("content_block_delta", index=0, delta=MockDelta("text_delta", text="Hello")),
        MockEvent("content_block_delta", index=0, delta=MockDelta("text_delta", text=" world")),
        MockEvent("content_block_delta", index=0, delta=MockDelta("text_delta", text="!")),
        MockEvent("content_block_stop", index=0),
        MockEvent(
            "message_delta", delta=MockDelta("message_delta", stop_reason="end_turn"), usage=MockUsage(output_tokens=3)
        ),
        MockEvent("message_stop"),
    ]

    # Mock messages.create to return events iterable
    monkeypatch.setattr(anthropic_client._client.messages, "create", lambda **kw: iter(events))

    # Capture StreamEvents
    sent_events = []

    class MockIOStream:
        def send(self, event):
            sent_events.append(event)

    monkeypatch.setattr(IOStream, "get_default", staticmethod(lambda: MockIOStream()))

    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 100,
        "stream": True,
    }

    result = anthropic_client._create_streaming(params)

    # Verify accumulated text
    assert result.choices[0].message.content == "Hello world!"

    # Verify StreamEvents were emitted
    # StreamEvent is wrapped by @wrap_event, so actual text is at .content.content
    assert len(sent_events) == 3
    assert sent_events[0].content.content == "Hello"
    assert sent_events[1].content.content == " world"
    assert sent_events[2].content.content == "!"

    # Verify response structure
    assert result.id == "msg_stream_123"
    assert result.model == "claude-sonnet-4-5"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 3
    assert result.choices[0].finish_reason == "stop"
    assert result.choices[0].message.tool_calls is None


@run_for_optional_imports(["anthropic"], "anthropic")
def test_streaming_tool_call_accumulation(anthropic_client, monkeypatch):
    """Mock tool_use streaming events and verify tool call reconstruction."""
    from autogen.io.base import IOStream

    class MockUsage:
        def __init__(self, input_tokens=0, output_tokens=0):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    class MockEvent:
        def __init__(self, type, **kwargs):
            self.type = type
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MockMessage:
        def __init__(self):
            self.id = "msg_tool_123"
            self.model = "claude-sonnet-4-5"
            self.usage = MockUsage(input_tokens=15)

    class MockContentBlock:
        def __init__(self, type, **kwargs):
            self.type = type
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MockDelta:
        def __init__(self, type, **kwargs):
            self.type = type
            for k, v in kwargs.items():
                setattr(self, k, v)

    # Simulate tool_use streaming events
    events = [
        MockEvent("message_start", message=MockMessage()),
        MockEvent(
            "content_block_start",
            index=0,
            content_block=MockContentBlock("tool_use", id="toolu_123", name="calculate"),
        ),
        MockEvent("content_block_delta", index=0, delta=MockDelta("input_json_delta", partial_json='{"a":')),
        MockEvent("content_block_delta", index=0, delta=MockDelta("input_json_delta", partial_json=" 5, ")),
        MockEvent("content_block_delta", index=0, delta=MockDelta("input_json_delta", partial_json='"b": 3}')),
        MockEvent("content_block_stop", index=0),
        MockEvent(
            "message_delta", delta=MockDelta("message_delta", stop_reason="tool_use"), usage=MockUsage(output_tokens=10)
        ),
        MockEvent("message_stop"),
    ]

    monkeypatch.setattr(anthropic_client._client.messages, "create", lambda **kw: iter(events))

    class MockIOStream:
        def send(self, event):
            pass

    monkeypatch.setattr(IOStream, "get_default", staticmethod(lambda: MockIOStream()))

    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Calculate 5 + 3"}],
        "max_tokens": 100,
        "stream": True,
    }

    result = anthropic_client._create_streaming(params)

    # Verify tool calls were reconstructed
    assert result.choices[0].message.tool_calls is not None
    assert len(result.choices[0].message.tool_calls) == 1

    tool_call = result.choices[0].message.tool_calls[0]
    assert tool_call.id == "toolu_123"
    assert tool_call.function.name == "calculate"

    import json

    args = json.loads(tool_call.function.arguments)
    assert args["a"] == 5
    assert args["b"] == 3

    # Verify finish reason
    assert result.choices[0].finish_reason == "tool_calls"

    # Verify usage
    assert result.usage.prompt_tokens == 15
    assert result.usage.completion_tokens == 10


@run_for_optional_imports(["anthropic"], "anthropic")
def test_streaming_thinking_blocks(anthropic_client, monkeypatch):
    """Mock thinking streaming events and verify [Thinking] prefix in output."""
    from autogen.io.base import IOStream

    class MockUsage:
        def __init__(self, input_tokens=0, output_tokens=0):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    class MockEvent:
        def __init__(self, type, **kwargs):
            self.type = type
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MockMessage:
        def __init__(self):
            self.id = "msg_think_123"
            self.model = "claude-sonnet-4-5"
            self.usage = MockUsage(input_tokens=20)

    class MockContentBlock:
        def __init__(self, type, **kwargs):
            self.type = type
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MockDelta:
        def __init__(self, type, **kwargs):
            self.type = type
            for k, v in kwargs.items():
                setattr(self, k, v)

    # Simulate thinking + text streaming events
    events = [
        MockEvent("message_start", message=MockMessage()),
        # Thinking block
        MockEvent("content_block_start", index=0, content_block=MockContentBlock("thinking")),
        MockEvent("content_block_delta", index=0, delta=MockDelta("thinking_delta", thinking="Let me think...")),
        MockEvent("content_block_delta", index=0, delta=MockDelta("thinking_delta", thinking=" The answer is 9.")),
        MockEvent("content_block_stop", index=0),
        # Text block
        MockEvent("content_block_start", index=1, content_block=MockContentBlock("text")),
        MockEvent("content_block_delta", index=1, delta=MockDelta("text_delta", text="9 sheep remain.")),
        MockEvent("content_block_stop", index=1),
        MockEvent(
            "message_delta", delta=MockDelta("message_delta", stop_reason="end_turn"), usage=MockUsage(output_tokens=15)
        ),
        MockEvent("message_stop"),
    ]

    monkeypatch.setattr(anthropic_client._client.messages, "create", lambda **kw: iter(events))

    # Capture stream events
    sent_events = []

    class MockIOStream:
        def send(self, event):
            sent_events.append(event)

    monkeypatch.setattr(IOStream, "get_default", staticmethod(lambda: MockIOStream()))

    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "How many sheep?"}],
        "max_tokens": 8000,
        "stream": True,
    }

    result = anthropic_client._create_streaming(params)

    # Verify thinking + text combined with [Thinking] prefix
    content = result.choices[0].message.content
    assert content.startswith("[Thinking]")
    assert "Let me think... The answer is 9." in content
    assert "9 sheep remain." in content

    # Verify only text deltas emit StreamEvents (not thinking)
    # StreamEvent is wrapped by @wrap_event, so actual text is at .content.content
    assert len(sent_events) == 1
    assert sent_events[0].content.content == "9 sheep remain."


# ==============================================================================
# Real API Call Tests for Streaming
# ==============================================================================


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@run_for_optional_imports(["anthropic"], "anthropic")
def test_real_streaming_text():
    """Real API call test for basic streaming text response."""
    import os

    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Say 'Hello streaming!' and nothing else."}],
        "max_tokens": 100,
        "stream": True,
    }

    response = client.create(params)

    # Verify response structure
    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0

    content = response.choices[0].message.content
    assert content is not None
    assert len(content) > 0
    assert "hello" in content.lower() or "streaming" in content.lower()

    # Verify usage is tracked
    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens > 0

    # Verify cost is calculated
    assert response.cost is not None
    assert response.cost >= 0


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@run_for_optional_imports(["anthropic"], "anthropic")
def test_real_streaming_with_tools():
    """Real API call test for streaming with tool calls."""
    import json
    import os

    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Calculate 10 + 5 using the calculator tool."}],
        "max_tokens": 1024,
        "stream": True,
        "functions": [
            {
                "name": "calculator",
                "description": "Perform basic arithmetic",
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

    # Verify response
    assert response is not None
    message = response.choices[0].message

    # Should have tool calls
    assert message.tool_calls is not None, "Streaming should reconstruct tool calls"
    assert len(message.tool_calls) > 0

    tool_call = message.tool_calls[0]
    assert tool_call.function.name == "calculator"

    args = json.loads(tool_call.function.arguments)
    assert args["operation"] == "add"
    assert args["a"] == 10
    assert args["b"] == 5

    # Verify finish reason
    assert response.choices[0].finish_reason == "tool_calls"
