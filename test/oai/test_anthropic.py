# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

from unittest.mock import patch

import pytest

from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.llm_config import LLMConfig
from autogen.oai.anthropic import AnthropicClient, AnthropicLLMConfigEntry, _calculate_cost

with optional_import_block() as result:
    from anthropic.types import Message, TextBlock


from typing import Literal

from pydantic import BaseModel


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


@run_for_optional_imports(["anthropic"], "anthropic")
def test_convert_response_format_to_output_format_with_pydantic(anthropic_client):
    """Test converting Pydantic model to output_format schema."""

    class ContactInfo(BaseModel):
        name: str
        email: str
        age: int
        interests: list[str]

    result = anthropic_client._convert_response_format_to_output_format(ContactInfo)

    assert result["type"] == "json_schema"
    assert "schema" in result
    schema = result["schema"]
    assert schema["type"] == "object"
    assert "properties" in schema
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["email"]["type"] == "string"
    assert schema["properties"]["age"]["type"] == "integer"
    assert schema["properties"]["interests"]["type"] == "array"
    assert "name" in schema.get("required", [])
    assert schema.get("additionalProperties") is False


@run_for_optional_imports(["anthropic"], "anthropic")
def test_convert_response_format_to_output_format_with_dict(anthropic_client):
    """Test converting dict schema to output_format schema."""
    dict_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
        "required": ["name", "email"],
    }

    result = anthropic_client._convert_response_format_to_output_format(dict_schema)

    assert result["type"] == "json_schema"
    assert result["schema"] == {
        "type": "object",
        "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
        "required": ["name", "email"],
        "additionalProperties": False,
    }


@run_for_optional_imports(["anthropic"], "anthropic")
def test_parse_structured_output_response_with_pydantic(anthropic_client):
    """Test parsing structured output response with Pydantic model."""

    class ContactInfo(BaseModel):
        name: str
        email: str
        age: int

    anthropic_client._response_format = ContactInfo

    json_text = '{"name": "John Doe", "email": "john@example.com", "age": 30}'
    result = anthropic_client._parse_structured_output_response(json_text)

    assert isinstance(result, ContactInfo)
    assert result.name == "John Doe"
    assert result.email == "john@example.com"
    assert result.age == 30


@run_for_optional_imports(["anthropic"], "anthropic")
def test_parse_structured_output_response_with_dict(anthropic_client):
    """Test parsing structured output response with dict schema."""
    anthropic_client._response_format = {"type": "object"}

    json_text = '{"name": "John Doe", "email": "john@example.com"}'
    result = anthropic_client._parse_structured_output_response(json_text)

    assert result == json_text


@run_for_optional_imports(["anthropic"], "anthropic")
def test_parse_structured_output_response_without_format(anthropic_client):
    """Test parsing structured output response without response_format."""
    anthropic_client._response_format = None

    json_text = '{"name": "John Doe", "email": "john@example.com"}'
    result = anthropic_client._parse_structured_output_response(json_text)

    assert result == json_text


@run_for_optional_imports(["anthropic"], "anthropic")
def test_parse_structured_output_response_invalid_json(anthropic_client):
    """Test parsing structured output response with invalid JSON."""

    class ContactInfo(BaseModel):
        name: str
        email: str

    anthropic_client._response_format = ContactInfo

    invalid_json = '{"name": "John Doe", "email": invalid}'

    with pytest.raises(ValueError, match="Failed to parse structured output response"):
        anthropic_client._parse_structured_output_response(invalid_json)


@run_for_optional_imports(["anthropic"], "anthropic")
def test_load_config_with_output_format(anthropic_client):
    """Test load_config with output_format parameter."""
    output_format = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    }

    params = {
        "model": "claude-sonnet-4-5",
        "output_format": output_format,
        "temperature": 0.7,
        "max_tokens": 100,
    }

    result = anthropic_client.load_config(params)

    assert result["model"] == "claude-sonnet-4-5"
    assert result["output_format"] == output_format
    assert result["temperature"] == 0.7
    assert result["max_tokens"] == 100


@run_for_optional_imports(["anthropic"], "anthropic")
def test_create_with_output_format_mocked(anthropic_client):
    """Test create method with output_format using mocked beta API."""
    from autogen.oai.oai_models import ChatCompletion

    # Mock the beta API response
    mock_response = Message(
        id="msg_123",
        content=[
            TextBlock(
                text='{"name": "John Doe", "email": "john@example.com", "age": 30, "interests": ["Python"]}',
                type="text",
            )
        ],
        model="claude-sonnet-4-5",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage={"input_tokens": 10, "output_tokens": 25},
    )

    output_format = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "age": {"type": "integer"},
                "interests": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "email", "age", "interests"],
            "additionalProperties": False,
        },
    }

    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Extract contact info"}],
        "output_format": output_format,
        "max_tokens": 100,
    }

    # Mock beta.messages.create
    with patch.object(anthropic_client._client.beta.messages, "create", return_value=mock_response) as mock_beta_create:
        result = anthropic_client.create(params)

        # Verify beta API was called
        mock_beta_create.assert_called_once()
        call_kwargs = mock_beta_create.call_args[1]
        assert "betas" in call_kwargs
        assert "structured-outputs-2025-11-13" in call_kwargs["betas"]

        # Verify output_format was passed
        assert call_kwargs["output_format"] == output_format

    # Verify response structure
    assert isinstance(result, ChatCompletion)
    assert (
        result.choices[0].message.content
        == '{"name": "John Doe", "email": "john@example.com", "age": 30, "interests": ["Python"]}'
    )


@run_for_optional_imports(["anthropic"], "anthropic")
def test_create_with_response_format_conversion_mocked(anthropic_client):
    """Test create method converts response_format to output_format."""
    from autogen.oai.oai_models import ChatCompletion

    class ContactInfo(BaseModel):
        name: str
        email: str
        age: int

    # Mock the beta API response
    mock_response = Message(
        id="msg_123",
        content=[
            TextBlock(
                text='{"name": "John Doe", "email": "john@example.com", "age": 30}',
                type="text",
            )
        ],
        model="claude-sonnet-4-5",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage={"input_tokens": 10, "output_tokens": 25},
    )

    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Extract contact info"}],
        "response_format": ContactInfo,  # Pydantic model
        "max_tokens": 100,
    }

    # Mock beta.messages.create
    with patch.object(anthropic_client._client.beta.messages, "create", return_value=mock_response) as mock_beta_create:
        result = anthropic_client.create(params)

        # Verify beta API was called
        mock_beta_create.assert_called_once()
        call_kwargs = mock_beta_create.call_args[1]

        # Verify output_format was created from response_format
        assert "output_format" in call_kwargs
        assert call_kwargs["output_format"]["type"] == "json_schema"
        assert "schema" in call_kwargs["output_format"]

        # Verify betas header was added
        assert "betas" in call_kwargs
        assert "structured-outputs-2025-11-13" in call_kwargs["betas"]

    # Verify response_format was stored for validation
    assert anthropic_client._response_format == ContactInfo

    # Verify response structure
    assert isinstance(result, ChatCompletion)
    assert "John Doe" in result.choices[0].message.content
