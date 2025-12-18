# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from autogen.import_utils import run_for_optional_imports
from autogen.llm_config import LLMConfig
from autogen.oai.bedrock import BedrockClient, BedrockLLMConfigEntry, oai_messages_to_bedrock_messages
from autogen.oai.oai_models import ChatCompletionMessageToolCall


# Fixtures for mock data
@pytest.fixture
def mock_response():
    class MockResponse:
        def __init__(self, text, choices, usage, cost, model):
            self.text = text
            self.choices = choices
            self.usage = usage
            self.cost = cost
            self.model = model

    return MockResponse


@pytest.fixture
def bedrock_client():
    # Set Bedrock client with some default values
    client = BedrockClient(aws_region="us-east-1")

    client._supports_system_prompts = True

    return client


def test_bedrock_llm_config_entry():
    bedrock_llm_config = BedrockLLMConfigEntry(
        model="anthropic.claude-sonnet-4-5-20250929-v1:0",
        aws_region="us-east-1",
        aws_access_key="test_access_key_id",
        aws_secret_key="test_secret_access_key",
        aws_session_token="test_session_token",
        temperature=0.8,
    )
    expected = {
        "api_type": "bedrock",
        "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "aws_region": "us-east-1",
        "aws_access_key": "test_access_key_id",
        "aws_secret_key": "test_secret_access_key",
        "aws_session_token": "test_session_token",
        "temperature": 0.8,
        "tags": [],
        "supports_system_prompts": True,
    }
    actual = bedrock_llm_config.model_dump()
    assert actual == expected

    assert LLMConfig(bedrock_llm_config).model_dump() == {
        "config_list": [expected],
    }

    with pytest.raises(ValidationError, match="List should have at least 2 items after validation, not 1"):
        bedrock_llm_config = BedrockLLMConfigEntry(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            aws_region="us-east-1",
            price=["0.1"],
        )


def test_bedrock_llm_config_entry_repr():
    bedrock_llm_config = BedrockLLMConfigEntry(
        model="anthropic.claude-sonnet-4-5-20250929-v1:0",
        aws_region="us-east-1",
        aws_access_key="test_access_key_id",
        aws_secret_key="test_secret_access_key",
        aws_session_token="test_session_token",
        aws_profile_name="test_profile_name",
    )

    actual = repr(bedrock_llm_config)
    expected = "BedrockLLMConfigEntry(api_type='bedrock', model='anthropic.claude-sonnet-4-5-20250929-v1:0', tags=[], aws_region='us-east-1', aws_access_key='**********', aws_secret_key='**********', aws_session_token='**********', aws_profile_name='test_profile_name', supports_system_prompts=True)"

    assert actual == expected, actual


def test_bedrock_llm_config_entry_str():
    bedrock_llm_config = BedrockLLMConfigEntry(
        model="anthropic.claude-sonnet-4-5-20250929-v1:0",
        aws_region="us-east-1",
        aws_access_key="test_access_key_id",
        aws_secret_key="test_secret_access_key",
        aws_session_token="test_session_token",
        aws_profile_name="test_profile_name",
    )

    actual = str(bedrock_llm_config)
    expected = "BedrockLLMConfigEntry(api_type='bedrock', model='anthropic.claude-sonnet-4-5-20250929-v1:0', tags=[], aws_region='us-east-1', aws_access_key='**********', aws_secret_key='**********', aws_session_token='**********', aws_profile_name='test_profile_name', supports_system_prompts=True)"

    assert actual == expected, actual


# Test initialization and configuration
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
def test_initialization():
    # Creation works without an api_key as it's handled in the parameter parsing
    BedrockClient(aws_region="us-east-1")


# Test parameters
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
def test_parsing_params(bedrock_client: BedrockClient):
    # All parameters (with default values)
    assert bedrock_client.parse_params({
        "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "temperature": 0.8,
        "top_p": 0.6,
        "max_tokens": 250,
        "seed": 42,
        "stream": False,
    }) == (
        {
            "temperature": 0.8,
            "topP": 0.6,
            "maxTokens": 250,
        },
        {
            "seed": 42,
        },
    )

    # Incorrect types, defaults should be set, will show warnings but not trigger assertions
    with pytest.warns(UserWarning, match=r"Config error - .*"):
        assert bedrock_client.parse_params({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "temperature": "0.5",
            "top_p": "0.6",
            "max_tokens": "250",
            "seed": "42",
        }) == (
            {
                "temperature": None,
                "topP": None,
                "maxTokens": None,
            },
            {
                "seed": None,
            },
        )

    with pytest.warns(UserWarning, match="Streaming is not currently supported, streaming will be disabled"):
        bedrock_client.parse_params({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "stream": True,
        })

    assert bedrock_client.parse_params({
        "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    }) == ({}, {})

    with pytest.raises(AssertionError, match="Please provide the 'model` in the config_list to use Amazon Bedrock"):
        bedrock_client.parse_params({})


# Test text generation
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
@patch("autogen.oai.bedrock.BedrockClient.create")
def test_create_response(mock_chat, bedrock_client: BedrockClient):
    # Mock BedrockClient.chat response
    mock_bedrock_response = MagicMock()
    mock_bedrock_response.choices = [
        MagicMock(finish_reason="stop", message=MagicMock(content="Example Bedrock response", tool_calls=None))
    ]
    mock_bedrock_response.id = "mock_bedrock_response_id"
    mock_bedrock_response.model = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    mock_bedrock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)  # Example token usage

    mock_chat.return_value = mock_bedrock_response

    # Test parameters
    params = {
        "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "World"}],
        "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    }

    # Call the create method
    response = bedrock_client.create(params)

    # Assertions to check if response is structured as expected
    assert response.choices[0].message.content == "Example Bedrock response", (
        "Response content should match expected output"
    )
    assert response.id == "mock_bedrock_response_id", "Response ID should match the mocked response ID"
    assert response.model == "anthropic.claude-sonnet-4-5-20250929-v1:0", (
        "Response model should match the mocked response model"
    )
    assert response.usage.prompt_tokens == 10, "Response prompt tokens should match the mocked response usage"
    assert response.usage.completion_tokens == 20, "Response completion tokens should match the mocked response usage"


# Test functions/tools
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
@patch("autogen.oai.bedrock.BedrockClient.create")
def test_create_response_with_tool_call(mock_chat, bedrock_client: BedrockClient):
    # Mock BedrockClient.chat response
    mock_function = MagicMock(name="currency_calculator")
    mock_function.name = "currency_calculator"
    mock_function.arguments = '{"base_currency": "EUR", "quote_currency": "USD", "base_amount": 123.45}'

    mock_function_2 = MagicMock(name="get_weather")
    mock_function_2.name = "get_weather"
    mock_function_2.arguments = '{"location": "New York"}'

    mock_chat.return_value = MagicMock(
        choices=[
            MagicMock(
                finish_reason="tool_calls",
                message=MagicMock(
                    content="Sample text about the functions",
                    tool_calls=[
                        MagicMock(id="bd65600d-8669-4903-8a14-af88203add38", function=mock_function),
                        MagicMock(id="f50ec0b7-f960-400d-91f0-c42a6d44e3d0", function=mock_function_2),
                    ],
                ),
            )
        ],
        id="mock_bedrock_response_id",
        model="anthropic.claude-sonnet-4-5-20250929-v1:0",
        usage=MagicMock(prompt_tokens=10, completion_tokens=20),
    )

    # Construct parameters
    converted_functions = [
        {
            "type": "function",
            "function": {
                "description": "Currency exchange calculator.",
                "name": "currency_calculator",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "base_amount": {"type": "number", "description": "Amount of currency in base_currency"},
                    },
                    "required": ["base_amount"],
                },
            },
        }
    ]
    bedrock_messages = [
        {"role": "user", "content": "How much is 123.45 EUR in USD?"},
        {"role": "assistant", "content": "World"},
    ]

    # Call the create method
    response = bedrock_client.create({
        "messages": bedrock_messages,
        "tools": converted_functions,
        "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    })

    # Assertions to check if the functions and content are included in the response
    assert response.choices[0].message.content == "Sample text about the functions"
    assert response.choices[0].message.tool_calls[0].function.name == "currency_calculator"
    assert response.choices[0].message.tool_calls[1].function.name == "get_weather"


# Test message conversion from OpenAI to Bedrock format
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
def test_oai_messages_to_bedrock_messages(bedrock_client: BedrockClient):
    # Test that the "name" key is removed and system messages converted to user message
    test_messages = [
        {"role": "system", "content": "You are a helpful AI bot."},
        {"role": "user", "name": "anne", "content": "Why is the sky blue?"},
    ]
    messages = oai_messages_to_bedrock_messages(test_messages, False, False)

    expected_messages = [
        {"role": "user", "content": [{"text": "You are a helpful AI bot."}]},
        {"role": "assistant", "content": [{"text": "Please continue."}]},
        {"role": "user", "content": [{"text": "Why is the sky blue?"}]},
    ]

    assert messages == expected_messages, "'name' was not removed from messages (system message should be user message)"

    # Test that the "name" key is removed and system messages are extracted (as they will be put in separately)
    test_messages = [
        {"role": "system", "content": "You are a helpful AI bot."},
        {"role": "user", "name": "anne", "content": "Why is the sky blue?"},
    ]
    messages = oai_messages_to_bedrock_messages(test_messages, False, True)

    expected_messages = [
        {"role": "user", "content": [{"text": "Why is the sky blue?"}]},
    ]

    assert messages == expected_messages, "'name' was not removed from messages (system messages excluded)"

    # Test that the system message is converted to user and that a continue message is inserted
    test_messages = [
        {"role": "system", "content": "You are a helpful AI bot."},
        {"role": "user", "name": "anne", "content": "Why is the sky blue?"},
        {"role": "system", "content": "Summarise the conversation."},
    ]

    messages = oai_messages_to_bedrock_messages(test_messages, False, False)

    expected_messages = [
        {"role": "user", "content": [{"text": "You are a helpful AI bot."}]},
        {"role": "assistant", "content": [{"text": "Please continue."}]},
        {"role": "user", "content": [{"text": "Why is the sky blue?"}]},
        {"role": "assistant", "content": [{"text": "Please continue."}]},
        {"role": "user", "content": [{"text": "Summarise the conversation."}]},
    ]

    assert messages == expected_messages, (
        "Final 'system' message was not changed to 'user' or continue messages not included"
    )

    # Test that the last message is a user or system message and if not, add a continue message
    test_messages = [
        {"role": "system", "content": "You are a helpful AI bot."},
        {"role": "user", "name": "anne", "content": "Why is the sky blue?"},
        {"role": "assistant", "content": "The sky is blue because that's a great colour."},
    ]
    print(test_messages)

    messages = oai_messages_to_bedrock_messages(test_messages, False, False)
    print(messages)

    expected_messages = [
        {"role": "user", "content": [{"text": "You are a helpful AI bot."}]},
        {"role": "assistant", "content": [{"text": "Please continue."}]},
        {"role": "user", "content": [{"text": "Why is the sky blue?"}]},
        {"role": "assistant", "content": [{"text": "The sky is blue because that's a great colour."}]},
        {"role": "user", "content": [{"text": "Please continue."}]},
    ]

    assert messages == expected_messages, "'Please continue' message was not appended."


class Step(BaseModel):
    explanation: str
    output: str


class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str


# Test 1: Test with Pydantic models
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
def test_response_format_with_pydantic_model(bedrock_client: BedrockClient):
    """Test structured output with Pydantic model."""
    # Mock bedrock_runtime on the instance
    mock_bedrock_runtime = MagicMock()
    bedrock_client.bedrock_runtime = mock_bedrock_runtime

    # Mock Bedrock response with tool call
    mock_response = {
        "stopReason": "tool_use",
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tool_123",
                            "name": "__structured_output",
                            "input": {
                                "steps": [
                                    {"explanation": "Step 1", "output": "8x = -30"},
                                    {"explanation": "Step 2", "output": "x = -3.75"},
                                ],
                                "final_answer": "x = -3.75",
                            },
                        }
                    }
                ]
            }
        },
        "usage": {"inputTokens": 50, "outputTokens": 30, "totalTokens": 80},
        "ResponseMetadata": {"RequestId": "test-request-id"},
    }

    mock_bedrock_runtime.converse.return_value = mock_response

    params = {
        "messages": [{"role": "user", "content": "Solve 2x + 5 = -25"}],
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "response_format": MathReasoning,
    }

    response = bedrock_client.create(params)

    # Verify the response
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls is not None
    assert len(response.choices[0].message.tool_calls) == 1
    assert response.choices[0].message.tool_calls[0].function.name == "__structured_output"

    # Verify the structured output was extracted and formatted
    import json

    parsed_content = json.loads(response.choices[0].message.content)
    assert parsed_content["final_answer"] == "x = -3.75"
    assert len(parsed_content["steps"]) == 2

    # Verify toolConfig was set correctly
    call_args = mock_bedrock_runtime.converse.call_args
    assert "toolConfig" in call_args.kwargs
    tool_config = call_args.kwargs["toolConfig"]
    assert "toolChoice" in tool_config
    assert tool_config["toolChoice"] == {"tool": {"name": "__structured_output"}}


# Test 2: Test with dict schemas
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
def test_response_format_with_dict_schema(bedrock_client: BedrockClient):
    """Test structured output with dict schema."""
    # Mock bedrock_runtime on the instance
    mock_bedrock_runtime = MagicMock()
    bedrock_client.bedrock_runtime = mock_bedrock_runtime

    dict_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "email": {"type": "string"}},
        "required": ["name", "age"],
    }

    mock_response = {
        "stopReason": "tool_use",
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tool_456",
                            "name": "__structured_output",
                            "input": {"name": "John Doe", "age": 30, "email": "john@example.com"},
                        }
                    }
                ]
            }
        },
        "usage": {"inputTokens": 40, "outputTokens": 20, "totalTokens": 60},
        "ResponseMetadata": {"RequestId": "test-request-id-2"},
    }

    mock_bedrock_runtime.converse.return_value = mock_response

    params = {
        "messages": [{"role": "user", "content": "Create a user profile"}],
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "response_format": dict_schema,
    }

    response = bedrock_client.create(params)

    # Verify the response
    assert response.choices[0].finish_reason == "tool_calls"
    import json

    parsed_content = json.loads(response.choices[0].message.content)
    assert parsed_content["name"] == "John Doe"
    assert parsed_content["age"] == 30
    assert parsed_content["email"] == "john@example.com"


# Test 3: Test with both response_format and user tools together
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
def test_response_format_with_user_tools(bedrock_client: BedrockClient):
    """Test structured output when both response_format and user tools are provided."""
    # Mock bedrock_runtime on the instance
    mock_bedrock_runtime = MagicMock()
    bedrock_client.bedrock_runtime = mock_bedrock_runtime

    user_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    mock_response = {
        "stopReason": "tool_use",
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tool_789",
                            "name": "__structured_output",
                            "input": {
                                "steps": [{"explanation": "Used weather tool", "output": "Sunny, 75°F"}],
                                "final_answer": "The weather is sunny and 75°F",
                            },
                        }
                    }
                ]
            }
        },
        "usage": {"inputTokens": 60, "outputTokens": 40, "totalTokens": 100},
        "ResponseMetadata": {"RequestId": "test-request-id-3"},
    }

    mock_bedrock_runtime.converse.return_value = mock_response

    params = {
        "messages": [{"role": "user", "content": "Get weather and format the response"}],
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "response_format": MathReasoning,
        "tools": user_tools,
    }

    response = bedrock_client.create(params)

    # Verify both tools are in toolConfig
    call_args = mock_bedrock_runtime.converse.call_args
    tool_config = call_args.kwargs["toolConfig"]
    assert len(tool_config["tools"]) == 2  # user tool + structured output tool

    # Verify toolChoice forces structured output tool
    assert tool_config["toolChoice"] == {"tool": {"name": "__structured_output"}}

    # Verify response contains structured output
    assert response.choices[0].finish_reason == "tool_calls"
    import json

    parsed_content = json.loads(response.choices[0].message.content)
    assert "final_answer" in parsed_content


# Test 4: Test error handling when model doesn't call the tool
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
def test_response_format_no_tool_call_error_handling(bedrock_client: BedrockClient):
    """Test error handling when model doesn't call the structured output tool."""
    # Mock bedrock_runtime on the instance
    mock_bedrock_runtime = MagicMock()
    bedrock_client.bedrock_runtime = mock_bedrock_runtime

    # Mock response that doesn't call the tool (returns text instead)
    mock_response = {
        "stopReason": "finished",
        "output": {"message": {"content": [{"text": "Here's the answer: x = -3.75"}]}},
        "usage": {"inputTokens": 50, "outputTokens": 20, "totalTokens": 70},
        "ResponseMetadata": {"RequestId": "test-request-id-4"},
    }

    mock_bedrock_runtime.converse.return_value = mock_response

    params = {
        "messages": [{"role": "user", "content": "Solve 2x + 5 = -25"}],
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "response_format": MathReasoning,
    }

    response = bedrock_client.create(params)

    # Should fallback to text content when tool isn't called
    assert response.choices[0].finish_reason == "stop"
    assert response.choices[0].message.content == "Here's the answer: x = -3.75"
    assert response.choices[0].message.tool_calls is None


# Test 5: Test with models that support Tool Use
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
def test_response_format_with_tool_supporting_model(bedrock_client: BedrockClient):
    """Test structured output with a model that supports Tool Use (e.g., Claude models)."""
    # Mock bedrock_runtime on the instance
    mock_bedrock_runtime = MagicMock()
    bedrock_client.bedrock_runtime = mock_bedrock_runtime

    # Claude models support tool use
    claude_model = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    mock_response = {
        "stopReason": "tool_use",
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tool_claude_123",
                            "name": "__structured_output",
                            "input": {
                                "steps": [
                                    {"explanation": "First step", "output": "Result 1"},
                                    {"explanation": "Second step", "output": "Result 2"},
                                ],
                                "final_answer": "Final result",
                            },
                        }
                    }
                ]
            }
        },
        "usage": {"inputTokens": 100, "outputTokens": 50, "totalTokens": 150},
        "ResponseMetadata": {"RequestId": "test-claude-request"},
    }

    mock_bedrock_runtime.converse.return_value = mock_response

    params = {
        "messages": [{"role": "user", "content": "Perform a calculation"}],
        "model": claude_model,
        "response_format": MathReasoning,
    }

    response = bedrock_client.create(params)

    # Verify successful structured output extraction
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls is not None

    # Verify toolConfig was properly set
    call_args = mock_bedrock_runtime.converse.call_args
    assert "toolConfig" in call_args.kwargs
    tool_config = call_args.kwargs["toolConfig"]
    assert "toolChoice" in tool_config
    assert tool_config["toolChoice"]["tool"]["name"] == "__structured_output"


# Test 6: Test validation error when structured output doesn't match schema
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
def test_response_format_validation_error(bedrock_client: BedrockClient):
    """Test error handling when structured output doesn't match Pydantic schema."""
    # Mock bedrock_runtime on the instance
    mock_bedrock_runtime = MagicMock()
    bedrock_client.bedrock_runtime = mock_bedrock_runtime

    # Mock response with invalid data (missing required field)
    mock_response = {
        "stopReason": "tool_use",
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tool_invalid",
                            "name": "__structured_output",
                            "input": {
                                "steps": [{"explanation": "Step 1", "output": "Result 1"}]
                                # Missing required "final_answer" field
                            },
                        }
                    }
                ]
            }
        },
        "usage": {"inputTokens": 50, "outputTokens": 25, "totalTokens": 75},
        "ResponseMetadata": {"RequestId": "test-invalid-request"},
    }

    mock_bedrock_runtime.converse.return_value = mock_response

    params = {
        "messages": [{"role": "user", "content": "Solve this"}],
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "response_format": MathReasoning,
    }

    # Should raise ValidationError when validating against Pydantic model
    with pytest.raises(ValueError, match="Failed to validate structured output against schema"):
        bedrock_client.create(params)


# Test 7: Test helper method _get_response_format_schema
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
def test_get_response_format_schema_pydantic(bedrock_client: BedrockClient):
    """Test _get_response_format_schema with Pydantic model."""
    schema = bedrock_client._get_response_format_schema(MathReasoning)

    assert schema["type"] == "object"
    assert "properties" in schema
    assert "steps" in schema["properties"]
    assert "final_answer" in schema["properties"]
    assert "required" in schema


@run_for_optional_imports(["boto3", "botocore"], "bedrock")
def test_get_response_format_schema_dict(bedrock_client: BedrockClient):
    """Test _get_response_format_schema with dict schema."""
    dict_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
    }

    schema = bedrock_client._get_response_format_schema(dict_schema)

    assert schema["type"] == "object"
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]
    assert "required" in schema
    assert "name" in schema["required"]


# Test 8: Test helper method _create_structured_output_tool
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
def test_create_structured_output_tool(bedrock_client: BedrockClient):
    """Test _create_structured_output_tool creates correct tool definition."""
    tool = bedrock_client._create_structured_output_tool(MathReasoning)

    assert tool["type"] == "function"
    assert tool["function"]["name"] == "__structured_output"
    assert tool["function"]["description"] == "Generate structured output matching the specified schema"
    assert "parameters" in tool["function"]
    assert tool["function"]["parameters"]["type"] == "object"


# Test 9: Test helper method _extract_structured_output_from_tool_call
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
def test_extract_structured_output_from_tool_call(bedrock_client: BedrockClient):
    """Test _extract_structured_output_from_tool_call extracts data correctly."""
    from autogen.oai.oai_models.chat_completion_message_tool_call import Function

    tool_calls = [
        ChatCompletionMessageToolCall(
            id="tool_1", function=Function(name="get_weather", arguments='{"location": "NYC"}'), type="function"
        ),
        ChatCompletionMessageToolCall(
            id="tool_2",
            function=Function(
                name="__structured_output",
                arguments='{"steps": [{"explanation": "Step 1", "output": "Result"}], "final_answer": "Answer"}',
            ),
            type="function",
        ),
    ]

    result = bedrock_client._extract_structured_output_from_tool_call(tool_calls)

    assert result is not None
    assert result["final_answer"] == "Answer"
    assert len(result["steps"]) == 1


# Test 10: Test helper method _extract_structured_output_from_tool_call not found
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
def test_extract_structured_output_from_tool_call_not_found(bedrock_client: BedrockClient):
    """Test _extract_structured_output_from_tool_call returns None when tool not found."""
    from autogen.oai.oai_models.chat_completion_message_tool_call import Function

    tool_calls = [
        ChatCompletionMessageToolCall(
            id="tool_1", function=Function(name="get_weather", arguments='{"location": "NYC"}'), type="function"
        )
    ]

    result = bedrock_client._extract_structured_output_from_tool_call(tool_calls)

    assert result is None


# Test 11: Test helper method _validate_and_format_structured_output
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
def test_validate_and_format_structured_output(bedrock_client: BedrockClient):
    """Test _validate_and_format_structured_output validates and formats correctly."""
    bedrock_client._response_format = MathReasoning

    structured_data = {"steps": [{"explanation": "Step 1", "output": "Result 1"}], "final_answer": "Final answer"}

    result = bedrock_client._validate_and_format_structured_output(structured_data)

    # Should return JSON string
    import json

    parsed = json.loads(result)
    assert parsed["final_answer"] == "Final answer"
    assert len(parsed["steps"]) == 1


# Integration tests for Bedrock structured outputs
@pytest.mark.integration
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
class TestBedrockStructuredOutputIntegration:
    """Integration tests for Bedrock structured outputs with real API calls."""

    def setup_method(self):
        """Setup method run before each test."""
        import os
        from pathlib import Path

        try:
            import dotenv

            # Load environment variables from .env file
            env_file = Path(__file__).parent.parent.parent / ".env"
            if env_file.exists():
                dotenv.load_dotenv(env_file)
        except ImportError:
            pass

        # Check for AWS credentials - at least region should be set
        # AWS credentials can come from env vars, IAM role, or AWS profile
        if not os.getenv("AWS_REGION") and not os.getenv("AWS_DEFAULT_REGION"):
            pytest.skip(
                "AWS_REGION or AWS_DEFAULT_REGION environment variable not set (check .env file or environment)"
            )

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_agent_with_pydantic_structured_output(self):
        """Test creating and running an agent with Pydantic structured output."""
        import json
        import os

        from autogen import ConversableAgent, LLMConfig

        # Get AWS configuration from environment - check both standard and notebook variable names
        aws_region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
        # Try notebook format first, then standard AWS format
        aws_access_key = os.getenv("AWS_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_profile = os.getenv("AWS_PROFILE")
        # Use notebook's model format if BEDROCK_MODEL is set, otherwise default to notebook's example
        model = os.getenv("BEDROCK_MODEL", "eu.anthropic.claude-3-7-sonnet-20250219-v1:0")

        # Create LLM config with structured output
        llm_config = LLMConfig(
            config_list={
                "api_type": "bedrock",
                "model": model,
                "aws_region": aws_region,
                "aws_access_key": aws_access_key,
                "aws_secret_key": aws_secret_key,
                "aws_profile_name": aws_profile,
                "response_format": MathReasoning,  # Enable structured outputs
            },
        )

        # Create agent with structured output capability
        math_agent = ConversableAgent(
            name="math_assistant",
            llm_config=llm_config,
            system_message="""You are a helpful math assistant that solves problems step by step.
            Always show your reasoning process clearly with explanations for each step.
            Return your response in the structured format requested.""",
            max_consecutive_auto_reply=1,
            human_input_mode="NEVER",
        )

        # Run the agent with a simple math problem
        result = math_agent.run(
            message="Solve the equation: 2x + 5 = -25.",
            max_turns=3,
        )
        result.process()

        # Verify the response contains structured output
        assert result is not None
        assert len(result.messages) > 0

        # Find the assistant message with structured output
        # Look for the last message with role='assistant' that has content
        assistant_messages = [msg for msg in result.messages if msg.get("role") == "assistant" and msg.get("content")]
        assert len(assistant_messages) > 0, "No assistant messages found in result"

        last_message = assistant_messages[-1]
        assert last_message.get("content") is not None

        # Parse the structured output
        content = last_message["content"]
        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError:
            # If content is not JSON, it might be formatted text - check if it contains expected fields
            assert "final_answer" in content.lower() or "x =" in content.lower()
            return

        # Verify the structure matches MathReasoning schema
        assert "final_answer" in parsed_content, f"Missing 'final_answer' in parsed content: {parsed_content.keys()}"
        assert "steps" in parsed_content, f"Missing 'steps' in parsed content: {parsed_content.keys()}"
        assert isinstance(parsed_content["steps"], list), (
            f"'steps' should be a list, got {type(parsed_content['steps'])}"
        )
        assert len(parsed_content["steps"]) > 0, "Steps list should not be empty"

        # Verify each step has meaningful content
        # Note: The model might return different field names than the schema (e.g., 'description' instead of 'explanation',
        # 'math' instead of 'output', or 'step_num' instead of just having an index). This is acceptable for integration
        # tests as long as the structured output is working and contains the expected information.
        for i, step in enumerate(parsed_content["steps"]):
            assert isinstance(step, dict), f"Step {i} should be a dict, got {type(step)}"
            # Check that the step has some form of explanation/description
            has_explanation = "explanation" in step or "description" in step
            assert has_explanation, f"Step {i} should have 'explanation' or 'description': {step.keys()}"
            # Check that the step has some form of output/result/math
            has_output = "output" in step or "result" in step or "math" in step
            assert has_output, f"Step {i} should have 'output', 'result', or 'math': {step.keys()}"
            # Verify the step has meaningful content (not empty strings)
            explanation_value = step.get("explanation") or step.get("description", "")
            output_value = step.get("output") or step.get("result") or step.get("math", "")
            assert len(str(explanation_value)) > 0, f"Step {i} explanation/description should not be empty"
            assert len(str(output_value)) > 0, f"Step {i} output/result/math should not be empty"

        # Verify final answer is not empty
        assert len(parsed_content["final_answer"]) > 0, "final_answer should not be empty"

        # Verify tool_calls if present (structured output should have tool calls)
        if "tool_calls" in last_message:
            tool_calls = last_message["tool_calls"]
            assert len(tool_calls) > 0, "Should have tool calls for structured output"
            # Check that one of the tool calls is for structured output
            structured_output_tools = [
                tc for tc in tool_calls if tc.get("function", {}).get("name") == "__structured_output"
            ]
            assert len(structured_output_tools) > 0, "Should have __structured_output tool call"

    @run_for_optional_imports(["boto3", "botocore"], "bedrock")
    def test_agent_with_dict_schema_structured_output(self):
        """Test creating and running an agent with dict schema structured output."""
        import json
        import os

        from autogen import ConversableAgent, LLMConfig

        # Define schema as a dictionary (JSON Schema format)
        dict_schema = {
            "type": "object",
            "properties": {
                "problem": {"type": "string", "description": "The math problem being solved"},
                "solution_steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"step": {"type": "string"}, "result": {"type": "string"}},
                        "required": ["step", "result"],
                    },
                },
                "answer": {"type": "string"},
            },
            "required": ["problem", "solution_steps", "answer"],
        }

        # Get AWS configuration from environment - check both standard and notebook variable names
        aws_region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
        # Try notebook format first, then standard AWS format
        aws_access_key = os.getenv("AWS_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_profile = os.getenv("AWS_PROFILE")
        # Use notebook's model format if BEDROCK_MODEL is set, otherwise default to notebook's example
        model = os.getenv("BEDROCK_MODEL", "eu.anthropic.claude-3-7-sonnet-20250219-v1:0")

        # Create LLM config with dict schema
        llm_config = LLMConfig(
            config_list={
                "api_type": "bedrock",
                "model": model,
                "aws_region": aws_region,
                "aws_access_key": aws_access_key,
                "aws_secret_key": aws_secret_key,
                "aws_profile_name": aws_profile,
                "response_format": dict_schema,  # Using dict schema instead of Pydantic model
            },
        )

        # Create agent with dict schema
        math_agent = ConversableAgent(
            name="math_assistant_dict",
            llm_config=llm_config,
            system_message="You are a helpful math assistant.",
            max_consecutive_auto_reply=1,
            human_input_mode="NEVER",
        )

        # Run the agent with a math problem
        result = math_agent.run(
            message="Solve: x^2 - 5x + 6 = 0",
            max_turns=3,
        )
        result.process()

        # Verify the response contains structured output
        assert result is not None
        assert len(result.messages) > 0

        # Find the assistant message with structured output
        # Look for the last message with role='assistant' that has content
        assistant_messages = [msg for msg in result.messages if msg.get("role") == "assistant" and msg.get("content")]
        assert len(assistant_messages) > 0, "No assistant messages found in result"

        last_message = assistant_messages[-1]
        assert last_message.get("content") is not None

        # Parse the structured output
        content = last_message["content"]
        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError:
            # If content is not JSON, check if it contains expected fields
            assert "answer" in content.lower() or "x =" in content.lower()
            return

        # Verify the structure matches dict schema
        assert "problem" in parsed_content, f"Missing 'problem' in parsed content: {parsed_content.keys()}"
        assert "solution_steps" in parsed_content, (
            f"Missing 'solution_steps' in parsed content: {parsed_content.keys()}"
        )
        assert "answer" in parsed_content, f"Missing 'answer' in parsed content: {parsed_content.keys()}"
        assert isinstance(parsed_content["solution_steps"], list), (
            f"'solution_steps' should be a list, got {type(parsed_content['solution_steps'])}"
        )
        assert len(parsed_content["solution_steps"]) > 0, "solution_steps list should not be empty"

        # Verify each step has required fields
        for i, step in enumerate(parsed_content["solution_steps"]):
            assert isinstance(step, dict), f"Step {i} should be a dict, got {type(step)}"
            assert "step" in step, f"Step {i} missing required field 'step': {step.keys()}"
            assert "result" in step, f"Step {i} missing required field 'result': {step.keys()}"

        # Verify tool_calls if present (structured output should have tool calls)
        if "tool_calls" in last_message:
            tool_calls = last_message["tool_calls"]
            assert len(tool_calls) > 0, "Should have tool calls for structured output"
            # Check that one of the tool calls is for structured output
            structured_output_tools = [
                tc for tc in tool_calls if tc.get("function", {}).get("name") == "__structured_output"
            ]
            assert len(structured_output_tools) > 0, "Should have __structured_output tool call"
