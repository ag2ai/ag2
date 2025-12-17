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

# Test additionalModelRequestFields parsing
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
def test_parsing_params_with_additional_model_request_fields(bedrock_client: BedrockClient):
    """Test that additionalModelRequestFields are correctly parsed and added to additional_params."""
    # Test with thinking configuration (main use case)
    base_params, additional_params = bedrock_client.parse_params({
        "model": "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "temperature": 0.8,
        "additionalModelRequestFields": {
            "thinking": {
                "type": "enabled",
                "budget_tokens": 1024,
            },
        },
    })

    assert base_params == {"temperature": 0.8}
    assert additional_params == {
        "thinking": {
            "type": "enabled",
            "budget_tokens": 1024,
        },
    }

    # Test with multiple fields in additionalModelRequestFields
    base_params, additional_params = bedrock_client.parse_params({
        "model": "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "additionalModelRequestFields": {
            "thinking": {
                "type": "enabled",
                "budget_tokens": 512,
            },
            "custom_field": "custom_value",
            "nested_config": {
                "key1": "value1",
                "key2": 42,
            },
        },
    })

    assert base_params == {}
    assert additional_params == {
        "thinking": {
            "type": "enabled",
            "budget_tokens": 512,
        },
        "custom_field": "custom_value",
        "nested_config": {
            "key1": "value1",
            "key2": 42,
        },
    }

    # Test that config_only_fields are excluded from additional_params
    base_params, additional_params = bedrock_client.parse_params({
        "model": "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "additionalModelRequestFields": {
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "api_type": "should_be_excluded",
            "model": "should_be_excluded",
            "aws_region": "should_be_excluded",
            "messages": "should_be_excluded",
            "tools": "should_be_excluded",
            "response_format": "should_be_excluded",
        },
    })

    assert "thinking" in additional_params
    assert "api_type" not in additional_params
    assert "model" not in additional_params
    assert "aws_region" not in additional_params
    assert "messages" not in additional_params
    assert "tools" not in additional_params
    assert "response_format" not in additional_params

    # Test with empty additionalModelRequestFields dict
    base_params, additional_params = bedrock_client.parse_params({
        "model": "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "additionalModelRequestFields": {},
    })

    assert base_params == {}
    assert additional_params == {}

    # Test that additionalModelRequestFields merges with other additional params like seed
    base_params, additional_params = bedrock_client.parse_params({
        "model": "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "seed": 42,
        "top_k": 10,
        "additionalModelRequestFields": {
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
    })

    assert base_params == {}
    assert additional_params == {
        "seed": 42,
        "top_k": 10,
        "thinking": {
            "type": "enabled",
            "budget_tokens": 1024,
        },
    }

    # Test with None additionalModelRequestFields (should be ignored)
    base_params, additional_params = bedrock_client.parse_params({
        "model": "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "additionalModelRequestFields": None,
    })

    assert base_params == {}
    assert additional_params == {}

    # Test with non-dict additionalModelRequestFields (should be ignored)
    base_params, additional_params = bedrock_client.parse_params({
        "model": "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "additionalModelRequestFields": "not_a_dict",
    })

    assert base_params == {}
    assert additional_params == {}


# Test BedrockLLMConfigEntry with additionalModelRequestFields
def test_bedrock_llm_config_entry_with_additional_model_request_fields():
    """Test BedrockLLMConfigEntry accepts additionalModelRequestFields."""
    thinking_config = {
        "type": "enabled",
        "budget_tokens": 1024,
    }

    bedrock_llm_config = BedrockLLMConfigEntry(
        model="anthropic.claude-3-7-sonnet-20250219-v1:0",
        aws_region="us-east-1",
        additionalModelRequestFields={"thinking": thinking_config},
        temperature=0.8,
    )

    expected = {
        "api_type": "bedrock",
        "model": "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "aws_region": "us-east-1",
        "temperature": 0.8,
        "tags": [],
        "supports_system_prompts": True,
        "additionalModelRequestFields": {"thinking": thinking_config},
    }

    actual = bedrock_llm_config.model_dump()
    assert actual == expected

    # Verify it works with LLMConfig
    llm_config = LLMConfig(bedrock_llm_config)
    config_list = llm_config.model_dump()["config_list"]
    assert len(config_list) == 1
    assert config_list[0]["additionalModelRequestFields"] == {"thinking": thinking_config}


# Test edge case: additionalModelRequestFields with None value
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
def test_parsing_params_additional_model_request_fields_with_none_values(bedrock_client: BedrockClient):
    """Test that None values in additionalModelRequestFields are handled correctly."""
    base_params, additional_params = bedrock_client.parse_params({
        "model": "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "additionalModelRequestFields": {
            "thinking": None,
            "valid_field": "valid_value",
        },
    })

    # None values should still be included (let Bedrock API handle validation)
    assert "thinking" in additional_params
    assert additional_params["thinking"] is None
    assert additional_params["valid_field"] == "valid_value"