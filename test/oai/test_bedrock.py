# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import asyncio
import json
import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from autogen.agentchat import ConversableAgent
from autogen.import_utils import run_for_optional_imports
from autogen.llm_config import LLMConfig
from autogen.oai.bedrock import BedrockClient, BedrockLLMConfigEntry, format_tools, oai_messages_to_bedrock_messages
from test.credentials import Credentials, get_credentials_from_env_vars


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


def test_format_tools_handles_various_property_shapes():
    """format_tools should faithfully copy every supported JSON Schema shape (scalars, enums, unions, arrays, nested objects)."""
    cases = [
        (
            "simple_type",
            {"type": "string", "description": "plain type"},
            {"type": "string", "description": "plain type"},
        ),
        (
            "enum_default",
            {"type": "integer", "enum": [1, 2], "default": 1},
            {"type": "integer", "enum": [1, 2], "default": 1, "description": ""},
        ),
        (
            "union_anyof",
            {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "description": "optional text",
            },
            {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "description": "optional text",
            },
        ),
        (
            "array_items",
            {"type": "array", "items": {"type": "number"}, "minItems": 1},
            {"type": "array", "items": {"type": "number"}, "minItems": 1, "description": ""},
        ),
        (
            "object_additional",
            {
                "type": "object",
                "additionalProperties": {"type": "boolean"},
                "required": [],
            },
            {
                "type": "object",
                "additionalProperties": {"type": "boolean"},
                "description": "",
            },
        ),
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "schema_tester",
                "description": "verifies schema copying",
                "parameters": {
                    "type": "object",
                    "properties": {name: prop for name, prop, _ in cases},
                },
            },
        }
    ]

    converted_props = format_tools(tools)["tools"][0]["toolSpec"]["inputSchema"]["json"]["properties"]

    for name, _, expected in cases:
        assert converted_props[name] == expected, f"schema mismatch for {name}"


def test_format_tools_rejects_non_dict_properties():
    """format_tools should raise TypeError when a property schema is not a dict, mirroring runtime validation."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bad_prop",
                "description": "schema with malformed property",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "oops": "not a dict",
                    },
                },
            },
        }
    ]

    with pytest.raises(TypeError, match="Property 'oops' schema must be a dict"):
        format_tools(tools)


# Integration tests with real Bedrock credentials
@pytest.fixture
def bedrock_credentials() -> Credentials:
    """Fixture to get Bedrock credentials from environment variables."""
    try:
        return get_credentials_from_env_vars(filter_dict={"api_type": "bedrock"})
    except Exception:
        pytest.skip("Bedrock credentials not available (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY required)")


@run_for_optional_imports(["boto3", "botocore"], "bedrock")
@pytest.mark.integration
def test_execute_function_resolves_async_tool_with_bedrock(bedrock_credentials: Credentials):
    """Integration test: execute_function should await async tools instead of returning coroutine reprs with real Bedrock client."""
    agent = ConversableAgent(name="agent", llm_config=bedrock_credentials.llm_config)
    observed_inputs: list[str] = []

    @agent.register_for_execution()
    @agent.register_for_llm(description="Uppercase text asynchronously")
    async def uppercase_tool(text: str) -> str:
        observed_inputs.append(text)
        await asyncio.sleep(0)
        return text.upper()

    success, payload = agent.execute_function(
        {"name": "uppercase_tool", "arguments": json.dumps({"text": "nyc"})},
        call_id="tool-call-1",
    )

    assert success is True
    assert payload["content"] == "NYC"
    assert observed_inputs == ["nyc"]


@run_for_optional_imports(["boto3", "botocore"], "bedrock")
@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.skipif(
    not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")),
    reason="Bedrock credentials not available (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY required)",
)
async def test_async_tool_execution_with_bedrock_integration(bedrock_credentials: Credentials):
    """Integration test: async tool execution with real Bedrock client, checking summary and chat history."""
    agent = ConversableAgent(
        name="agent",
        llm_config=bedrock_credentials.llm_config,
        system_message="You are a helpful assistant. Use tools when needed.",
    )

    observed_inputs: list[str] = []

    @agent.register_for_execution()
    @agent.register_for_llm(description="Uppercase text asynchronously. Useful for converting text to uppercase.")
    async def uppercase_tool(text: str) -> str:
        """Uppercase the given text."""
        observed_inputs.append(text)
        await asyncio.sleep(0.1)  # Simulate async operation
        return text.upper()

    # Run the agent with a message that should trigger tool usage
    result = await agent.a_run(
        message="say 'hello bedrock'",
        max_turns=3,
        user_input=False,
        summary_method="reflection_with_llm",
    )

    # Wait for the result to process
    await result.process()

    # Assert summary is generated and contains relevant information
    summary = await result.summary
    assert "hello" in summary.lower()
    all_content = " ".join([
        msg.get("content", "") for msg in result.chat_history if isinstance(msg.get("content"), str)
    ])
    assert "HELLO BEDROCK" in all_content or "hello bedrock".upper() in all_content, (
        f"Chat history should contain the uppercase result. Content: {all_content[:200]}"
    )
