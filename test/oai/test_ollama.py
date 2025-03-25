# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from autogen.import_utils import run_for_optional_imports
from autogen.llm_config import LLMConfig
from autogen.oai.ollama import OllamaClient, OllamaLLMConfigEntry, response_to_tool_call


# Define test Pydantic model
class Step(BaseModel):
    explanation: str
    output: str


class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str


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
def ollama_client():
    # Set Ollama client with some default values
    client = OllamaClient()

    client._native_tool_calls = True
    client._tools_in_conversation = False

    return client


@pytest.fixture
def ollama_client_maths_format():
    # Set Ollama client with some default values
    client = OllamaClient(response_format=MathReasoning)

    client._native_tool_calls = True
    client._tools_in_conversation = False

    return client


def test_ollama_llm_config_entry():
    ollama_llm_config = OllamaLLMConfigEntry(model="llama3.1:8b")
    expected = {
        "api_type": "ollama",
        "model": "llama3.1:8b",
        "num_ctx": 2048,
        "num_predict": -1,
        "repeat_penalty": 1.1,
        "seed": 0,
        "stream": False,
        "tags": [],
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.9,
        "hide_tools": "never",
    }
    actual = ollama_llm_config.model_dump()
    assert actual == expected, actual

    llm_config = LLMConfig(
        config_list=[ollama_llm_config],
    )
    assert llm_config.model_dump() == {
        "config_list": [expected],
    }


# Test initialization and configuration
@run_for_optional_imports(["ollama", "fix_busted_json"], "ollama")
def test_initialization():
    # Creation works without an api_key
    OllamaClient()


# Test parameters
@run_for_optional_imports(["ollama", "fix_busted_json"], "ollama")
def test_parsing_params(ollama_client):
    # All parameters (with default values)
    params = {
        "model": "llama3.1:8b",
        "temperature": 0.8,
        "num_predict": 128,
        "num_ctx": 2048,
        "repeat_penalty": 1.1,
        "seed": 42,
        "top_k": 40,
        "top_p": 0.9,
        "stream": False,
    }
    expected_params = {
        "model": "llama3.1:8b",
        "options": {
            "repeat_penalty": 1.1,
            "seed": 42,
            "temperature": 0.8,
            "num_predict": 128,
            "num_ctx": 2048,
            "top_k": 40,
            "top_p": 0.9,
        },
        "stream": False,
    }
    result = ollama_client.parse_params(params)
    assert result == expected_params

    # Incorrect types, defaults should be set, will show warnings but not trigger assertions
    params = {
        "model": "llama3.1:8b",
        "temperature": "0.5",
        "num_predict": "128",
        "num_ctx": 2048,
        "repeat_penalty": "1.1",
        "seed": "42",
        "top_k": "40",
        "top_p": "0.9",
        "stream": "True",
    }
    result = ollama_client.parse_params(params)
    assert result == expected_params

    # Only model, others set as defaults if they are mandatory
    params = {
        "model": "llama3.1:8b",
    }
    expected_params = {"model": "llama3.1:8b", "stream": False}
    result = ollama_client.parse_params(params)
    assert result == expected_params

    # No model
    params = {
        "temperature": 0.8,
    }

    with pytest.raises(AssertionError) as assertinfo:
        result = ollama_client.parse_params(params)

    assert "Please specify the 'model' in your config list entry to nominate the Ollama model to use." in str(
        assertinfo.value
    )


# Test text generation
@run_for_optional_imports(["ollama", "fix_busted_json"], "ollama")
@patch("autogen.oai.ollama.OllamaClient.create")
def test_create_response(mock_chat, ollama_client):
    # Mock OllamaClient.chat response
    mock_ollama_response = MagicMock()
    mock_ollama_response.choices = [
        MagicMock(finish_reason="stop", message=MagicMock(content="Example Ollama response", tool_calls=None))
    ]
    mock_ollama_response.id = "mock_ollama_response_id"
    mock_ollama_response.model = "llama3.1:8b"
    mock_ollama_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)  # Example token usage

    mock_chat.return_value = mock_ollama_response

    # Test parameters
    params = {
        "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "World"}],
        "model": "llama3.1:8b",
    }

    # Call the create method
    response = ollama_client.create(params)

    # Assertions to check if response is structured as expected
    assert response.choices[0].message.content == "Example Ollama response", (
        "Response content should match expected output"
    )
    assert response.id == "mock_ollama_response_id", "Response ID should match the mocked response ID"
    assert response.model == "llama3.1:8b", "Response model should match the mocked response model"
    assert response.usage.prompt_tokens == 10, "Response prompt tokens should match the mocked response usage"
    assert response.usage.completion_tokens == 20, "Response completion tokens should match the mocked response usage"


# Test text generation
@run_for_optional_imports(["ollama", "fix_busted_json"], "ollama")
@patch("autogen.oai.ollama.OllamaClient.create")
def test_ollama_client_host_value(ollama_client):
    from autogen import ConversableAgent, LLMConfig

    config_list = [
        {
            "model": "llama3.3",
            "api_type": "ollama",
            "api_key": "NULL",
            "client_host": "http://localhost:11434",
            "stream": False,
        }
    ]

    llm_config = LLMConfig(config_list=config_list)
    system_message = "You are a helpful assistant."

    # Create the agent with the specified configuration
    my_agent = ConversableAgent(name="helpful_agent", llm_config=llm_config, system_message=system_message)
    assert my_agent.client is not None, "Client should be initialized"
    assert my_agent.client._config_list is not None, "Client config list should be initialized"
    assert my_agent.client._config_list[0]["model"] == "llama3.3", "Model should match the specified value"
    assert str(my_agent.client._config_list[0]["client_host"]) == "http://localhost:11434/", (
        "client_host should match the specified value"
    )


# Test functions/tools
@run_for_optional_imports(["ollama", "fix_busted_json"], "ollama")
@patch("autogen.oai.ollama.OllamaClient.create")
def test_create_response_with_tool_call(mock_chat, ollama_client):
    # Mock OllamaClient.chat response
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
                        MagicMock(id="gdRdrvnHh", function=mock_function),
                        MagicMock(id="abRdrvnHh", function=mock_function_2),
                    ],
                ),
            )
        ],
        id="mock_ollama_response_id",
        model="llama3.1:8b",
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
    ollama_messages = [
        {"role": "user", "content": "How much is 123.45 EUR in USD?"},
        {"role": "assistant", "content": "World"},
    ]

    # Call the create method
    response = ollama_client.create({"messages": ollama_messages, "tools": converted_functions, "model": "llama3.1:8b"})

    # Assertions to check if the functions and content are included in the response
    assert response.choices[0].message.content == "Sample text about the functions"
    assert response.choices[0].message.tool_calls[0].function.name == "currency_calculator"
    assert response.choices[0].message.tool_calls[1].function.name == "get_weather"


# Test function parsing with manual tool calling
@run_for_optional_imports(["ollama", "fix_busted_json"], "ollama")
def test_manual_tool_calling_parsing(ollama_client):
    # Test the parsing of a tool call within the response content (fully correct)
    response_content = """[{"name": "weather_forecast", "arguments":{"location": "New York"}},{"name": "currency_calculator", "arguments":{"base_amount": 123.45, "quote_currency": "EUR", "base_currency": "USD"}}]"""

    response_tool_calls = response_to_tool_call(response_content)

    expected_tool_calls = [
        {"name": "weather_forecast", "arguments": {"location": "New York"}},
        {
            "name": "currency_calculator",
            "arguments": {"base_amount": 123.45, "quote_currency": "EUR", "base_currency": "USD"},
        },
    ]

    assert response_tool_calls == expected_tool_calls, (
        "Manual Tool Calling Parsing of response did not yield correct tool_calls (full string match)"
    )

    # Test the parsing with a substring containing the response content (should still pass)
    response_content = """I will call two functions, weather_forecast and currency_calculator:\n[{"name": "weather_forecast", "arguments":{"location": "New York"}},{"name": "currency_calculator", "arguments":{"base_amount": 123.45, "quote_currency": "EUR", "base_currency": "USD"}}]"""

    response_tool_calls = response_to_tool_call(response_content)

    assert response_tool_calls == expected_tool_calls, (
        "Manual Tool Calling Parsing of response did not yield correct tool_calls (partial string match)"
    )

    # Test the parsing with an invalid function call
    response_content = """[{"function": "weather_forecast", "args":{"location": "New York"}},{"function": "currency_calculator", "args":{"base_amount": 123.45, "quote_currency": "EUR", "base_currency": "USD"}}]"""

    response_tool_calls = response_to_tool_call(response_content)

    assert response_tool_calls is None, (
        "Manual Tool Calling Parsing of response did not yield correct tool_calls (invalid function call)"
    )

    # Test the parsing with plain text
    response_content = """Call the weather_forecast function and pass in 'New York' as the 'location' argument."""

    response_tool_calls = response_to_tool_call(response_content)

    assert response_tool_calls is None, (
        "Manual Tool Calling Parsing of response did not yield correct tool_calls (no function in text)"
    )


# Test message conversion from OpenAI to Ollama format
@run_for_optional_imports(["ollama", "fix_busted_json"], "ollama")
def test_oai_messages_to_ollama_messages(ollama_client):
    # Test that the "name" key is removed
    test_messages = [
        {"role": "system", "content": "You are a helpful AI bot."},
        {"role": "user", "name": "anne", "content": "Why is the sky blue?"},
    ]
    messages = ollama_client.oai_messages_to_ollama_messages(test_messages, None)

    expected_messages = [
        {"role": "system", "content": "You are a helpful AI bot."},
        {"role": "user", "content": "Why is the sky blue?"},
    ]

    assert messages == expected_messages, "'name' was not removed from messages"

    # Test that there isn't a final system message and it's changed to user
    test_messages.append({"role": "system", "content": "Summarise the conversation."})

    messages = ollama_client.oai_messages_to_ollama_messages(test_messages, None)

    expected_messages = [
        {"role": "system", "content": "You are a helpful AI bot."},
        {"role": "user", "content": "Why is the sky blue?"},
        {"role": "user", "content": "Summarise the conversation."},
    ]

    assert messages == expected_messages, "Final 'system' message was not changed to 'user'"

    # Test that the last message is a user or system message and if not, add a continue message
    test_messages[2] = {"role": "assistant", "content": "The sky is blue because that's a great colour."}

    messages = ollama_client.oai_messages_to_ollama_messages(test_messages, None)

    expected_messages = [
        {"role": "system", "content": "You are a helpful AI bot."},
        {"role": "user", "content": "Why is the sky blue?"},
        {"role": "assistant", "content": "The sky is blue because that's a great colour."},
        {"role": "user", "content": "Please continue."},
    ]

    assert messages == expected_messages, "'Please continue' message was not appended."


# Test message conversion from OpenAI to Ollama format
@run_for_optional_imports(["ollama", "fix_busted_json"], "ollama")
def test_extract_json_response(ollama_client):
    # Set up the response format
    ollama_client._response_format = MathReasoning

    # Test case 1: JSON within tags - CORRECT
    tagged_response = """{
                "steps": [
                    {"explanation": "Step 1", "output": "8x = -30"},
                    {"explanation": "Step 2", "output": "x = -3.75"}
                ],
                "final_answer": "x = -3.75"
            }"""

    result = ollama_client._convert_json_response(tagged_response)
    assert isinstance(result, MathReasoning)
    assert len(result.steps) == 2
    assert result.final_answer == "x = -3.75"

    # Test case 2: Invalid JSON - RAISE ERROR
    invalid_response = """{
                "steps": [
                    {"explanation": "Step 1", "output": "8x = -30"},
                    {"explanation": "Missing closing brace"
                ],
                "final_answer": "x = -3.75"
            """

    with pytest.raises(
        ValueError, match="Failed to parse response as valid JSON matching the schema for Structured Output: "
    ):
        ollama_client._convert_json_response(invalid_response)

    # Test case 3: No JSON content - RAISE ERROR
    no_json_response = "This response contains no JSON at all."

    with pytest.raises(
        ValueError,
        match="Failed to parse response as valid JSON matching the schema for Structured Output:",
    ):
        ollama_client._convert_json_response(no_json_response)


# Test message conversion from OpenAI to Ollama format
@run_for_optional_imports(["ollama", "fix_busted_json"], "ollama")
def test_extract_json_response_client(ollama_client_maths_format):
    # Test case 1: JSON within tags - CORRECT
    tagged_response = """{
                "steps": [
                    {"explanation": "Step 1", "output": "8x = -30"},
                    {"explanation": "Step 2", "output": "x = -3.75"}
                ],
                "final_answer": "x = -3.75"
            }"""

    result = ollama_client_maths_format._convert_json_response(tagged_response)
    assert isinstance(result, MathReasoning)
    assert len(result.steps) == 2
    assert result.final_answer == "x = -3.75"

    # Test case 2: Invalid JSON - RAISE ERROR
    invalid_response = """{
                "steps": [
                    {"explanation": "Step 1", "output": "8x = -30"},
                    {"explanation": "Missing closing brace"
                ],
                "final_answer": "x = -3.75"
            """

    with pytest.raises(
        ValueError, match="Failed to parse response as valid JSON matching the schema for Structured Output: "
    ):
        ollama_client_maths_format._convert_json_response(invalid_response)

    # Test case 3: No JSON content - RAISE ERROR
    no_json_response = "This response contains no JSON at all."

    with pytest.raises(
        ValueError,
        match="Failed to parse response as valid JSON matching the schema for Structured Output:",
    ):
        ollama_client_maths_format._convert_json_response(no_json_response)


# Test message conversion from OpenAI to Ollama format
@run_for_optional_imports(["ollama", "fix_busted_json"], "ollama")
def test_extract_json_response_params(ollama_client):
    # All parameters (with default values)
    params = {
        "model": "llama3.1:8b",
        "temperature": 0.8,
        "num_predict": 128,
        "num_ctx": 2048,
        "repeat_penalty": 1.1,
        "seed": 42,
        "top_k": 40,
        "top_p": 0.9,
        "stream": False,
        "response_format": MathReasoning,
    }

    ollama_params = ollama_client.parse_params(params)

    converted_dict = MathReasoning.model_json_schema()

    assert isinstance(ollama_params["format"], dict)
    assert ollama_params["format"] == converted_dict
