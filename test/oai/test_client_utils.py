# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

from unittest.mock import MagicMock

import pytest

from autogen.import_utils import run_for_optional_imports
from autogen.oai.client_utils import (
    should_hide_tools,
    standardize_api_error,
    validate_openai_client,
    validate_parameter,
)


def test_validate_parameter():
    # Test valid parameters
    params = {
        "model": "Qwen/Qwen2-72B-Instruct",
        "max_tokens": 1000,
        "stream": False,
        "temperature": 1,
        "top_p": 0.8,
        "top_k": 50,
        "repetition_penalty": 0.5,
        "presence_penalty": 1.5,
        "frequency_penalty": 1.5,
        "min_p": 0.2,
        "safety_model": "Meta-Llama/Llama-Guard-7b",
    }

    # Should return the original value as they are valid
    assert params["model"] == validate_parameter(params, "model", str, False, None, None, None)
    assert params["max_tokens"] == validate_parameter(params, "max_tokens", int, True, 512, (0, None), None)
    assert params["stream"] == validate_parameter(params, "stream", bool, False, False, None, None)
    assert params["temperature"] == validate_parameter(params, "temperature", (int, float), True, None, None, None)
    assert params["top_k"] == validate_parameter(params, "top_k", int, True, None, None, None)
    assert params["repetition_penalty"] == validate_parameter(
        params, "repetition_penalty", float, True, None, None, None
    )
    assert params["presence_penalty"] == validate_parameter(
        params, "presence_penalty", (int, float), True, None, (-2, 2), None
    )
    assert params["safety_model"] == validate_parameter(params, "safety_model", str, True, None, None, None)

    # Test None allowed
    params = {
        "max_tokens": None,
    }

    # Should remain None
    assert validate_parameter(params, "max_tokens", int, True, 512, (0, None), None) is None

    # Test not None allowed
    params = {
        "max_tokens": None,
    }

    # Should return default
    assert validate_parameter(params, "max_tokens", int, False, 512, (0, None), None) == 512

    # Test invalid parameters
    params = {
        "stream": "Yes",
        "temperature": "0.5",
        "top_p": "0.8",
        "top_k": "50",
        "repetition_penalty": "0.5",
        "presence_penalty": "1.5",
        "frequency_penalty": "1.5",
        "min_p": "0.2",
        "safety_model": False,
    }

    # Should all be set to defaults
    assert validate_parameter(params, "stream", bool, False, False, None, None) is not None
    assert validate_parameter(params, "temperature", (int, float), True, None, None, None) is None
    assert validate_parameter(params, "top_p", (int, float), True, None, None, None) is None
    assert validate_parameter(params, "top_k", int, True, None, None, None) is None
    assert validate_parameter(params, "repetition_penalty", float, True, None, None, None) is None
    assert validate_parameter(params, "presence_penalty", (int, float), True, None, (-2, 2), None) is None
    assert validate_parameter(params, "frequency_penalty", (int, float), True, None, (-2, 2), None) is None
    assert validate_parameter(params, "min_p", (int, float), True, None, (0, 1), None) is None
    assert validate_parameter(params, "safety_model", str, True, None, None, None) is None

    # Test parameters outside of bounds
    params = {
        "max_tokens": -200,
        "presence_penalty": -5,
        "frequency_penalty": 5,
        "min_p": -0.5,
    }

    # Should all be set to defaults
    assert validate_parameter(params, "max_tokens", int, True, 512, (0, None), None) == 512
    assert validate_parameter(params, "presence_penalty", (int, float), True, None, (-2, 2), None) is None
    assert validate_parameter(params, "frequency_penalty", (int, float), True, None, (-2, 2), None) is None
    assert validate_parameter(params, "min_p", (int, float), True, None, (0, 1), None) is None

    # Test valid list options
    params = {
        "safety_model": "Meta-Llama/Llama-Guard-7b",
    }

    # Should all be set to defaults
    assert (
        validate_parameter(
            params, "safety_model", str, True, None, None, ["Meta-Llama/Llama-Guard-7b", "Meta-Llama/Llama-Guard-13b"]
        )
        == "Meta-Llama/Llama-Guard-7b"
    )

    # Test invalid list options
    params = {
        "stream": True,
    }

    # Should all be set to defaults
    assert not validate_parameter(params, "stream", bool, False, False, None, [False])

    # test invalid type
    params = {
        "temperature": None,
    }

    # should be set to defaults
    assert validate_parameter(params, "temperature", (int, float), False, 0.7, (0.0, 1.0), None) == 0.7

    # test value out of bounds
    params = {
        "temperature": 23,
    }

    # should be set to defaults
    assert validate_parameter(params, "temperature", (int, float), False, 1.0, (0.0, 1.0), None) == 1.0

    # type error for the parameters
    with pytest.raises(TypeError):
        validate_parameter({}, "param", str, True, None, None, "not_a_list")

    # passing empty params, which will set to defaults
    assert validate_parameter({}, "max_tokens", int, True, 512, (0, None), None) == 512


def test_should_hide_tools():
    # Test messages
    no_tools_called_messages = [
        {"content": "You are a chess program and are playing for player white.", "role": "system"},
        {"content": "Let's play chess! Make a move.", "role": "user"},
        {
            "tool_calls": [
                {
                    "id": "call_abcde56o5jlugh9uekgo84c6",
                    "function": {"arguments": "{}", "name": "get_legal_moves"},
                    "type": "function",
                }
            ],
            "content": None,
            "role": "assistant",
        },
        {
            "tool_calls": [
                {
                    "id": "call_p1fla56o5jlugh9uekgo84c6",
                    "function": {"arguments": "{}", "name": "get_legal_moves"},
                    "type": "function",
                }
            ],
            "content": None,
            "role": "assistant",
        },
        {
            "tool_calls": [
                {
                    "id": "call_lcow1j0ehuhrcr3aakdmd9ju",
                    "function": {"arguments": '{"move":"g1f3"}', "name": "make_move"},
                    "type": "function",
                }
            ],
            "content": None,
            "role": "assistant",
        },
    ]
    one_tool_called_messages = [
        {"content": "You are a chess program and are playing for player white.", "role": "system"},
        {"content": "Let's play chess! Make a move.", "role": "user"},
        {
            "tool_calls": [
                {
                    "id": "call_abcde56o5jlugh9uekgo84c6",
                    "function": {"arguments": "{}", "name": "get_legal_moves"},
                    "type": "function",
                }
            ],
            "content": None,
            "role": "assistant",
        },
        {
            "tool_call_id": "call_abcde56o5jlugh9uekgo84c6",
            "role": "user",
            "content": "Possible moves are: g1h3,g1f3,b1c3,b1a3,h2h3,g2g3,f2f3,e2e3,d2d3,c2c3,b2b3,a2a3,h2h4,g2g4,f2f4,e2e4,d2d4,c2c4,b2b4,a2a4",
        },
        {
            "tool_calls": [
                {
                    "id": "call_lcow1j0ehuhrcr3aakdmd9ju",
                    "function": {"arguments": '{"move":"g1f3"}', "name": "make_move"},
                    "type": "function",
                }
            ],
            "content": None,
            "role": "assistant",
        },
    ]
    messages = [
        {"content": "You are a chess program and are playing for player white.", "role": "system"},
        {"content": "Let's play chess! Make a move.", "role": "user"},
        {
            "tool_calls": [
                {
                    "id": "call_abcde56o5jlugh9uekgo84c6",
                    "function": {"arguments": "{}", "name": "get_legal_moves"},
                    "type": "function",
                }
            ],
            "content": None,
            "role": "assistant",
        },
        {
            "tool_call_id": "call_abcde56o5jlugh9uekgo84c6",
            "role": "user",
            "content": "Possible moves are: g1h3,g1f3,b1c3,b1a3,h2h3,g2g3,f2f3,e2e3,d2d3,c2c3,b2b3,a2a3,h2h4,g2g4,f2f4,e2e4,d2d4,c2c4,b2b4,a2a4",
        },
        {
            "tool_calls": [
                {
                    "id": "call_p1fla56o5jlugh9uekgo84c6",
                    "function": {"arguments": "{}", "name": "get_legal_moves"},
                    "type": "function",
                }
            ],
            "content": None,
            "role": "assistant",
        },
        {
            "tool_call_id": "call_p1fla56o5jlugh9uekgo84c6",
            "role": "user",
            "content": "Possible moves are: g1h3,g1f3,b1c3,b1a3,h2h3,g2g3,f2f3,e2e3,d2d3,c2c3,b2b3,a2a3,h2h4,g2g4,f2f4,e2e4,d2d4,c2c4,b2b4,a2a4",
        },
        {
            "tool_calls": [
                {
                    "id": "call_lcow1j0ehuhrcr3aakdmd9ju",
                    "function": {"arguments": '{"move":"g1f3"}', "name": "make_move"},
                    "type": "function",
                }
            ],
            "content": None,
            "role": "assistant",
        },
        {"tool_call_id": "call_lcow1j0ehuhrcr3aakdmd9ju", "role": "user", "content": "Moved knight (♘) from g1 to f3."},
    ]

    # Test if no tools
    no_tools = []
    all_tools = [
        {
            "type": "function",
            "function": {
                "description": "Call this tool to make a move after you have the list of legal moves.",
                "name": "make_move",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "move": {"type": "string", "description": "A move in UCI format. (e.g. e2e4 or e7e5 or e7e8q)"}
                    },
                    "required": ["move"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "description": "Call this tool to make a move after you have the list of legal moves.",
                "name": "get_legal_moves",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]

    # Should not hide for any hide_tools value
    assert not should_hide_tools(messages, no_tools, "if_all_run")
    assert not should_hide_tools(messages, no_tools, "if_any_run")
    assert not should_hide_tools(messages, no_tools, "never")

    # Has run tools but never hide, should be false
    assert not should_hide_tools(messages, all_tools, "never")

    # Has run tools, should be true if all or any
    assert should_hide_tools(messages, all_tools, "if_all_run")
    assert should_hide_tools(messages, all_tools, "if_any_run")

    # Hasn't run any tools, should be false for all
    assert not should_hide_tools(no_tools_called_messages, all_tools, "if_all_run")
    assert not should_hide_tools(no_tools_called_messages, all_tools, "if_any_run")
    assert not should_hide_tools(no_tools_called_messages, all_tools, "never")

    # Has run one of the two tools, should be true only for 'if_any_run'
    assert not should_hide_tools(one_tool_called_messages, all_tools, "if_all_run")
    assert should_hide_tools(one_tool_called_messages, all_tools, "if_any_run")
    assert not should_hide_tools(one_tool_called_messages, all_tools, "never")

    # Parameter validation
    with pytest.raises(TypeError):
        assert not should_hide_tools(one_tool_called_messages, all_tools, "not_a_valid_value")


@run_for_optional_imports(["openai"], "openai")
def test_validate_openai_client():
    """Test the validate_openai_client function."""

    # Test with None client
    with pytest.raises(ValueError, match="OpenAI client cannot be None"):
        validate_openai_client(None)

    # Test with custom client type
    with pytest.raises(ValueError, match="CustomAPI client cannot be None"):
        validate_openai_client(None, "CustomAPI")

    # Test with client missing api_key attribute
    mock_client = MagicMock()
    del mock_client.api_key  # Remove api_key attribute
    with pytest.raises(ValueError, match="OpenAI client must have a valid API key"):
        validate_openai_client(mock_client)

    # Test with client with None api_key
    mock_client = MagicMock()
    mock_client.api_key = None
    with pytest.raises(ValueError, match="OpenAI client must have a valid API key"):
        validate_openai_client(mock_client)

    # Test with client with empty api_key
    mock_client = MagicMock()
    mock_client.api_key = ""
    with pytest.raises(ValueError, match="OpenAI client must have a valid API key"):
        validate_openai_client(mock_client)

    # Test with valid client (no _validate method)
    mock_client = MagicMock()
    mock_client.api_key = "valid_key"
    # Should not raise any exception
    validate_openai_client(mock_client)

    # Test with valid client that has _validate method
    mock_client = MagicMock()
    mock_client.api_key = "valid_key"
    mock_client._validate = MagicMock()
    # Should not raise any exception and call _validate
    validate_openai_client(mock_client)
    mock_client._validate.assert_called_once()

    # Test with client that has _validate method that raises
    mock_client = MagicMock()
    mock_client.api_key = "valid_key"
    mock_client._validate = MagicMock(side_effect=Exception("Validation failed"))
    with pytest.raises(ValueError, match="Invalid OpenAI client configuration: Validation failed"):
        validate_openai_client(mock_client)


@run_for_optional_imports(["openai"], "openai")
def test_standardize_api_error():
    """Test the standardize_api_error function."""

    # Test API key errors
    api_key_error = Exception("Invalid API key provided")
    result = standardize_api_error(api_key_error, "TestAPI", "test-model")
    assert "Invalid API key for TestAPI" in str(result)

    unauthorized_error = Exception("Unauthorized access")
    result = standardize_api_error(unauthorized_error, "TestAPI", "test-model")
    assert "Invalid API key for TestAPI" in str(result)

    # Test model not found errors
    model_error = Exception("Model gpt-5 not found")
    result = standardize_api_error(model_error, "TestAPI", "gpt-5")
    assert "Model 'gpt-5' not found or not supported by TestAPI" in str(result)

    model_error2 = Exception("Model does not exist")
    result = standardize_api_error(model_error2, "TestAPI", "test-model")
    assert "Model 'test-model' not found or not supported by TestAPI" in str(result)

    # Test rate limit errors
    rate_limit_error = Exception("Rate limit exceeded")
    result = standardize_api_error(rate_limit_error, "TestAPI", "test-model")
    assert "API rate limit exceeded for TestAPI" in str(result)

    # Test quota errors
    quota_error = Exception("Insufficient quota available")
    result = standardize_api_error(quota_error, "TestAPI", "test-model")
    assert "Insufficient API quota for TestAPI" in str(result)

    # Test timeout errors
    timeout_error = Exception("Request timeout occurred")
    result = standardize_api_error(timeout_error, "TestAPI", "test-model")
    assert "API request timeout for TestAPI" in str(result)

    # Test connection errors
    connection_error = Exception("Connection error to server")
    result = standardize_api_error(connection_error, "TestAPI", "test-model")
    assert "Network connection error for TestAPI" in str(result)

    network_error = Exception("Network is unreachable")
    result = standardize_api_error(network_error, "TestAPI", "test-model")
    assert "Network connection error for TestAPI" in str(result)

    # Test generic error
    generic_error = Exception("Some unexpected error")
    result = standardize_api_error(generic_error, "TestAPI", "test-model")
    assert "TestAPI error: Some unexpected error" in str(result)

    # Test default parameters
    generic_error = Exception("Default test")
    result = standardize_api_error(generic_error)
    assert "API error: Default test" in str(result)

    # Test case sensitivity (errors should be detected regardless of case)
    mixed_case_error = Exception("API KEY is Invalid")
    result = standardize_api_error(mixed_case_error, "TestAPI", "test-model")
    assert "Invalid API key for TestAPI" in str(result)


def test_validate_parameter_edge_cases():
    """Test additional edge cases for validate_parameter to improve coverage."""

    # Test when param_value is None and allow_None is False (line 72-73)
    params = {"test_param": None}
    result = validate_parameter(params, "test_param", str, False, "default", None, None)
    assert result == "default"

    # Test tuple type formatting vs single type (line 76-80)
    params = {"test_param": 123}
    # Single type - should format as just the type name
    result = validate_parameter(params, "test_param", str, False, "default", None, None)
    assert result == "default"

    # Test numerical bounds with both lower and upper bounds (lines 88-96)
    params = {"test_param": 5}
    result = validate_parameter(params, "test_param", int, True, 10, (1, 3), None)
    assert result == 10  # Should use default due to being out of bounds

    # Test numerical bounds with only upper bound (line 90-94)
    params = {"test_param": 10}
    result = validate_parameter(params, "test_param", int, False, 5, (None, 8), None)
    assert result == 5  # Should use default due to exceeding upper bound

    # Test numerical bounds with allow_None (line 94-95)
    params = {"test_param": 10}
    result = validate_parameter(params, "test_param", int, True, None, (1, 5), None)
    assert result is None  # Should use default None

    # Test allowed_values with allow_None and param_value is None (line 99)
    params = {"test_param": None}
    result = validate_parameter(params, "test_param", str, True, "default", None, ["a", "b"])
    assert result is None  # None should be allowed

    # Test allowed_values when param not in allowed values (line 99-100)
    params = {"test_param": "c"}
    result = validate_parameter(params, "test_param", str, True, "default", None, ["a", "b"])
    assert result == "default"


def test_should_hide_tools_none_tools():
    """Test should_hide_tools with None tools parameter."""
    messages = [{"content": "test", "role": "user"}]

    # Test with None tools (line 130)
    result = should_hide_tools(messages, None, "if_all_run")
    assert result is False

    # Test with empty tools list (line 130)
    result = should_hide_tools(messages, [], "if_any_run")
    assert result is False


def test_validate_openai_client_edge_cases():
    """Test additional edge cases for validate_openai_client."""

    # Test client without _validate method but with api_key (lines 201-202)
    mock_client = MagicMock()
    mock_client.api_key = "valid_key"
    # Remove _validate method entirely
    if hasattr(mock_client, "_validate"):
        del mock_client._validate

    # Should pass without calling _validate
    validate_openai_client(mock_client, "TestClient")

    # Test client with _validate method that is not callable (line 201)
    mock_client = MagicMock()
    mock_client.api_key = "valid_key"
    mock_client._validate = "not_callable"  # Not a callable

    # Should pass without calling _validate since it's not callable
    validate_openai_client(mock_client, "TestClient")


def test_standardize_api_error_edge_cases():
    """Test additional edge cases for standardize_api_error."""

    # Test model error with both "model" and "not found" (line 236)
    model_error = Exception("The model xyz not found in our system")
    result = standardize_api_error(model_error, "TestAPI", "xyz")
    assert "Model 'xyz' not found or not supported by TestAPI" in str(result)

    # Test model error with both "model" and "does not exist" (line 236)
    model_error2 = Exception("model abc does not exist")
    result = standardize_api_error(model_error2, "TestAPI", "abc")
    assert "Model 'abc' not found or not supported by TestAPI" in str(result)

    # Test insufficient quota error (line 240-241)
    quota_error = Exception("insufficient quota remaining")
    result = standardize_api_error(quota_error, "TestAPI", "test-model")
    assert "Insufficient API quota for TestAPI" in str(result)

    # Test connection error (line 244-245)
    connection_error = Exception("connection refused by server")
    result = standardize_api_error(connection_error, "TestAPI", "test-model")
    assert "Network connection error for TestAPI" in str(result)

    # Test network error (line 244-245)
    network_error = Exception("network connection failed")
    result = standardize_api_error(network_error, "TestAPI", "test-model")
    assert "Network connection error for TestAPI" in str(result)


if __name__ == "__main__":
    # test_validate_parameter()
    test_should_hide_tools()
    test_validate_openai_client()
    test_standardize_api_error()
    test_validate_parameter_edge_cases()
    test_should_hide_tools_none_tools()
    test_validate_openai_client_edge_cases()
    test_standardize_api_error_edge_cases()
