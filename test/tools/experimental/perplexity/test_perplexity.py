"""
Test suite for the PerplexitySearchTool class.
Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
SPDX-License-Identifier: Apache-2.0"

This module contains unit tests that verify the functionality of the Perplexity AI
search integration, including authentication, query execution, and response handling.

Classes:
    TestPerplexitySearchTool: Test suite for PerplexitySearchTool functionality
"""

from json import JSONDecodeError
from typing import Any
from unittest.mock import Mock, patch

import pytest
import requests

from autogen import AssistantAgent
from autogen.import_utils import run_for_optional_imports
from autogen.tools.experimental.perplexity import PerplexitySearchTool
from autogen.tools.experimental.perplexity.perplexity_search import PerplexityChatCompletionResponse, SearchResponse

from ....conftest import Credentials


class TestPerplexitySearchTool:
    """
    Test suite for the PerplexitySearchTool class.

    This class contains comprehensive tests for validating the functionality
    of the PerplexitySearchTool, including initialization, schema validation,
    parameter validation, and API interactions.
    """

    @pytest.fixture
    def mock_response(self) -> dict[str, Any]:
        """
        Provide a mock response fixture for testing.

        Returns:
            dict[str, Any]: A dictionary containing a simulated Perplexity API response
        """
        return {
            "id": "test-id",
            "model": "sonar",
            "created": 1234567890,
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60, "search_context_size": "high"},
            "citations": ["https://example.com/source1", "https://example.com/source2"],
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Test response content"},
                }
            ],
        }

    @pytest.mark.parametrize("use_internal_auth", [True, False])
    def test_initialization(self, use_internal_auth: bool) -> None:
        """
        Test the initialization of PerplexitySearchTool.

        Args:
            use_internal_auth (bool): Flag to test with/without internal authentication

        Raises:
            ValueError: When API key is missing in internal auth mode
        """
        if use_internal_auth:
            with pytest.raises(ValueError) as exc_info:
                PerplexitySearchTool(api_key=None)
            assert "Perplexity API key is missing" in str(exc_info.value)
        else:
            tool = PerplexitySearchTool(api_key="valid_key")
            assert tool.name == "perplexity-search"
            assert "Perplexity AI search tool for web search" in tool.description
            assert tool.model == "sonar"
            assert tool.max_tokens == 1000

    def test_tool_schema(self) -> None:
        """
        Test the validation of the tool's JSON schema.

        Verifies:
            - Correct function name
            - Proper description inclusion
            - Parameter structure and requirements
            - Schema type definition
            - Query parameter specification

        Raises:
            AssertionError: When the schema does not match expected values
        """
        tool = PerplexitySearchTool(api_key="test_key")
        expected_schema = {
            "function": {
                "name": "perplexity-search",
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The search query."}},
                    "required": ["query"],
                },
            },
            "type": "function",
        }
        assert tool.tool_schema == expected_schema

    @pytest.mark.parametrize(
        ("search_params", "expected_error"),
        [
            ({"api_key": "valid", "max_tokens": -100}, "max_tokens must be positive"),
            ({"api_key": "valid", "search_domain_filter": "invalid"}, "search_domain_filter must be a list"),
            ({"api_key": "valid", "model": ""}, "model cannot be empty"),
            ({"api_key": None}, "Perplexity API key is missing"),
        ],
    )
    def test_parameter_validation(self, search_params: dict[str, Any], expected_error: str) -> None:
        """
        Test validation of tool parameters.

        Args:
            search_params (dict[str, Any]): Input parameters to test
            expected_error (str): Expected error message

        Raises:
            ValueError: When invalid parameters are provided
        """
        with pytest.raises(ValueError) as exc_info:
            PerplexitySearchTool(**search_params)
        assert expected_error in str(exc_info.value)

    @patch("requests.request")
    def test_execute_query_success(self, mock_request: Mock, mock_response: dict[str, Any]) -> None:
        """
        Test successful execution of API query.

        Args:
            mock_request (Mock): Mocked requests.request function
            mock_response (dict[str, Any]): Fixture containing mock API response

        Tests:
            - API request formatting
            - Response parsing
            - Header validation
            - Payload structure
        """
        mock_request.return_value = Mock(
            status_code=200, json=Mock(return_value=mock_response), raise_for_status=Mock()
        )

        tool = PerplexitySearchTool(api_key="valid_test_key")

        # Execute the query
        payload = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": "Be precise and concise."},
                {"role": "user", "content": "Test query"},
            ],
            "max_tokens": 1000,
            "search_domain_filter": None,
            "web_search_options": {"search_context_size": "high"},
        }

        response = tool._execute_query(payload)

        # Validate response
        assert isinstance(response, PerplexityChatCompletionResponse)
        assert response.choices[0].message.content == "Test response content"
        assert response.citations == mock_response["citations"]

        # Verify API call parameters
        mock_request.assert_called_once_with(
            "POST",
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": "Bearer valid_test_key", "Content-Type": "application/json"},
            json=payload,
        )

    @patch("requests.post")
    def test_execute_query_error(self, mock_post: Mock) -> None:
        """
        Test error handling during query execution.

        Args:
            mock_post (Mock): Mocked requests.post function

        Tests:
            - Invalid JSON response handling
            - Error status code handling
            - Exception propagation
        """
        mock_post.return_value = Mock(
            status_code=401,
            text="<html>Unauthorized</html>",
            json=Mock(side_effect=JSONDecodeError("Expecting value", "", 0)),
        )

        tool = PerplexitySearchTool(api_key="test_key")

        with pytest.raises(requests.exceptions.JSONDecodeError) as exc_info:
            tool._execute_query({})
        assert "Expecting value" in str(exc_info.value)

    @patch.object(PerplexitySearchTool, "_execute_query")
    def test_search(self, mock_execute: Mock, mock_response: dict[str, Any]) -> None:
        """
        Test the main search functionality.

        Args:
            mock_execute (Mock): Mocked _execute_query method
            mock_response (dict[str, Any]): Fixture containing mock API response

        Tests:
            - Search response formatting
            - Content extraction
            - Citation handling
            - Query parameter validation
        """
        mock_execute.return_value = PerplexityChatCompletionResponse(**mock_response)
        tool = PerplexitySearchTool(api_key="test_key")
        result = tool.search("Test query")
        assert isinstance(result, SearchResponse)
        assert result.content == "Test response content"
        assert result.citations is not None
        assert result.citations[0] == "https://example.com/source1"
        assert result.citations[1] == "https://example.com/source2"
        assert result.error is None
        mock_execute.assert_called_once_with({
            "model": "sonar",
            "messages": [
                {"role": "system", "content": "Be precise and concise."},
                {"role": "user", "content": "Test query"},
            ],
            "max_tokens": 1000,
            "search_domain_filter": None,
            "web_search_options": {"search_context_size": "high"},
        })

    def test_search_exception_case(self) -> None:
        """
        Test the error handling of the search method in PerplexitySearchTool.

        Tests:
          - 'content' is set to None.
          - 'citations' is set to None.
          - 'error' contains an error message with the expected exception details.
        """
        # Patch _execute_query to simulate an exception
        with patch.object(PerplexitySearchTool, "_execute_query", side_effect=Exception("Test exception")):
            tool = PerplexitySearchTool(api_key="test_key")
            response: SearchResponse = tool.search("Test query")
            assert response.content is None
            assert response.citations is None
            assert response.error is not None
            assert "Test exception" in response.error

    @run_for_optional_imports("openai", "openai")
    def test_agent_integration(self, credentials_gpt_4o_mini: Credentials) -> None:
        """
        Test integration with AssistantAgent.

        Args:
            credentials_gpt_4o_mini (Credentials): Test credentials fixture

        Tests:
            - Tool registration with agent
            - Configuration verification
            - Tool accessibility in agent
        """
        search_tool = PerplexitySearchTool(api_key="test_key")
        assistant = AssistantAgent(
            name="assistant",
            system_message="You are a helpful assistant. Use the perplexity-search tool when needed.",
            llm_config=credentials_gpt_4o_mini.llm_config,
        )
        search_tool.register_for_llm(assistant)
        assert isinstance(assistant.tools[0], PerplexitySearchTool)
        assert assistant.tools[0].name == "perplexity-search"
