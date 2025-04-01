"""
Test suite for the PerplexitySearchTool class.
Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
SPDX-License-Identifier: Apache-2.0

This module contains unit tests that verify the functionality of the Perplexity AI
search integration, including authentication, query execution, and response handling.
"""

import json
from typing import Any
from unittest.mock import Mock, patch

import pytest

from autogen import AssistantAgent
from autogen.import_utils import run_for_optional_imports
from autogen.tools.experimental.perplexity import PerplexitySearchTool
from autogen.tools.experimental.perplexity.perplexity_search import PerplexityChatCompletionResponse, SearchResponse

from ....conftest import Credentials


class TestPerplexitySearchTool:
    """
    Test suite for the PerplexitySearchTool class.
    """

    @pytest.fixture
    def mock_response(self) -> dict[str, Any]:
        """
        Provide a mock response fixture for testing.
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
        """
        tool = PerplexitySearchTool(api_key="test_key")
        expected_schema = {
            "function": {
                "name": "perplexity-search",
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "query"}},
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
        """
        with pytest.raises(ValueError) as exc_info:
            PerplexitySearchTool(**search_params)
        assert expected_error in str(exc_info.value)

    @patch("requests.request")
    def test_execute_query_success(self, mock_request: Mock, mock_response: dict[str, Any]) -> None:
        """
        Test successful execution of an API query.
        """
        mock_request.return_value = Mock(
            status_code=200, json=Mock(return_value=mock_response), raise_for_status=Mock()
        )

        tool = PerplexitySearchTool(api_key="valid_test_key")

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
        assert isinstance(response, PerplexityChatCompletionResponse)
        assert response.choices[0].message.content == "Test response content"
        assert response.citations == mock_response["citations"]

        mock_request.assert_called_once_with(
            "POST",
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": "Bearer valid_test_key", "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )

    @patch("requests.request")
    def test_execute_query_error(self, mock_request: Mock) -> None:
        """
        Test error handling during query execution (invalid JSON response).
        """
        # Create a mock response that raises JSONDecodeError when calling .json()
        mock_response_obj = Mock(status_code=401, text="<html>Unauthorized</html>")
        mock_response_obj.raise_for_status.side_effect = None  # No HTTP error is raised here
        mock_response_obj.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_request.return_value = mock_response_obj

        tool = PerplexitySearchTool(api_key="test_key")

        with pytest.raises(RuntimeError) as exc_info:
            tool._execute_query({})
        assert "Invalid JSON response received" in str(exc_info.value)

    @patch.object(PerplexitySearchTool, "_execute_query")
    def test_search(self, mock_execute: Mock, mock_response: dict[str, Any]) -> None:
        """
        Test the main search functionality.
        """
        mock_execute.return_value = PerplexityChatCompletionResponse(**mock_response)
        tool = PerplexitySearchTool(api_key="test_key")
        result = tool.search("Test query")
        assert isinstance(result, SearchResponse)
        assert result.content == "Test response content"
        assert result.citations == mock_response["citations"]
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

    def test_search_invalid_query(self) -> None:
        """
        Test that an invalid (empty) query string raises a ValueError.
        """
        tool = PerplexitySearchTool(api_key="test_key")
        with pytest.raises(ValueError) as exc_info:
            tool.search("")
        assert "A valid non-empty query string must be provided" in str(exc_info.value)

    def test_search_exception_case(self) -> None:
        """
        Test error handling in the search method when an exception occurs.
        """
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
