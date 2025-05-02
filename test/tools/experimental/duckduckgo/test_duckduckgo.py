# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from autogen import AssistantAgent
from autogen.import_utils import run_for_optional_imports
from autogen.tools.experimental.duckduckgo import DuckDuckGoSearchTool

from ....conftest import Credentials


class TestDuckDuckGoSearchTool:
    """
    Test suite for the DuckDuckGoSearchTool class.
    """

    @pytest.fixture
    def mock_response(self) -> list[dict[str, Any]]:
        """
        Provide a mock response fixture for testing.
        """
        return [
            {
                "title": "Test Result",
                "href": "https://example.com",  # Use 'href' as per duckduckgo-search
                "body": "This is a test snippet.",  # Use 'body' as per duckduckgo-search
            }
        ]

    def test_initialization(self) -> None:
        """
        Test the initialization of DuckDuckGoSearchTool.
        """
        tool = DuckDuckGoSearchTool()
        assert tool.name == "duckduckgo_search"
        assert "Use the DuckDuckGo Search API to perform a search" in tool.description

    def test_schema_validation(self) -> None:
        """
        Test the validation of the tool's JSON schema.
        """
        tool = DuckDuckGoSearchTool()
        expected_schema = {
            "function": {
                "name": "duckduckgo_search",
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."},
                        "num_results": {
                            "type": "integer",
                            "default": 5,
                            "description": "The number of results to return.",
                        },
                    },
                    "required": ["query"],
                },
            },
            "type": "function",
        }
        assert tool.tool_schema == expected_schema

    @patch("autogen.tools.experimental.duckduckgo.duckduckgo_search._execute_duckduckgo_query")
    def test_execute_query_success(self, mock_execute: Mock, mock_response: list[dict[str, Any]]) -> None:
        """
        Test successful execution of an API query.
        """
        mock_execute.return_value = mock_response

        tool = DuckDuckGoSearchTool()
        # DuckDuckGo tool doesn't take API key in the call
        result = tool(query="Test query")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["title"] == "Test Result"
        assert result[0]["link"] == "https://example.com"
        assert result[0]["snippet"] == "This is a test snippet."

        mock_execute.assert_called_once_with(
            query="Test query",
            num_results=5,  # Default num_results
        )

    @patch("autogen.tools.experimental.duckduckgo.duckduckgo_search._execute_duckduckgo_query")
    def test_search(self, mock_execute: Mock, mock_response: list[dict[str, Any]]) -> None:
        """
        Test the main search functionality.
        """
        mock_execute.return_value = mock_response
        tool = DuckDuckGoSearchTool()
        result = tool(query="Test query", num_results=3)  # Test with non-default num_results
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["title"] == "Test Result"
        assert result[0]["link"] == "https://example.com"
        assert result[0]["snippet"] == "This is a test snippet."
        mock_execute.assert_called_once_with(
            query="Test query",
            num_results=3,
        )

    def test_search_invalid_query(self) -> None:
        """
        Test that an invalid (empty) query string raises a Pydantic ValidationError.
        """
        tool = DuckDuckGoSearchTool()
        with pytest.raises(ValidationError) as exc_info:
            tool(query=None)  # type: ignore[arg-type]
        assert "Input should be a valid string" in str(exc_info.value)

    @run_for_optional_imports("openai", "openai")
    def test_agent_integration(self, credentials_gpt_4o_mini: Credentials) -> None:
        """
        Test integration with AssistantAgent.
        """
        search_tool = DuckDuckGoSearchTool()
        assistant = AssistantAgent(
            name="assistant",
            system_message="You are a helpful assistant. Use the duckduckgo_search tool when needed.",
            llm_config=credentials_gpt_4o_mini.llm_config,
        )
        search_tool.register_for_llm(assistant)
        # Mock the underlying query execution function
        with patch(
            "autogen.tools.experimental.duckduckgo.duckduckgo_search._execute_duckduckgo_query"
        ) as mock_execute_query:
            # Provide a mock return value for the search
            mock_execute_query.return_value = [
                {
                    "title": "Hurricane News",
                    "href": "https://example.com/hurricane",
                    "body": "Latest hurricane updates.",
                }
            ]
            response = assistant.run(
                message="Get me the latest news on hurricanes",
                tools=assistant.tools,
                max_turns=2,
                user_input=False,
            )
            response.process()  # Process the response to ensure tool calls are handled
            assert mock_execute_query.called  # Check if the mocked function was called
        assert isinstance(assistant.tools[0], DuckDuckGoSearchTool)
        assert assistant.tools[0].name == "duckduckgo_search"
