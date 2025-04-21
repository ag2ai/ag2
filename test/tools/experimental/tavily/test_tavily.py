# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from autogen import AssistantAgent
from autogen.import_utils import run_for_optional_imports
from autogen.tools.experimental.tavily import TavilySearchTool

from ....conftest import Credentials


class TestTavilySearchTool:
    """
    Test suite for the TavilySearchTool class.
    """

    @pytest.fixture
    def mock_response(self) -> dict[str, Any]:
        """
        Provide a mock response fixture for testing.
        """
        return {
            "results": [
                {
                    "title": "Test Result",
                    "url": "https://example.com",
                    "content": "This is a test snippet.",
                }
            ]
        }

    @pytest.mark.parametrize("use_internal_auth", [True, False])
    def test_initialization(self, use_internal_auth: bool) -> None:
        """
        Test the initialization of TavilySearchTool.
        """
        if use_internal_auth:
            with pytest.raises(ValueError) as exc_info:
                TavilySearchTool(tavily_api_key=None)
            assert "tavily_api_key must be provided" in str(exc_info.value)
        else:
            tool = TavilySearchTool(tavily_api_key="valid_key")
            assert tool.name == "tavily_search"
            assert "Use the Tavily Search API to perform a search." in tool.description
            assert tool.tavily_api_key == "valid_key"

    def test_tool_schema(self) -> None:
        """
        Test the validation of the tool's JSON schema.
        """
        tool = TavilySearchTool(tavily_api_key="test_key")
        expected_schema = {
            "function": {
                "name": "tavily_search",
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."},
                        "search_depth": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "null"},
                            ],
                            "default": "basic",
                            "description": "Either 'advanced' or 'basic'",
                        },
                        "include_answer": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "null"},
                            ],
                            "default": "basic",
                            "description": "Either 'advanced' or 'basic'",
                        },
                        "include_raw_content": {
                            "anyOf": [
                                {"type": "boolean"},
                                {"type": "null"},
                            ],
                            "default": False,
                            "description": "Include the raw contents",
                        },
                        "include_domains": {
                            "anyOf": [
                                {"type": "array", "items": {"type": "string"}},
                                {"type": "null"},
                            ],
                            "default": [],
                            "description": "Specific web domains to search",
                        },
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

    @pytest.mark.parametrize(
        ("search_params", "expected_error"),
        [
            ({"tavily_api_key": None}, "tavily_api_key must be provided"),
        ],
    )
    def test_parameter_validation(self, search_params: dict[str, Any], expected_error: str) -> None:
        """
        Test validation of tool parameters.
        """
        with pytest.raises(ValueError) as exc_info:
            TavilySearchTool(**search_params)
        assert expected_error in str(exc_info.value)

    @patch("autogen.tools.experimental.tavily.tavily_search._execute_tavily_query")
    def test_execute_query_success(self, mock_execute: Mock, mock_response: dict[str, Any]) -> None:
        """
        Test successful execution of an API query.
        """
        mock_execute.return_value = mock_response

        tool = TavilySearchTool(tavily_api_key="valid_test_key")
        result = tool(query="Test query", tavily_api_key="valid_test_key")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["title"] == "Test Result"
        assert result[0]["link"] == "https://example.com"
        assert result[0]["snippet"] == "This is a test snippet."

        mock_execute.assert_called_once_with(
            query="Test query",
            tavily_api_key="valid_test_key",
            search_depth="basic",
            topic="general",
            include_answer="basic",
            include_raw_content=False,
            include_domains=[],
            num_results=5,
        )

    @patch("autogen.tools.experimental.tavily.tavily_search._execute_tavily_query")
    def test_search(self, mock_execute: Mock, mock_response: dict[str, Any]) -> None:
        """
        Test the main search functionality.
        """
        mock_execute.return_value = mock_response
        tool = TavilySearchTool(tavily_api_key="test_key")
        result = tool(query="Test query", tavily_api_key="test_key")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["title"] == "Test Result"
        assert result[0]["link"] == "https://example.com"
        assert result[0]["snippet"] == "This is a test snippet."
        mock_execute.assert_called_once_with(
            query="Test query",
            tavily_api_key="test_key",
            search_depth="basic",
            topic="general",
            include_answer="basic",
            include_raw_content=False,
            include_domains=[],
            num_results=5,
        )

    def test_search_invalid_query(self) -> None:
        """
        Test that an invalid (empty) query string raises a Pydantic ValidationError.
        """
        tool = TavilySearchTool(tavily_api_key="test_key")
        with pytest.raises(ValidationError) as exc_info:
            tool(query=None, tavily_api_key="test_key")  # type: ignore[arg-type]
        assert "Input should be a valid string" in str(exc_info.value)

    @run_for_optional_imports("openai", "openai")
    def test_agent_integration(self, credentials_gpt_4o_mini: Credentials) -> None:
        """
        Test integration with AssistantAgent.
        """
        search_tool = TavilySearchTool(tavily_api_key="test_key")
        assistant = AssistantAgent(
            name="assistant",
            system_message="You are a helpful assistant. Use the tavily_search tool when needed.",
            llm_config=credentials_gpt_4o_mini.llm_config,
        )
        search_tool.register_for_llm(assistant)
        with patch("autogen.tools.experimental.tavily.tavily_search._execute_tavily_query") as mock_execute_query:
            response = assistant.run(
                message="Get me the latest news on hurricanes",
                tools=assistant.tools,
                max_turns=2,
                user_input=False,
            )
            response.process()
            assert mock_execute_query.called
        assert isinstance(assistant.tools[0], TavilySearchTool)
        assert assistant.tools[0].name == "tavily_search"
