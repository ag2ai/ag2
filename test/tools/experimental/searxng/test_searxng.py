# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from autogen import AssistantAgent
from autogen.tools.experimental.searxng import SearxngSearchTool
from test.conftest import Credentials


class TestSearxngSearchTool:
    """
    Test suite for the SearxngSearchTool class.
    """

    @pytest.fixture
    def mock_response(self) -> list[dict[str, str]]:
        return [{"title": "Test Result", "url": "https://example.com", "content": "This is a test snippet."}]

    def test_initialization(self) -> None:
        tool = SearxngSearchTool()
        assert tool.name == "searxng_search"
        assert "Use the SearxNG API to perform a search" in tool.description

    def test_schema_validation(self) -> None:
        tool = SearxngSearchTool()
        expected_schema = {
            "function": {
                "name": "searxng_search",
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."},
                        "max_results": {
                            "type": "integer",
                            "default": 5,
                            "description": "The number of results to return.",
                        },
                        "categories": {
                            "anyOf": [
                                {
                                    "items": {"type": "string"},
                                    "type": "array",
                                },
                                {"type": "null"},
                            ],
                            "default": None,
                            "description": "List of categories to search in.",
                        },
                        "language": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "description": "Language code (e.g., 'en-US').",
                        },
                    },
                    "required": ["query"],
                },
            },
            "type": "function",
        }
        assert tool.tool_schema == expected_schema

    @patch("autogen.tools.experimental.searxng.searxng_search._execute_searxng_query")
    def test_execute_query_success(self, mock_execute: Any, mock_response: list[dict[str, str]]) -> None:
        mock_execute.return_value = mock_response
        tool = SearxngSearchTool()
        result = tool(query="Test query")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["title"] == "Test Result"
        assert result[0]["link"] == "https://example.com"
        assert result[0]["snippet"] == "This is a test snippet."
        mock_execute.assert_called_once_with(
            query="Test query",
            max_results=5,
            categories=None,
            language=None,
            base_url="https://searxng.site/search",
        )

    @patch("autogen.tools.experimental.searxng.searxng_search._execute_searxng_query")
    def test_search(self, mock_execute: Any, mock_response: list[dict[str, str]]) -> None:
        mock_execute.return_value = mock_response
        tool = SearxngSearchTool()
        result = tool(query="Test query", max_results=3)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["title"] == "Test Result"
        assert result[0]["link"] == "https://example.com"
        assert result[0]["snippet"] == "This is a test snippet."
        mock_execute.assert_called_once_with(
            query="Test query",
            max_results=3,
            categories=None,
            language=None,
            base_url="https://searxng.site/search",
        )

    def test_search_invalid_query(self) -> None:
        tool = SearxngSearchTool()
        with pytest.raises(ValidationError):
            tool(query=None)  # type: ignore[arg-type]

    @pytest.mark.skip("Integration test - requires live SearxNG instance")
    def test_integration_live(self) -> None:
        tool = SearxngSearchTool()
        results = tool(query="open source search engine", max_results=1)
        assert isinstance(results, list)
        assert len(results) >= 0

    @pytest.mark.skip("Integration test - requires live SearxNG instance and credentials")
    def test_agent_integration(self, credentials_gpt_4o_mini: Credentials) -> None:
        search_tool = SearxngSearchTool()
        assistant = AssistantAgent(
            name="assistant",
            system_message="You are a helpful assistant. Use the searxng_search tool when needed.",
            llm_config=credentials_gpt_4o_mini.llm_config,
        )
        search_tool.register_for_llm(assistant)
        response = assistant.run(
            message="Get me the latest news on open source search engines",
            tools=assistant.tools,
            max_turns=2,
            user_input=False,
        )
        response.process()
        assert isinstance(assistant.tools[0], SearxngSearchTool)
        assert assistant.tools[0].name == "searxng_search"
