# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from autogen import AssistantAgent
from autogen.tools.experimental.firecrawl import FirecrawlTool
from test.conftest import Credentials


class TestFirecrawlTool:
    """
    Test suite for the FirecrawlTool class.
    """

    @pytest.fixture
    def mock_scrape_response(self) -> dict[str, Any]:
        return {
            "markdown": "# Test Page\nThis is test content.",
            "html": "<h1>Test Page</h1><p>This is test content.</p>",
            "metadata": {
                "title": "Test Page",
                "sourceURL": "https://example.com",
            },
        }

    @pytest.fixture
    def mock_crawl_response(self) -> dict[str, Any]:
        return {
            "data": [
                {
                    "markdown": "# Test Page 1\nContent 1",
                    "html": "<h1>Test Page 1</h1><p>Content 1</p>",
                    "metadata": {
                        "title": "Test Page 1",
                        "sourceURL": "https://example.com/page1",
                    },
                },
                {
                    "markdown": "# Test Page 2\nContent 2",
                    "html": "<h1>Test Page 2</h1><p>Content 2</p>",
                    "metadata": {
                        "title": "Test Page 2",
                        "sourceURL": "https://example.com/page2",
                    },
                },
            ],
        }

    @pytest.fixture
    def mock_map_response(self) -> dict[str, Any]:
        return {
            "links": [
                "https://example.com",
                "https://example.com/about",
                "https://example.com/contact",
            ],
        }

    def test_initialization(self) -> None:
        tool = FirecrawlTool(firecrawl_api_key="test-key")
        assert tool.name == "firecrawl_scrape"
        assert "Use the Firecrawl API to scrape content from a single URL" in tool.description

    def test_initialization_with_api_url(self) -> None:
        tool = FirecrawlTool(firecrawl_api_key="test-key", firecrawl_api_url="https://custom.firecrawl.com")
        assert tool.firecrawl_api_key == "test-key"
        assert tool.firecrawl_api_url == "https://custom.firecrawl.com"

    def test_initialization_no_api_key(self) -> None:
        with pytest.raises(ValueError, match="firecrawl_api_key must be provided"):
            FirecrawlTool()

    def test_schema_validation(self) -> None:
        tool = FirecrawlTool(firecrawl_api_key="test-key")
        expected_schema = {
            "function": {
                "name": "firecrawl_scrape",
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL to scrape."},
                        "formats": {
                            "anyOf": [
                                {
                                    "items": {"type": "string"},
                                    "type": "array",
                                },
                                {"type": "null"},
                            ],
                            "default": None,
                            "description": "Output formats (e.g., ['markdown', 'html'])",
                        },
                        "include_tags": {
                            "anyOf": [
                                {
                                    "items": {"type": "string"},
                                    "type": "array",
                                },
                                {"type": "null"},
                            ],
                            "default": None,
                            "description": "HTML tags to include",
                        },
                        "exclude_tags": {
                            "anyOf": [
                                {
                                    "items": {"type": "string"},
                                    "type": "array",
                                },
                                {"type": "null"},
                            ],
                            "default": None,
                            "description": "HTML tags to exclude",
                        },
                        "headers": {
                            "anyOf": [
                                {
                                    "additionalProperties": {"type": "string"},
                                    "type": "object",
                                },
                                {"type": "null"},
                            ],
                            "default": None,
                            "description": "HTTP headers to use",
                        },
                        "wait_for": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "description": "Time to wait for page load in milliseconds",
                        },
                        "timeout": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "description": "Request timeout in milliseconds",
                        },
                    },
                    "required": ["url"],
                },
            },
            "type": "function",
        }
        assert tool.tool_schema == expected_schema

    @patch("autogen.tools.experimental.firecrawl.firecrawl_tool._execute_firecrawl_scrape")
    def test_scrape_success(self, mock_execute: Any, mock_scrape_response: dict[str, Any]) -> None:
        mock_execute.return_value = mock_scrape_response
        tool = FirecrawlTool(firecrawl_api_key="test-key")
        results = tool(url="https://example.com")

        assert isinstance(results, list)
        assert len(results) == 1
        result = results[0]
        assert result["title"] == "Test Page"
        assert result["url"] == "https://example.com"
        assert result["content"] == "# Test Page\nThis is test content."
        assert "metadata" in result

    @patch("autogen.tools.experimental.firecrawl.firecrawl_tool._execute_firecrawl_scrape")
    def test_scrape_with_options(self, mock_execute: Any, mock_scrape_response: dict[str, Any]) -> None:
        mock_execute.return_value = mock_scrape_response
        tool = FirecrawlTool(firecrawl_api_key="test-key")
        results = tool(
            url="https://example.com",
            formats=["markdown", "html"],
            include_tags=["h1", "p"],
            exclude_tags=["script"],
            headers={"User-Agent": "test"},
            wait_for=1000,
            timeout=5000,
        )

        assert isinstance(results, list)
        assert len(results) == 1
        # Verify that the mock was called with the right parameters
        mock_execute.assert_called_once()

    @patch("autogen.tools.experimental.firecrawl.firecrawl_tool._execute_firecrawl_crawl")
    def test_crawl_success(self, mock_execute: Any, mock_crawl_response: dict[str, Any]) -> None:
        mock_execute.return_value = mock_crawl_response
        tool = FirecrawlTool(firecrawl_api_key="test-key")
        results = tool.crawl(url="https://example.com", limit=2, firecrawl_api_key="test-key", firecrawl_api_url=None)

        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]["title"] == "Test Page 1"
        assert results[1]["title"] == "Test Page 2"

    @patch("autogen.tools.experimental.firecrawl.firecrawl_tool._execute_firecrawl_map")
    def test_map_success(self, mock_execute: Any, mock_map_response: dict[str, Any]) -> None:
        mock_execute.return_value = mock_map_response
        tool = FirecrawlTool(firecrawl_api_key="test-key")
        results = tool.map(url="https://example.com", firecrawl_api_key="test-key", firecrawl_api_url=None)

        assert isinstance(results, list)
        assert len(results) == 3
        assert results[0]["url"] == "https://example.com"
        assert results[1]["url"] == "https://example.com/about"
        assert results[2]["url"] == "https://example.com/contact"

    @patch("autogen.tools.experimental.firecrawl.firecrawl_tool._execute_firecrawl_scrape")
    def test_scrape_with_custom_api_url(self, mock_execute: Any, mock_scrape_response: dict[str, Any]) -> None:
        mock_execute.return_value = mock_scrape_response

        tool = FirecrawlTool(firecrawl_api_key="test-key", firecrawl_api_url="https://custom.firecrawl.com")
        result = tool(url="https://example.com")

        # Verify the execution function was called with the custom API URL
        mock_execute.assert_called_once_with(
            url="https://example.com",
            firecrawl_api_key="test-key",
            firecrawl_api_url="https://custom.firecrawl.com",
            formats=None,
            include_tags=None,
            exclude_tags=None,
            headers=None,
            wait_for=None,
            timeout=None,
        )

        assert result == [
            {
                "title": "Test Page",
                "url": "https://example.com",
                "content": "# Test Page\nThis is test content.",
                "metadata": {
                    "title": "Test Page",
                    "sourceURL": "https://example.com",
                },
            }
        ]

    @patch("autogen.tools.experimental.firecrawl.firecrawl_tool._execute_firecrawl_crawl")
    def test_crawl_with_custom_api_url(self, mock_execute: Any, mock_crawl_response: dict[str, Any]) -> None:
        mock_execute.return_value = mock_crawl_response

        tool = FirecrawlTool(firecrawl_api_key="test-key", firecrawl_api_url="https://custom.firecrawl.com")
        result = tool.crawl(
            url="https://example.com",
            limit=2,
            firecrawl_api_key="test-key",
            firecrawl_api_url="https://custom.firecrawl.com",
        )

        # Verify the execution function was called with the custom API URL
        mock_execute.assert_called_once_with(
            url="https://example.com",
            firecrawl_api_key="test-key",
            firecrawl_api_url="https://custom.firecrawl.com",
            limit=2,
            formats=None,
            include_paths=None,
            exclude_paths=None,
            max_depth=None,
            allow_backward_crawling=False,
            allow_external_content_links=False,
        )

        expected_result = [
            {
                "title": "Test Page 1",
                "url": "https://example.com/page1",
                "content": "# Test Page 1\nContent 1",
                "metadata": {
                    "title": "Test Page 1",
                    "sourceURL": "https://example.com/page1",
                },
            },
            {
                "title": "Test Page 2",
                "url": "https://example.com/page2",
                "content": "# Test Page 2\nContent 2",
                "metadata": {
                    "title": "Test Page 2",
                    "sourceURL": "https://example.com/page2",
                },
            },
        ]
        assert result == expected_result

    @patch("autogen.tools.experimental.firecrawl.firecrawl_tool._execute_firecrawl_map")
    def test_map_with_custom_api_url(self, mock_execute: Any, mock_map_response: dict[str, Any]) -> None:
        mock_execute.return_value = mock_map_response

        tool = FirecrawlTool(firecrawl_api_key="test-key", firecrawl_api_url="https://custom.firecrawl.com")
        result = tool.map(
            url="https://example.com", firecrawl_api_key="test-key", firecrawl_api_url="https://custom.firecrawl.com"
        )

        # Verify the execution function was called with the custom API URL
        mock_execute.assert_called_once_with(
            url="https://example.com",
            firecrawl_api_key="test-key",
            firecrawl_api_url="https://custom.firecrawl.com",
            search=None,
            ignore_sitemap=False,
            include_subdomains=False,
            limit=5000,
        )

        expected_result = [
            {"url": "https://example.com", "title": "", "content": ""},
            {"url": "https://example.com/about", "title": "", "content": ""},
            {"url": "https://example.com/contact", "title": "", "content": ""},
        ]
        assert result == expected_result

    def test_search_invalid_query(self) -> None:
        tool = FirecrawlTool(firecrawl_api_key="test-key")
        with pytest.raises(ValidationError):
            tool(url=None)  # type: ignore[arg-type]

    @pytest.mark.skip("Integration test - requires live Firecrawl instance")
    def test_integration_live_scrape(self) -> None:
        tool = FirecrawlTool(firecrawl_api_key="test-key")
        results = tool(url="https://example.com")
        assert isinstance(results, list)
        assert len(results) >= 0

    @pytest.mark.skip("Integration test - requires live Firecrawl instance and credentials")
    def test_agent_integration(self, credentials_gpt_4o_mini: Credentials) -> None:
        firecrawl_tool = FirecrawlTool(firecrawl_api_key="test-key")
        assistant = AssistantAgent(
            name="assistant",
            system_message="You are a helpful assistant. Use the firecrawl tool when needed.",
            llm_config=credentials_gpt_4o_mini.llm_config,
        )
        firecrawl_tool.register_for_llm(assistant)
        response = assistant.run(
            message="Scrape the content from https://example.com",
            tools=assistant.tools,
            max_turns=2,
        )
        print(f"Response: {response}")
        assert response is not None
