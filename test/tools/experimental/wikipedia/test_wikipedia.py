# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
import requests

from autogen import AssistantAgent
from autogen.import_utils import run_for_optional_imports
from autogen.tools.experimental.wikipedia.wikipedia import (
    Document,
    WikipediaClient,
    WikipediaPageLoadTool,
    WikipediaQueryRunTool,
)

from ....conftest import Credentials


# A simple fake page class to simulate a wikipediaapi.WikipediaPage.
class FakePage:
    def __init__(self, exists: bool, summary: str = "", text: str = "") -> None:
        self._exists = exists
        self.summary = summary
        self.text = text

    def exists(self) -> bool:
        return self._exists


@run_for_optional_imports("wikipediaapi", "wikipedia")
class TestWikipediaClient:
    """
    Test suite for the WikipediaClient class.
    """

    @patch("autogen.tools.experimental.wikipedia.wikipedia.requests.get")
    def test_search_success(self, mock_get: MagicMock) -> None:
        # Simulate a valid JSON response from Wikipedia API.
        fake_json = {
            "query": {
                "search": [
                    {
                        "title": "Test Page",
                        "pageid": 123,
                        "timestamp": "2023-01-01T00:00:00Z",
                        "wordcount": 100,
                        "size": 500,
                    }
                ]
            }
        }
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = fake_json
        mock_get.return_value = mock_response

        client = WikipediaClient()
        results = client.search("Test")
        assert len(results) == 1
        assert results[0]["title"] == "Test Page"

    @patch("autogen.tools.experimental.wikipedia.wikipedia.requests.get")
    def test_search_http_error(self, mock_get: MagicMock) -> None:
        # Simulate an HTTP error response.
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("HTTP Error")
        mock_get.return_value = mock_response

        client = WikipediaClient()
        with pytest.raises(requests.HTTPError):
            client.search("Test")

    def test_get_page_exists(self) -> None:
        # Simulate a page that exists.
        client = WikipediaClient()
        fake_page = FakePage(True, summary="Fake summary", text="Fake text")
        with patch.object(client.wiki, "page", return_value=fake_page):
            page = client.get_page("Fake Page")
            assert page is not None
            assert page.summary == "Fake summary"

    def test_get_page_nonexistent(self) -> None:
        # Simulate a page that does not exist.
        client = WikipediaClient()
        fake_page = FakePage(False)
        with patch.object(client.wiki, "page", return_value=fake_page):
            page = client.get_page("Nonexistent Page")
            assert page is None


@run_for_optional_imports("wikipediaapi", "wikipedia")
class TestWikipediaQueryRunTool:
    """
    Test suite for the WikipediaQueryRunTool class.
    """

    @pytest.fixture
    def tool(self) -> WikipediaQueryRunTool:
        """
        Provide a WikipediaQueryRunTool instance for testing.

        Returns:
            WikipediaQueryRunTool: A configured tool instance with verbose off
        """
        return WikipediaQueryRunTool(verbose=False)

    def test_query_run_success(self, tool: WikipediaQueryRunTool) -> None:
        # Patch the search method.
        with (
            patch.object(
                tool.wiki_cli,
                "search",
                return_value=[
                    {
                        "title": "Test Page",
                        "pageid": 123,
                        "timestamp": "2023-01-01T00:00:00Z",
                        "wordcount": 100,
                        "size": 500,
                    }
                ],
            ),
            patch.object(
                tool.wiki_cli,
                "get_page",
                return_value=FakePage(True, summary="Test summary", text="Test text"),
            ),
        ):
            # Simulate query run success scenario
            result = tool.query_run("Some test query")
            # Expect a list with formatted summary.
            assert isinstance(result, list)
            assert "Page: Test Page" in result[0]
            assert "Summary: Test summary" in result[0]

    def test_query_run_no_results(self, tool: WikipediaQueryRunTool) -> None:
        # Simulate no search results.
        with patch.object(tool.wiki_cli, "search", return_value=[]):
            result = tool.query_run("Some test query")
            assert result == "No good Wikipedia Search Result was found"

    def test_query_run_exception(self, tool: WikipediaQueryRunTool) -> None:
        # Simulate an exception during search.
        with patch.object(tool.wiki_cli, "search", side_effect=Exception("fail")):
            result = tool.query_run("Some test query")
            assert isinstance(result, str)
            assert result.startswith("wikipedia search failed: ")

    @run_for_optional_imports("openai", "openai")
    def test_agent_integration(self, credentials_gpt_4o_mini: Credentials) -> None:
        # Integration test for verifying the registration of the WikipediaQueryRunTool with an AssistantAgent.
        search_tool = WikipediaPageLoadTool()
        assistant = AssistantAgent(
            name="assistant",
            system_message="You are a helpful assistant. Use the wikipedia page load tool when needed.",
            llm_config=credentials_gpt_4o_mini.llm_config,
        )
        search_tool.register_for_llm(assistant)
        assert isinstance(assistant.tools[0], WikipediaQueryRunTool)
        assert assistant.tools[0].name == "wikipedia-query-run"


@run_for_optional_imports("wikipediaapi", "wikipedia")
class TestWikipediaPageLoadTool:
    """
    Test suite for the WikipediaPageLoadTool class.
    """

    @pytest.fixture
    def tool(self) -> WikipediaPageLoadTool:
        """
        Provide a WikipediaPageLoadTool instance for testing.

        Returns:
            WikipediaPageLoadTool: A configured tool instance with verbose off
        """
        return WikipediaPageLoadTool(verbose=False)

    def test_content_search_success(self, tool: WikipediaPageLoadTool) -> None:
        # Simulate successful search results.
        fake_search_result = [
            {
                "title": "Test Page",
                "pageid": 123,
                "timestamp": "2023-01-01T00:00:00Z",
                "wordcount": 100,
                "size": 500,
            }
        ]

        with (
            patch.object(
                tool.wiki_cli,
                "search",
                return_value=fake_search_result,
            ),
            patch.object(
                tool.wiki_cli,
                "get_page",
                return_value=FakePage(True, summary="Test summary", text="Test text content that is long enough"),
            ),
        ):
            result = tool.content_search("Some test query")
            assert isinstance(result, list)
            assert len(result) > 0
            assert isinstance(result[0], Document)
            assert result[0].metadata["title"] == "Test Page"
            assert result[0].page_content.startswith("Test text")

    def test_content_search_no_results(self, tool: WikipediaPageLoadTool) -> None:
        # Simulate no search results.
        with patch.object(tool.wiki_cli, "search", return_value=[]):
            result = tool.content_search("Some test query")
            assert result == "No good Wikipedia Search Result was found"

    def test_content_search_exception(self, tool: WikipediaPageLoadTool) -> None:
        # Simulate an exception during search.
        with patch.object(tool.wiki_cli, "search", side_effect=Exception("fail")):
            result = tool.content_search("Some test query")
            assert isinstance(result, str)
            assert result.startswith("wikipedia search failed: ")

    @run_for_optional_imports("openai", "openai")
    def test_agent_integration(self, credentials_gpt_4o_mini: Credentials) -> None:
        # Integration test for verifying the registration of the WikipediaPageLoadTool with an AssistantAgent.
        search_tool = WikipediaPageLoadTool()
        assistant = AssistantAgent(
            name="assistant",
            system_message="You are a helpful assistant. Use the wikipedia page load tool when needed.",
            llm_config=credentials_gpt_4o_mini.llm_config,
        )
        search_tool.register_for_llm(assistant)
        assert isinstance(assistant.tools[0], WikipediaPageLoadTool)
        assert assistant.tools[0].name == "wikipedia-page-load"
