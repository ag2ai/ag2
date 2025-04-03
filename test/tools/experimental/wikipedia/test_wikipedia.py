# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock, patch

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
class TestWikipediaClient(unittest.TestCase):
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
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Test Page")

    @patch("autogen.tools.experimental.wikipedia.wikipedia.requests.get")
    def test_search_http_error(self, mock_get: MagicMock) -> None:
        # Simulate an HTTP error response.
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("HTTP Error")
        mock_get.return_value = mock_response

        client = WikipediaClient()
        with self.assertRaises(requests.HTTPError):
            client.search("Test")

    def test_get_page_exists(self) -> None:
        # Simulate a page that exists.
        client = WikipediaClient()
        fake_page = FakePage(True, summary="Fake summary", text="Fake text")
        with patch.object(client.wiki, "page", return_value=fake_page):
            page = client.get_page("Fake Page")
            self.assertIsNotNone(page)
            if page is None:
                self.fail("Expected page to be not None")
            self.assertEqual(page.summary, "Fake summary")

    def test_get_page_nonexistent(self) -> None:
        # Simulate a page that does not exist.
        client = WikipediaClient()
        fake_page = FakePage(False)
        with patch.object(client.wiki, "page", return_value=fake_page):
            page = client.get_page("Nonexistent Page")
            self.assertIsNone(page)

@run_for_optional_imports("wikipediaapi", "wikipedia")
class TestWikipediaQueryRunTool(unittest.TestCase):
    def setUp(self) -> None:
        # Create an instance of the tool with verbose off.
        self.tool = WikipediaQueryRunTool(verbose=False)
        # Patch the search method.
        self.patcher_search = patch.object(
            self.tool.wiki_cli,
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
        )
        self.mock_search = self.patcher_search.start()
        self.addCleanup(self.patcher_search.stop)

        # Patch the get_page method.
        fake_page = FakePage(True, summary="Test summary", text="Test text")
        self.patcher_get_page = patch.object(
            self.tool.wiki_cli,
            "get_page",
            return_value=fake_page,
        )
        self.mock_get_page = self.patcher_get_page.start()
        self.addCleanup(self.patcher_get_page.stop)

    def test_query_run_success(self) -> None:
        # Simulate query run success scenario
        result = self.tool.query_run("Some test query")
        # Expect a list with formatted summary.
        self.assertIsInstance(result, list)
        if isinstance(result, list):
            self.assertIn("Page: Test Page", result[0])
            self.assertIn("Summary: Test summary", result[0])
        else:
            self.fail("Expected result to be a list")

    def test_query_run_no_results(self) -> None:
        # Simulate no search results.
        self.mock_search.return_value = []
        result = self.tool.query_run("Some test query")
        self.assertEqual(result, "No good Wikipedia Search Result was found")

    def test_query_run_exception(self) -> None:
        # Simulate an exception during search.
        self.mock_search.side_effect = Exception("fail")
        result = self.tool.query_run("Some test query")
        if isinstance(result, str):
            self.assertTrue(result.startswith("wikipedia search failed: "))
        else:
            self.fail("Expected result to be a string error message")

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
class TestWikipediaPageLoadTool(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = WikipediaPageLoadTool(verbose=False)
        self.fake_search_result = [
            {
                "title": "Test Page",
                "pageid": 123,
                "timestamp": "2023-01-01T00:00:00Z",
                "wordcount": 100,
                "size": 500,
            }
        ]
        # Patch the search method.
        self.patcher_search = patch.object(
            self.tool.wiki_cli,
            "search",
            return_value=self.fake_search_result,
        )
        self.mock_search = self.patcher_search.start()
        self.addCleanup(self.patcher_search.stop)

        # Patch the get_page method.
        fake_page = FakePage(True, summary="Test summary", text="Test text content that is long enough")
        self.patcher_get_page = patch.object(
            self.tool.wiki_cli,
            "get_page",
            return_value=fake_page,
        )
        self.mock_get_page = self.patcher_get_page.start()
        self.addCleanup(self.patcher_get_page.stop)

    def test_content_search_success(self) -> None:
        # Simulate successful search results.
        result = self.tool.content_search("Some test query")
        if isinstance(result, list):
            self.assertGreater(len(result), 0)
            self.assertIsInstance(result[0], Document)
            self.assertEqual(result[0].metadata["title"], "Test Page")
            self.assertTrue(result[0].page_content.startswith("Test text"))
        else:
            self.fail("Expected result to be a list of Document objects")

    def test_content_search_no_results(self) -> None:
        # Simulate no search results.
        self.mock_search.return_value = []
        result = self.tool.content_search("Some test query")
        self.assertEqual(result, "No good Wikipedia Search Result was found")

    def test_content_search_exception(self) -> None:
        # Simulate an exception during search.
        self.mock_search.side_effect = Exception("fail")
        result = self.tool.content_search("Some test query")
        if isinstance(result, str):
            self.assertTrue(result.startswith("wikipedia search failed: "))
        else:
            self.fail("Expected result to be a string error message")

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
