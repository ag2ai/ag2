# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Any
from unittest.mock import MagicMock, Mock, create_autospec, patch

import requests
import wikipediaapi

from autogen import AssistantAgent
from autogen.import_utils import run_for_optional_imports
from autogen.tools.experimental import (
    WikipediaPrefixSearchTool,
    WikipediaSummaryRetrieverTool,
    WikipediaTextRetrieverTool,
    WikipediaTopicSearchTool,
)

from ....conftest import Credentials


# Fake page object to simulate wikipediaapi behavior.
class FakePage:
    def __init__(self, exists: bool, text: str = "", summary: str = ""):
        self._exists = exists
        self.text = text
        self.summary = summary

    def exists(self) -> bool:
        return self._exists


class TestWikipediaPrefixSearchTool(unittest.TestCase):
    def setUp(self) -> None:
        # Initialize the tool instance (using English Wikipedia)
        self.tool = WikipediaPrefixSearchTool(language="en")

    @patch("requests.get")
    def test_prefix_search_success(self, mock_get: Mock) -> None:
        # Simulate a successful response from Wikipedia's OpenSearch API
        fake_titles = ["Apple", "Applesauce", "Applet"]
        fake_json = ["App", fake_titles, ["desc1", "desc2", "desc3"], ["url1", "url2", "url3"]]

        fake_response = Mock()
        fake_response.json.return_value = fake_json
        fake_response.raise_for_status.return_value = None  # No HTTP error

        mock_get.return_value = fake_response

        # Call the prefix_search method with a query string and a limit of 3
        result = self.tool.prefix_search("App", limit=3)

        # Verify that the method returns the list of titles from the fake response
        self.assertEqual(result, fake_titles)

        # Verify that the correct URL and parameters were used in the API request
        mock_get.assert_called_once()
        called_args = mock_get.call_args[1]
        self.assertEqual(called_args["url"], self.tool.base_url)
        self.assertEqual(called_args["params"]["search"], "App")
        self.assertEqual(called_args["params"]["limit"], "3")
        self.assertEqual(called_args["params"]["namespace"], "0")
        self.assertEqual(called_args["params"]["format"], "json")
        self.assertIn("User-Agent", called_args["headers"])

    @patch("requests.get")
    def test_prefix_search_failure(self, mock_get: Mock) -> None:
        # Simulate an exception (e.g., network error)
        mock_get.side_effect = Exception("Network error")

        result = self.tool.prefix_search("App", limit=3)

        # Check that an error message is returned indicating failure
        self.assertTrue(result.startswith("Prefix search failed:"))
        self.assertIn("Network error", result)

    @run_for_optional_imports("openai", "openai")
    def test_agent_integration(self, credentials_gpt_4o_mini: Credentials) -> None:
        """
        Test integration with AssistantAgent.
        """
        assistant = AssistantAgent(
            name="assistant",
            system_message="You are a helpful assistant.",
            llm_config=credentials_gpt_4o_mini.llm_config,
        )
        self.tool.register_for_llm(assistant)
        assert isinstance(assistant.tools[0], WikipediaPrefixSearchTool)
        assert assistant.tools[0].name == "wikipedia-prefix-search"


class TestWikipediaTextRetrieverTool(unittest.TestCase):
    def setUp(self) -> None:
        # Initialize the tool with the default language (English)
        self.tool = WikipediaTextRetrieverTool(language="en")

    def test_get_page_text_success(self) -> None:
        # Simulate a successful page retrieval: page exists and contains content.
        fake_page = FakePage(exists=True, text="Sample article content")
        # Patch the wiki.page method to return our fake page
        self.tool.wiki.page = MagicMock(return_value=fake_page)

        result: Any = self.tool.get_page_text("Sample Title")
        self.assertEqual(result, "Sample article content")
        self.tool.wiki.page.assert_called_once_with("Sample Title")

    def test_get_page_text_page_not_found(self) -> None:
        # Simulate the case where the page does not exist.
        fake_page = FakePage(exists=False, text="")
        self.tool.wiki.page = MagicMock(return_value=fake_page)

        result: Any = self.tool.get_page_text("Nonexistent Title")
        expected_message = "No Wikipedia page found with title: 'Nonexistent Title'"
        self.assertEqual(result, expected_message)
        self.tool.wiki.page.assert_called_once_with("Nonexistent Title")

    def test_get_page_text_exception(self) -> None:
        # Simulate an exception occurring during the API call.
        self.tool.wiki.page = MagicMock(side_effect=Exception("API error"))

        result: Any = self.tool.get_page_text("Any Title")
        self.assertTrue(result.startswith("Text content retrieve failed:"))
        self.assertIn("API error", result)

    @run_for_optional_imports("openai", "openai")
    def test_agent_integration(self, credentials_gpt_4o_mini: Credentials) -> None:
        """
        Test integration with AssistantAgent.
        """
        assistant = AssistantAgent(
            name="assistant",
            system_message="You are a helpful assistant.",
            llm_config=credentials_gpt_4o_mini.llm_config,
        )
        self.tool.register_for_llm(assistant)
        assert isinstance(assistant.tools[0], WikipediaTextRetrieverTool)
        assert assistant.tools[0].name == "wikipedia-text-extractor"


class TestWikipediaSummaryRetrieverTool(unittest.TestCase):
    def setUp(self) -> None:
        # Initialize the tool instance (default language 'en')
        self.tool = WikipediaSummaryRetrieverTool(language="en")

    def test_get_page_summary_success(self) -> None:
        # Simulate a page that exists with a valid summary
        fake_page = FakePage(exists=True, summary="This is the article summary.")
        self.tool.wiki.page = MagicMock(return_value=fake_page)

        result: Any = self.tool.get_page_summary("Sample Title")
        self.assertEqual(result, "This is the article summary.")
        self.tool.wiki.page.assert_called_once_with("Sample Title")

    def test_get_page_summary_page_not_found(self) -> None:
        # Simulate a page that does not exist
        fake_page = FakePage(exists=False, summary="")
        self.tool.wiki.page = MagicMock(return_value=fake_page)

        result: Any = self.tool.get_page_summary("Nonexistent Title")
        expected = "No Wikipedia page found with title: 'Nonexistent Title'"
        self.assertEqual(result, expected)
        self.tool.wiki.page.assert_called_once_with("Nonexistent Title")

    def test_get_page_summary_exception(self) -> None:
        # Simulate an exception during the API call
        self.tool.wiki.page = MagicMock(side_effect=Exception("API error"))

        result: Any = self.tool.get_page_summary("Any Title")
        self.assertTrue(result.startswith("Summary retrieve failed:"))
        self.assertIn("API error", result)

    @run_for_optional_imports("openai", "openai")
    def test_agent_integration(self, credentials_gpt_4o_mini: Credentials) -> None:
        """
        Test integration with AssistantAgent.
        """
        assistant = AssistantAgent(
            name="assistant",
            system_message="You are a helpful assistant.",
            llm_config=credentials_gpt_4o_mini.llm_config,
        )
        self.tool.register_for_llm(assistant)
        assert isinstance(assistant.tools[0], WikipediaSummaryRetrieverTool)
        assert assistant.tools[0].name == "wikipedia-summary-extractor"


class TestWikipediaTopicSearchTool(unittest.TestCase):
    """
    Unit tests for the WikipediaTopicSearchTool class.

    This test suite validates:
    - Initialization of the tool with default and custom languages.
    - Parameter validation for search limits and truncation.
    - Successful retrieval of Wikipedia pages.
    - Error handling for network issues and invalid pages.
    - Correct truncation of page content.
    """

    def setUp(self) -> None:
        """Set up mocks and the tool instance."""
        self.mock_wiki = create_autospec(wikipediaapi.Wikipedia)
        self.mock_page = MagicMock()
        self.mock_page.exists.return_value = True
        self.mock_page.text = "Sample content " * 200  # 3000+ characters
        self.mock_wiki.page.return_value = self.mock_page

        self.tool = WikipediaTopicSearchTool()
        self.tool.wiki = self.mock_wiki

    @patch("requests.get")
    def test_initialization_default_language(self, mock_get: Mock) -> None:
        tool = WikipediaTopicSearchTool()
        self.assertEqual(tool.base_url, "https://en.wikipedia.org/w/api.php")
        self.assertEqual(tool.wiki.language, "en")

    @patch("requests.get")
    def test_initialization_custom_language(self, mock_get: Mock) -> None:
        tool = WikipediaTopicSearchTool(language="es")
        self.assertEqual(tool.base_url, "https://es.wikipedia.org/w/api.php")
        self.assertEqual(tool.wiki.language, "es")

    @patch("requests.get")
    def test_get_pages_success(self, mock_get: Mock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "query": {
                "search": [
                    {
                        "title": "Test Page 1",
                        "pageid": 123,
                        "size": 4500,
                        "wordcount": 700,
                        "timestamp": "2024-01-01T12:00:00Z",
                    }
                ]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.tool._get_pages("test", limit=3)

        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["title"], "Test Page 1")
        mock_response.raise_for_status.assert_called_once()

    @patch("requests.get")
    def test_get_pages_api_error(self, mock_get: Mock) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("API Error")
        mock_get.return_value = mock_response

        with self.assertRaises(requests.HTTPError):
            self.tool._get_pages("test")

    # In TestWikipediaTopicSearchTool2 class
    @patch.object(WikipediaTopicSearchTool, "_get_pages")
    def test_search_topic_success(self, mock_get_pages: Mock) -> None:
        # Configure mock to return consistent data
        mock_get_pages.return_value = {"results": [{"title": "Test Page 1"}]}

        result = self.tool.search_topic("test")
        self.assertIn("Test Page 1", result)

    def test_search_topic_no_valid_pages(self) -> None:
        self.mock_page.exists.return_value = False

        result = self.tool.search_topic("test")
        self.assertEqual(result, {"error": "No valid pages found"})

    def test_search_topic_truncation(self) -> None:
        with patch.object(self.tool, "_get_pages") as mock_get_pages:
            mock_get_pages.return_value = {"results": [{"title": "Test Page 1"}]}

            # Test zero truncation
            result = self.tool.search_topic("test", truncate=0)
            self.assertEqual(len(result["Test Page 1"]), 0)

            # Test exact truncation
            result = self.tool.search_topic("test", truncate=50)
            self.assertEqual(len(result["Test Page 1"]), 50)

    @patch("requests.get")
    def test_search_topic_error_handling(self, mock_get: Mock) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.RequestException("Connection error")
        mock_get.return_value = mock_response

        result = self.tool.search_topic("test")
        self.assertEqual("topic search failed: Connection error", result["Error"])

    @patch.object(WikipediaTopicSearchTool, "_get_pages")
    def test_parameter_validation(self, mock_get_pages: Mock) -> None:
        # Test lower boundary
        self.tool.search_topic("test", limit=0)
        self.assertEqual(mock_get_pages.call_args[1]["limit"], 1)

        # Test upper boundary
        self.tool.search_topic("test", limit=100)
        self.assertEqual(mock_get_pages.call_args[1]["limit"], 50)

        # Test valid middle value
        self.tool.search_topic("test", limit=5)
        self.assertEqual(mock_get_pages.call_args[1]["limit"], 5)

    @patch.object(WikipediaTopicSearchTool, "_get_pages")
    def test_content_retrieval_priority(self, mock_get_pages: Mock) -> None:
        # Test multiple results handling
        mock_get_pages.return_value = {
            "results": [
                {"title": "Page1"},
                {"title": "Page2"},
                {"title": "Page3"},
            ]
        }

        result = self.tool.search_topic("test", limit=3)

        self.assertEqual(len(result), 3)
        self.assertEqual(self.mock_wiki.page.call_count, 3)

    @run_for_optional_imports("openai", "openai")
    def test_agent_integration(self, credentials_gpt_4o_mini: Credentials) -> None:
        """
        Test integration with AssistantAgent.
        """
        assistant = AssistantAgent(
            name="assistant",
            system_message="You are a helpful assistant.",
            llm_config=credentials_gpt_4o_mini.llm_config,
        )
        self.tool.register_for_llm(assistant)
        assert isinstance(assistant.tools[0], WikipediaTopicSearchTool)
        assert assistant.tools[0].name == "wikipedia-topic-search"


if __name__ == "__main__":
    unittest.main()
