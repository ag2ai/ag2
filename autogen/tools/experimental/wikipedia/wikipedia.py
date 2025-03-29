# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Union

import requests

from autogen.import_utils import optional_import_block, require_optional_import
from autogen.tools import Tool

with optional_import_block():
    import wikipediaapi


class WikipediaPrefixSearchTool(Tool):
    """
    A Wikipedia search tool that provides title suggestions using prefix matching.

    Specializes in finding article titles that begin with the given query string,
    ideal for auto-complete functionality and disambiguation of partial titles.

    Features:
    - Real-time title suggestions from Wikipedia's OpenSearch API
    - Configurable result limits
    - Error-resistant API handling
    - Main namespace filtering (excludes meta-pages)

    Args:
        language: ISO 639-1 language code (default: 'en')
    """

    def __init__(self, language: str = "en") -> None:
        """
        Initialize the prefix search tool with target Wikipedia language edition.
        """
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"
        super().__init__(
            name="wikipedia-prefix-search",
            description=(
                "Search Wikipedia article titles starting with input text. "
                "Use when the input text is incomplete, partial, or ambiguous search terms.\n\n"
                "Input: A partial title string (case-insensitive).\n"
                "Output: A List[str] of matching titles or an error message.\n\n"
                "Next Steps: Chain with summary tool for condensed insights or page content retrieval tool for detailed results."
            ),
            func_or_tool=self.prefix_search,
        )

    def prefix_search(self, query: str, limit: int = 3) -> Union[str, Any]:
        """
        Retrieve Wikipedia article titles matching a search prefix.

        Args:
            query: Initial characters of article titles to match
            limit: Maximum number of suggestions (1-100, default:10)

        Returns:
            Union[List[str], str]:
            - List of matching article titles in order of relevance
            - Error message with details if request fails

        Notes:
            - Matches only titles in Wikipedia's main article namespace
            - Results are case-insensitive but preserve Wikipedia's capitalization
            - Maximum API-enforced limit is 100 suggestions per request
        """
        params = {
            "action": "opensearch",
            "search": query,
            "limit": str(limit),
            "namespace": "0",
            "format": "json",
        }

        try:
            response = requests.get(
                url=self.base_url, params=params, headers={"User-Agent": "autogen.Agent (prefix-search)"}
            )
            response.raise_for_status()

            # OpenSearch format: [query, titles, descriptions, URLs]
            return response.json()[1]

        except Exception as e:
            return f"Prefix search failed: {str(e)}"


@require_optional_import(["wikipediaapi"], "wikipedia")
class WikipediaTextRetrieverTool(Tool):
    """
    A specialized tool for retrieving full text content from Wikipedia pages.

    Inherits from a base Tool class and implements Wikipedia integration using wikipediaapi.
    Designed for integration into agent workflows requiring factual content retrieval.

    Features:
    - Language customization (default: English)
    - Page existence validation
    - Direct text content extraction

    Args:
        language: ISO language code for Wikipedia version (default: 'en')
    """

    def __init__(self, language: str = "en") -> None:
        """
        Initialize Wikipedia API client and configure tool properties.
        """
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="autogen.Agent (wikipedia-text-extractor)",  # Identifies requests to Wikipedia's API
        )
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"
        super().__init__(
            name="wikipedia-text-extractor",
            description="Retrieves full text content of Wikipedia pages by exact title",
            func_or_tool=self.get_page_text,
        )

    def get_page_text(self, title: str) -> Union[str, Any]:
        """
        Retrieve complete text content from a Wikipedia page.

        Args:
            title: Exact title of the Wikipedia page (case-sensitive)

        Returns:
            str:
            - Page content as raw wiki-text string if page exists
            - Error message dictionary if page not found

        Notes:
            - Requires exact title match (use WikipediaSearchTool for discovery)
            - Returns wiki markup text format (not HTML)
            - Maximum page size depends on Wikipedia API limits
        """
        try:
            page = self.wiki.page(title)
            if not page.exists():
                return f"No Wikipedia page found with title: '{title}'"
            return page.text
        except Exception as e:
            return f"Text content retrieve failed: {str(e)}"


@require_optional_import(["wikipediaapi"], "wikipedia")
class WikipediaSummaryRetrieverTool(Tool):
    """
    A specialized tool for extracting structured summaries from Wikipedia articles.

    Inherits from base Tool class to provide agent-friendly access to Wikipedia's
    summary content. Optimized for factual accuracy and quick information retrieval.

    Features:
    - Language-specific Wikipedia version support
    - Summary extraction in clean text format
    - Automatic page existence verification
    - API request rate limiting compliance

    Args:
        language (str): ISO 639-1 language code (default: 'en')
    """

    def __init__(self, language: str = "en") -> None:
        """
        Initialize Wikipedia API client and configure tool properties.

        Args:
            language: Supported Wikipedia language edition code
        """
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="autogen.Agent (wikipedia-summary-extractor)",
        )
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"
        super().__init__(
            name="wikipedia-summary-extractor",
            description="Retrieves structured summary of Wikipedia pages by exact title",
            func_or_tool=self.get_page_summary,
        )

    def get_page_summary(self, title: str) -> Union[str, Any]:
        """
        Retrieve the lead section summary of a Wikipedia article.

        Args:
            title: Exact case-sensitive title of the target Wikipedia page

        Returns:
            str:
            - First section summary as plain text string (if found)
            - Error message with details (if page not found)

        Notes:
            - Summaries typically contain 2-5 paragraphs of key information
            - Returns cleaned text without wiki markup or references
            - Requires exact title match (use search tools for discovery)
        """
        try:
            page = self.wiki.page(title)
            if not page.exists():
                return f"No Wikipedia page found with title: '{title}'"
            return page.summary

        except Exception as e:
            return f"Summary retrieve failed: {str(e)}"


@require_optional_import(["wikipediaapi"], "wikipedia")
class WikipediaTopicSearchTool(Tool):
    """
    A comprehensive Wikipedia research tool combining search and batch content retrieval capabilities.

    Features:
    - Full-text search with metadata (word count, timestamps, relevance snippets)
    - Exact title matching verification
    - Batch content retrieval with length control
    - Multi-page topic analysis foundation

    Args:
        language: Wikipedia language edition code (default: 'en')

    Raises:
        ConnectionError: If Wikipedia API endpoints are unreachable
    """

    def __init__(self, language: str = "en") -> None:
        """
        Initialize Wikipedia API client with research-focused configuration.

        Args:
            language: Supported Wikipedia language code (e.g., 'fr' for French)
        """
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="autogen.Agent (wikipedia-topic-search)",
        )
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"
        super().__init__(
            name="wikipedia-topic-search",
            description="Search a topic on Wikipedia without and return the content of the pages found. You should terminate or wait for user instructions after returning the results",
            func_or_tool=self.search_topic,
        )

    def _get_pages(self, topic: str, limit: int = 3) -> dict[str, Any]:
        """
        Internal method: Execute Wikipedia search query and process raw results.

        Args:
            topic: Search phrase or page title
            limit: Maximum results to return (1-50 enforced by API)

        Returns:
            Dictionary with:
            - results: dict[str, Any] - Search matches with metadata:
                * title: Page title
                * pageid: Unique identifier
                * size: Content size in bytes
                * wordcount: Text length in words
                * timestamp: Last edit ISO timestamp

        Note:
            Case-insensitive title matching when exact_match=True
        """
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": topic,
            "srlimit": str(limit),
            "srprop": "size|wordcount|timestamp",
        }

        response = requests.get(url=self.base_url, params=params)
        response.raise_for_status()  # Propagates HTTP errors
        data = response.json()
        search_data = data.get("query", {}).get("search", [])
        result = {"results": search_data}

        return result

    def search_topic(self, topic: str, limit: int = 3, truncate: int = 2000) -> Union[dict[str, Any], str]:
        """
        Retrieve and process content for multiple related Wikipedia pages.

        Args:
            topic: Research subject or keyword phrase
            limit: Max pages to analyze (1-10 recommended)
            truncate: Character limit per page content (0-10000)

        Returns:
            dictionary:
            - Successful: {page_title: truncated_content}
            - Error: {"error": description}
        """
        try:
            search_results = self._get_pages(topic, limit=limit)
            content_dict = {}

            for item in search_results.get("results", []):
                page = self.wiki.page(item["title"])
                if page.exists() and page.text:
                    content_dict[item["title"]] = page.text[: max(0, truncate)]

            return content_dict or {"error": "No valid pages found"}

        except Exception as e:
            return f"Topic search failed: {str(e)}"
