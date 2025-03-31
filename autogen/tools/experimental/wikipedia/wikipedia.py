# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

import requests
from pydantic import BaseModel

from autogen.import_utils import optional_import_block, require_optional_import
from autogen.tools import Tool

with optional_import_block():
    import wikipediaapi

# Maximum allowed length for a query string.
MAX_QUERY_LENGTH = 300
# Maximum number of pages to retrieve from a search.
MAX_PAGE_RETRIEVE = 100
# Maximum number of characters to return from a Wikipedia page.
MAX_ARTICLE_LENGTH = 10000


class Document(BaseModel):
    """
    A Pydantic data model representing a Wikipedia document.

    Attributes:
        page_content (str): The textual content of the Wikipedia page (possibly truncated).
        metadata (dict[str, str]): A dictionary containing additional metadata such as:
            - source URL
            - title
            - pageid
            - timestamp
            - word count
            - size
    """

    page_content: str
    metadata: dict[str, str]


class WikipediaClient:
    """
    A client for interfacing with the Wikipedia API.

    This client supports basic search functionality and page retrieval for a specific
    language edition of Wikipedia.
    """

    def __init__(self, language: str = "en", tool_name: str = "wikipedia-client") -> None:
        """
        Initialize the WikipediaClient.
        """
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": f"autogen.Agent ({tool_name})"}
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent=f"autogen.Agent ({tool_name})",
        )

    def search(self, query: str, limit: int = 3) -> Any:
        """
        Search Wikipedia for pages matching the given query.
        """
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": str(limit),
            "srprop": "size|wordcount|timestamp",
        }

        response = requests.get(url=self.base_url, params=params, headers=self.headers)
        response.raise_for_status()
        data = response.json()
        search_data = data.get("query", {}).get("search", [])
        return search_data

    def get_page(self, title: str) -> Optional[Any]:
        """
        Retrieve a Wikipedia page by its title.

        Returns:
            A WikipediaPage object if it exists; otherwise, None.
        """
        page = self.wiki.page(title)
        if not page.exists():
            return None
        return page


@require_optional_import(["wikipediaapi"], "wikipedia")
class WikipediaQueryRunTool(Tool):
    """
    A tool to run a Wikipedia query and return summarized page results.
    """

    def __init__(self, language: str = "en", top_k: int = 3, verbose: bool = False) -> None:
        """
        Initialize the WikipediaQueryRunTool.
        """
        self.language = language
        self.tool_name = "wikipedia-query-run"
        self.wiki_cli = WikipediaClient(language, self.tool_name)
        self.top_k = min(top_k, MAX_PAGE_RETRIEVE)
        self.verbose = verbose
        super().__init__(
            name=self.tool_name,
            description="Use this tool to run a query in Wikipedia. It returns the summary of the pages found.",
            func_or_tool=self.query_run,
        )

    def query_run(self, query: str) -> Union[list[str], str]:
        """
        Execute a Wikipedia query and return summaries of the matching pages.
        """
        try:
            if self.verbose:
                print(f"INFO\t [{self.tool_name}] search query='{query[:MAX_QUERY_LENGTH]}' top_k={self.top_k}")
            search_results = self.wiki_cli.search(query[:MAX_QUERY_LENGTH], limit=self.top_k)
            summaries: list[str] = []
            for item in search_results:
                title = item["title"]
                page = self.wiki_cli.get_page(title)
                # Only format the summary if the page exists and has a summary.
                if page is not None and page.summary:
                    summary = f"Page: {title}\nSummary: {page.summary}"
                    summaries.append(summary)
            if not summaries:
                return "No good Wikipedia Search Result was found"
            return summaries
        except Exception as e:
            return f"wikipedia search failed: {str(e)}"


@require_optional_import(["wikipediaapi"], "wikipedia")
class WikipediaPageLoadTool(Tool):
    """
    A tool to load full Wikipedia page content along with metadata.
    """

    def __init__(self, language: str = "en", top_k: int = 3, truncate: int = 4000, verbose: bool = False) -> None:
        """
        Initialize the WikipediaPageLoadTool.
        """
        self.language = language
        self.top_k = min(top_k, MAX_PAGE_RETRIEVE)
        self.truncate = min(truncate, MAX_ARTICLE_LENGTH)
        self.verbose = verbose
        self.tool_name = "wikipedia-page-load"
        self.wiki_cli = WikipediaClient(language, self.tool_name)
        super().__init__(
            name=self.tool_name,
            description=(
                "Use this tool to search query in Wikipedia. "
                "This tool returns the content of multiple pages for a given query: set number of pages with 'limit' parameter. "
                "This tool uses a language-specific Wikipedia client to search for relevant articles and returns a list "
                "of Document objects containing truncated page content along with rich metadata (source URL, title, pageid, "
                "timestamp, word count, and size). Ideal for Agents requiring comprehensive, structured Wikipedia data "
                "for research, summarization, or contextual enrichment."
            ),
            func_or_tool=self.content_search,
        )

    def content_search(self, query: str) -> Union[list[Document], str]:
        """
        Execute a Wikipedia query and load full page content with metadata.
        """
        try:
            if self.verbose:
                print(f"INFO\t [{self.tool_name}] search query='{query[:MAX_QUERY_LENGTH]}' top_k={self.top_k}")
            search_results = self.wiki_cli.search(query[:MAX_QUERY_LENGTH], limit=self.top_k)
            docs: list[Document] = []
            for item in search_results:
                page = self.wiki_cli.get_page(item["title"])
                # Only process pages that exist and have text content.
                if page is not None and page.text:
                    document = Document(
                        page_content=page.text[: self.truncate],
                        metadata={
                            "source": f"https://{self.language}.wikipedia.org/?curid={item['pageid']}",
                            "title": item["title"],
                            "pageid": str(item["pageid"]),
                            "timestamp": str(item["timestamp"]),
                            "wordcount": str(item["wordcount"]),
                            "size": str(item["size"]),
                        },
                    )
                    docs.append(document)
            if not docs:
                return "No good Wikipedia Search Result was found"
            return docs

        except Exception as e:
            return f"wikipedia search failed: {str(e)}"
