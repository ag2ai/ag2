# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import BaseModel

from autogen.import_utils import optional_import_block, require_optional_import
from autogen.tools import Tool

with optional_import_block():
    import arxiv

# Maximum allowed length for a query string.
MAX_QUERY_LENGTH = 300
# Maximum number of results to retrieve from a search.
MAX_RESULTS = 50
# Maximum number of characters to return from a paper summary.
MAX_SUMMARY_LENGTH = 2000


class ArxivArticle(BaseModel):
    """Pydantic model representing an arXiv article.

    Attributes:
        title (str): Title of the article.
        authors (str): Comma-separated list of author names.
        summary (str): Abstract/summary of the article (possibly truncated).
        published (str): Publication date in 'YYYY-MM' format.
        entry_id (str): The arXiv entry URL (e.g., 'http://arxiv.org/abs/2301.00001v1').
        pdf_url (str): Direct URL to the PDF.
        doi (str): DOI of the article, if available.
        primary_category (str): Primary arXiv category (e.g., 'cs.AI').
    """

    title: str
    authors: str
    summary: str
    published: str
    entry_id: str
    pdf_url: str
    doi: str
    primary_category: str


class ArxivClient:
    """Client for interacting with the arXiv API.

    Supports searching for papers by query string with configurable
    sorting and result limits.

    Public methods:
        search(query: str, max_results: int, sort_by: str) -> list[ArxivArticle]

    Attributes:
        client (arxiv.Client): Low-level arXiv API client.
        truncate (int): Maximum number of characters for article summaries.
    """

    # Mapping of user-friendly sort names to arxiv.SortCriterion values.
    SORT_CRITERIA = {
        "relevance": "Relevance",
        "submittedDate": "SubmittedDate",
        "lastUpdatedDate": "LastUpdatedDate",
    }

    def __init__(self, truncate: int = MAX_SUMMARY_LENGTH) -> None:
        """Initialize the ArxivClient.

        Args:
            truncate (int): Maximum number of characters for each article summary.
        """
        self.client = arxiv.Client()
        self.truncate = min(truncate, MAX_SUMMARY_LENGTH)

    def search(self, query: str, max_results: int = 5, sort_by: str = "relevance") -> list[ArxivArticle]:
        """Search arXiv for papers matching a query string.

        Args:
            query (str): The search keywords or arXiv query expression.
            max_results (int): Max number of results to return (capped at MAX_RESULTS).
            sort_by (str): Sorting criterion. One of 'relevance', 'submittedDate',
                or 'lastUpdatedDate'.

        Returns:
            list[ArxivArticle]: A list of ArxivArticle objects with paper metadata.

        Raises:
            ValueError: If sort_by is not a recognized criterion.
            arxiv.ArxivError: On lower-level API errors.
        """
        max_results = min(max_results, MAX_RESULTS)

        criterion_name = self.SORT_CRITERIA.get(sort_by)
        if criterion_name is None:
            raise ValueError(
                f"Invalid sort_by value: '{sort_by}'. "
                f"Must be one of: {', '.join(self.SORT_CRITERIA.keys())}"
            )
        criterion = getattr(arxiv.SortCriterion, criterion_name)

        search = arxiv.Search(query=query, max_results=max_results, sort_by=criterion)
        results = self.client.results(search)

        articles: list[ArxivArticle] = []
        for result in results:
            article = ArxivArticle(
                title=result.title,
                authors=", ".join(a.name for a in result.authors),
                summary=result.summary[: self.truncate],
                published=result.published.strftime("%Y-%m"),
                entry_id=result.entry_id,
                pdf_url=result.pdf_url,
                doi=result.doi or "",
                primary_category=result.primary_category,
            )
            articles.append(article)

        return articles


@require_optional_import(["arxiv"], "arxiv")
class ArxivSearchTool(Tool):
    """Tool for searching arXiv and returning structured article metadata.

    This tool uses the `arxiv` Python package to search the arXiv repository
    and returns up to `max_results` articles with their title, authors,
    summary, publication date, entry ID, PDF URL, DOI, and primary category.

    Public methods:
        search(query: str) -> list[ArxivArticle] | str

    Attributes:
        max_results (int): Max number of articles returned per query (capped at MAX_RESULTS).
        sort_by (str): Sorting criterion for search results.
        truncate (int): Max characters for each article summary.
        verbose (bool): If True, enables debug logging to stdout.
        arxiv_client (ArxivClient): Internal client for arXiv API calls.
    """

    def __init__(
        self,
        max_results: int = 5,
        sort_by: str = "relevance",
        truncate: int = MAX_SUMMARY_LENGTH,
        verbose: bool = False,
    ) -> None:
        """Initialize the ArxivSearchTool.

        Args:
            max_results (int): Desired number of results (capped at MAX_RESULTS).
            sort_by (str): Sorting criterion. One of 'relevance', 'submittedDate',
                or 'lastUpdatedDate'.
            truncate (int): Maximum number of characters for article summaries
                (capped at MAX_SUMMARY_LENGTH).
            verbose (bool): If True, print debug information during searches.
        """
        self.max_results = min(max_results, MAX_RESULTS)
        self.sort_by = sort_by
        self.truncate = min(truncate, MAX_SUMMARY_LENGTH)
        self.verbose = verbose
        self.tool_name = "arxiv-search"
        self.arxiv_client = ArxivClient(truncate=self.truncate)
        super().__init__(
            name=self.tool_name,
            description=(
                "Search arXiv for academic papers matching a query. "
                "Returns a list of articles with title, authors, summary, "
                "publication date, arXiv entry ID, PDF URL, DOI, and primary category. "
                "Useful for research, literature review, and finding recent papers."
            ),
            func_or_tool=self.search,
        )

    def search(self, query: str) -> list[ArxivArticle] | str:
        """Search arXiv and return structured article results.

        Truncates the query to MAX_QUERY_LENGTH before searching.

        Args:
            query (str): Search term(s) or arXiv query expression to look up.

        Returns:
            list[ArxivArticle]: Each element contains article metadata.
            str: Error message if no results are found or on exception.

        Note:
            Automatically handles API exceptions and returns error strings
            for robust operation in agent workflows.
        """
        try:
            truncated_query = query[:MAX_QUERY_LENGTH]
            if self.verbose:
                print(
                    f"INFO\t [{self.tool_name}] search query='{truncated_query}' "
                    f"max_results={self.max_results} sort_by={self.sort_by}"
                )
            articles = self.arxiv_client.search(
                query=truncated_query,
                max_results=self.max_results,
                sort_by=self.sort_by,
            )
            if not articles:
                return "No arXiv search results found"
            return articles
        except Exception as e:
            return f"arXiv search failed: {str(e)}"
