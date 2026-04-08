# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

import requests
from pydantic import BaseModel

from autogen.tools import Tool

# Base URL for the Semantic Scholar Academic Graph API.
BASE_URL = "https://api.semanticscholar.org/graph/v1"

# Maximum allowed length for a query string.
MAX_QUERY_LENGTH = 300

# Maximum number of results to retrieve from a search.
MAX_RESULTS = 100

# Default fields to retrieve for each paper.
DEFAULT_FIELDS = "title,abstract,authors,year,citationCount,url,externalIds"


class Paper(BaseModel):
    """Pydantic model representing a Semantic Scholar paper.

    Attributes:
        paper_id (str): The Semantic Scholar paper ID.
        title (str): Title of the paper.
        abstract (Optional[str]): Abstract text, if available.
        authors (list[str]): List of author names.
        year (Optional[int]): Publication year, if available.
        citation_count (Optional[int]): Number of citations.
        url (Optional[str]): Semantic Scholar URL for the paper.
        external_ids (dict[str, str]): External identifiers (e.g., DOI, ArXiv).
    """

    paper_id: str
    title: str
    abstract: Optional[str] = None
    authors: list[str] = []
    year: Optional[int] = None
    citation_count: Optional[int] = None
    url: Optional[str] = None
    external_ids: dict[str, str] = {}


class SemanticScholarClient:
    """Client for interacting with the Semantic Scholar Academic Graph API.

    Provides methods to search for papers by keyword query and to look up
    individual papers by their Semantic Scholar ID or external ID (e.g., DOI, ArXiv).

    Public methods:
        search_papers(query, limit, fields) -> list[dict[str, Any]]
        get_paper(paper_id, fields) -> dict[str, Any]

    Attributes:
        base_url (str): Base URL for the API.
        headers (dict[str, str]): HTTP headers for requests.
        api_key (Optional[str]): Optional API key for higher rate limits.
    """

    def __init__(self, api_key: Optional[str] = None, tool_name: str = "semantic-scholar-client") -> None:
        """Initialize the SemanticScholarClient.

        Args:
            api_key (Optional[str]): Optional Semantic Scholar API key for higher rate limits.
                The public API works without a key but has lower rate limits.
            tool_name (str): Identifier for User-Agent header.
        """
        self.base_url = BASE_URL
        self.headers: dict[str, str] = {"User-Agent": f"autogen.Agent ({tool_name})"}
        if api_key:
            self.headers["x-api-key"] = api_key

    def search_papers(
        self, query: str, limit: int = 10, fields: str = DEFAULT_FIELDS
    ) -> list[dict[str, Any]]:
        """Search Semantic Scholar for papers matching a query string.

        Args:
            query (str): The search keywords.
            limit (int): Max number of results to return (capped at MAX_RESULTS).
            fields (str): Comma-separated list of fields to retrieve.

        Returns:
            list[dict[str, Any]]: Each dict contains paper data with the requested fields.

        Raises:
            requests.HTTPError: If the HTTP request to the API fails.
        """
        params = {
            "query": query,
            "limit": str(min(limit, MAX_RESULTS)),
            "fields": fields,
        }

        response = requests.get(
            url=f"{self.base_url}/paper/search",
            params=params,
            headers=self.headers,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])

    def get_paper(self, paper_id: str, fields: str = DEFAULT_FIELDS) -> dict[str, Any]:
        """Retrieve details for a specific paper by its ID.

        The paper_id can be a Semantic Scholar ID, DOI (prefixed with "DOI:"),
        ArXiv ID (prefixed with "ARXIV:"), or other supported external IDs.

        Args:
            paper_id (str): The paper identifier.
            fields (str): Comma-separated list of fields to retrieve.

        Returns:
            dict[str, Any]: Paper data with the requested fields.

        Raises:
            requests.HTTPError: If the HTTP request fails (e.g., 404 for unknown paper).
        """
        params = {"fields": fields}

        response = requests.get(
            url=f"{self.base_url}/paper/{paper_id}",
            params=params,
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()


def _parse_paper(raw: dict[str, Any]) -> Paper:
    """Parse a raw API response dict into a Paper model.

    Args:
        raw (dict[str, Any]): Raw paper data from the Semantic Scholar API.

    Returns:
        Paper: A structured Paper object.
    """
    authors = []
    for author in raw.get("authors", []) or []:
        name = author.get("name")
        if name:
            authors.append(name)

    external_ids = {}
    for key, value in (raw.get("externalIds") or {}).items():
        if value is not None:
            external_ids[key] = str(value)

    return Paper(
        paper_id=raw.get("paperId", ""),
        title=raw.get("title", ""),
        abstract=raw.get("abstract"),
        authors=authors,
        year=raw.get("year"),
        citation_count=raw.get("citationCount"),
        url=raw.get("url"),
        external_ids=external_ids,
    )


def _format_paper(paper: Paper) -> str:
    """Format a Paper object into a human-readable string.

    Args:
        paper (Paper): The paper to format.

    Returns:
        str: Formatted string with paper details.
    """
    parts = [f"Title: {paper.title}"]
    if paper.authors:
        parts.append(f"Authors: {', '.join(paper.authors)}")
    if paper.year is not None:
        parts.append(f"Year: {paper.year}")
    if paper.citation_count is not None:
        parts.append(f"Citations: {paper.citation_count}")
    if paper.abstract:
        parts.append(f"Abstract: {paper.abstract}")
    if paper.url:
        parts.append(f"URL: {paper.url}")
    if paper.external_ids:
        ids_str = ", ".join(f"{k}: {v}" for k, v in paper.external_ids.items())
        parts.append(f"External IDs: {ids_str}")
    return "\n".join(parts)


class SemanticScholarSearchTool(Tool):
    """Tool for searching academic papers on Semantic Scholar.

    This tool uses the Semantic Scholar Academic Graph API to search for
    papers by keyword query and retrieve structured paper information
    including title, authors, abstract, year, citation count, and external IDs.

    The public API is free and does not require an API key, though providing
    one increases rate limits (see https://www.semanticscholar.org/product/api).

    Public methods:
        search_papers(query: str) -> list[str] | str
        get_paper(paper_id: str) -> str

    Attributes:
        max_results (int): Max number of papers returned per search.
        fields (str): Comma-separated fields to request from the API.
        verbose (bool): If True, enables debug logging to stdout.
        client (SemanticScholarClient): Internal client for API calls.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 10,
        fields: str = DEFAULT_FIELDS,
        verbose: bool = False,
    ) -> None:
        """Initialize the SemanticScholarSearchTool.

        Args:
            api_key (Optional[str]): Optional API key for higher rate limits.
            max_results (int): Desired number of search results (capped by MAX_RESULTS).
            fields (str): Comma-separated list of paper fields to retrieve.
                Supported fields include: title, abstract, authors, year,
                citationCount, url, externalIds, referenceCount, influentialCitationCount,
                fieldsOfStudy, publicationTypes, publicationDate, journal, venue.
            verbose (bool): If True, print debug information during searches.
        """
        self.tool_name = "semantic-scholar-search"
        self.max_results = min(max_results, MAX_RESULTS)
        self.fields = fields
        self.verbose = verbose
        self.client = SemanticScholarClient(api_key=api_key, tool_name=self.tool_name)
        super().__init__(
            name=self.tool_name,
            description=(
                "Search Semantic Scholar for academic papers by keyword query. "
                "Returns paper titles, authors, abstracts, publication years, "
                "citation counts, and external identifiers (DOI, ArXiv, etc.). "
                "Useful for literature review, finding related work, or looking up "
                "specific academic publications."
            ),
            func_or_tool=self.search_papers,
        )

    def search_papers(self, query: str) -> list[str] | str:
        """Search Semantic Scholar and return formatted paper results.

        Truncates the query to MAX_QUERY_LENGTH before searching.

        Args:
            query (str): Search term(s) to look up on Semantic Scholar.

        Returns:
            list[str]: Each element is a formatted string with paper details.
            str: Error message if no results are found or on exception.

        Note:
            Automatically handles API exceptions and returns error strings
            for robust operation in agent pipelines.
        """
        try:
            truncated_query = query[:MAX_QUERY_LENGTH]
            if self.verbose:
                print(
                    f"INFO\t [{self.tool_name}] search query='{truncated_query}' "
                    f"max_results={self.max_results}"
                )
            raw_results = self.client.search_papers(
                query=truncated_query,
                limit=self.max_results,
                fields=self.fields,
            )
            if not raw_results:
                return "No papers found matching the query on Semantic Scholar"

            formatted: list[str] = []
            for raw in raw_results:
                paper = _parse_paper(raw)
                formatted.append(_format_paper(paper))
            return formatted

        except Exception as e:
            return f"Semantic Scholar search failed: {str(e)}"

    def get_paper(self, paper_id: str) -> str:
        """Look up a specific paper by its ID on Semantic Scholar.

        The paper_id can be:
        - A Semantic Scholar paper ID (e.g., "649def34f8be52c8b66281af98ae884c09aef38b")
        - A DOI prefixed with "DOI:" (e.g., "DOI:10.1234/example")
        - An ArXiv ID prefixed with "ARXIV:" (e.g., "ARXIV:2106.01345")
        - A Corpus ID prefixed with "CorpusId:" (e.g., "CorpusId:215416146")

        Args:
            paper_id (str): The paper identifier.

        Returns:
            str: Formatted string with paper details, or an error message.
        """
        try:
            if self.verbose:
                print(f"INFO\t [{self.tool_name}] get paper_id='{paper_id}'")
            raw = self.client.get_paper(paper_id=paper_id, fields=self.fields)
            paper = _parse_paper(raw)
            return _format_paper(paper)

        except Exception as e:
            return f"Semantic Scholar paper lookup failed: {str(e)}"
