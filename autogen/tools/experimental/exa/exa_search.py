# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Annotated, Any

from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import
from ... import Depends, Tool
from ...dependency_injection import on

with optional_import_block():
    from exa_py import Exa


@require_optional_import(
    [
        "exa_py",
    ],
    "exa",
)
def _execute_exa_search(
    query: str,
    exa_api_key: str,
    num_results: int = 10,
    search_type: str = "auto",
    include_text: bool = True,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    start_published_date: str | None = None,
    end_published_date: str | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Execute a search query using the Exa API and return formatted results.

    Args:
        query: The search query string.
        exa_api_key: The API key for Exa.
        num_results: The maximum number of results to return. Defaults to 10.
        search_type: The type of search to perform ('auto', 'neural', or 'keyword'). Defaults to "auto".
        include_text: Whether to include cleaned page text in results. Defaults to True.
        include_domains: A list of domains to restrict search to. Defaults to None.
        exclude_domains: A list of domains to exclude from search. Defaults to None.
        start_published_date: Filter results published after this ISO date. Defaults to None.
        end_published_date: Filter results published before this ISO date. Defaults to None.
        category: Filter by content category (e.g. 'research paper', 'news', 'company', 'tweet'). Defaults to None.

    Returns:
        A list of dictionaries containing search result data.
    """
    exa = Exa(api_key=exa_api_key)
    exa.headers["x-exa-integration"] = "ag2"

    kwargs: dict[str, Any] = {
        "num_results": num_results,
        "type": search_type,
        "text": include_text,
    }
    if include_domains is not None:
        kwargs["include_domains"] = include_domains
    if exclude_domains is not None:
        kwargs["exclude_domains"] = exclude_domains
    if start_published_date is not None:
        kwargs["start_published_date"] = start_published_date
    if end_published_date is not None:
        kwargs["end_published_date"] = end_published_date
    if category is not None:
        kwargs["category"] = category

    response = exa.search_and_contents(query, **kwargs)

    results = []
    for item in response.results:
        result: dict[str, Any] = {
            "title": item.title or "",
            "url": item.url or "",
        }
        if include_text and item.text:
            result["text"] = item.text
        if item.published_date:
            result["published_date"] = item.published_date
        if item.author:
            result["author"] = item.author
        if item.score is not None:
            result["score"] = item.score
        results.append(result)

    return results


@require_optional_import(
    [
        "exa_py",
    ],
    "exa",
)
def _execute_exa_find_similar(
    url: str,
    exa_api_key: str,
    num_results: int = 10,
    include_text: bool = True,
    exclude_source_domain: bool = True,
) -> list[dict[str, Any]]:
    """Find pages similar to a given URL using the Exa API.

    Args:
        url: The URL to find similar pages for.
        exa_api_key: The API key for Exa.
        num_results: The maximum number of results to return. Defaults to 10.
        include_text: Whether to include cleaned page text in results. Defaults to True.
        exclude_source_domain: Whether to exclude results from the source domain. Defaults to True.

    Returns:
        A list of dictionaries containing similar page data.
    """
    exa = Exa(api_key=exa_api_key)
    exa.headers["x-exa-integration"] = "ag2"

    response = exa.find_similar_and_contents(
        url,
        num_results=num_results,
        text=include_text,
        exclude_source_domain=exclude_source_domain,
    )

    results = []
    for item in response.results:
        result: dict[str, Any] = {
            "title": item.title or "",
            "url": item.url or "",
        }
        if include_text and item.text:
            result["text"] = item.text
        if item.published_date:
            result["published_date"] = item.published_date
        if item.score is not None:
            result["score"] = item.score
        results.append(result)

    return results


@export_module("autogen.tools.experimental")
class ExaSearchTool(Tool):
    """ExaSearchTool provides web search and similarity search via the Exa API.

    Exa is a search engine designed for AI, providing clean and relevant results
    with optional full-text content extraction. It supports neural (semantic),
    keyword, and auto search modes, as well as finding pages similar to a given URL.

    This tool requires an Exa API key, which can be provided during initialization
    or set as an environment variable ``EXA_API_KEY``.

    Attributes:
        exa_api_key (str): The API key used for authenticating with the Exa API.
    """

    def __init__(self, *, exa_api_key: str | None = None):
        """Initialize the ExaSearchTool.

        Args:
            exa_api_key (Optional[str]): The API key for the Exa API. If not provided,
                it attempts to read from the ``EXA_API_KEY`` environment variable.

        Raises:
            ValueError: If ``exa_api_key`` is not provided either directly or via the environment variable.
        """
        self.exa_api_key = exa_api_key or os.getenv("EXA_API_KEY")

        if self.exa_api_key is None:
            raise ValueError("exa_api_key must be provided either as an argument or via EXA_API_KEY env var")

        def exa_search(
            query: Annotated[str, "The search query."],
            exa_api_key: Annotated[str | None, Depends(on(self.exa_api_key))],
            num_results: Annotated[int, "The number of results to return."] = 10,
            search_type: Annotated[str, "Search type: 'auto', 'neural', or 'keyword'."] = "auto",
            include_text: Annotated[bool, "Include cleaned page text in results."] = True,
            include_domains: Annotated[list[str] | None, "Restrict search to these domains."] = None,
            exclude_domains: Annotated[list[str] | None, "Exclude these domains from search."] = None,
            start_published_date: Annotated[
                str | None, "Filter results published after this ISO date (e.g. '2024-01-01')."
            ] = None,
            end_published_date: Annotated[
                str | None, "Filter results published before this ISO date (e.g. '2025-01-01')."
            ] = None,
            category: Annotated[
                str | None,
                "Filter by content category: 'research paper', 'news', 'company', 'tweet', 'personal site', etc.",
            ] = None,
        ) -> list[dict[str, Any]]:
            """Search the web using Exa and return results with optional page content.

            Args:
                query: The search query string.
                exa_api_key: The API key for Exa (injected dependency).
                num_results: The maximum number of results to return. Defaults to 10.
                search_type: The search type. Defaults to "auto".
                include_text: Whether to include page text. Defaults to True.
                include_domains: Domains to restrict search to. Defaults to None.
                exclude_domains: Domains to exclude. Defaults to None.
                start_published_date: ISO date lower bound. Defaults to None.
                end_published_date: ISO date upper bound. Defaults to None.
                category: Content category filter. Defaults to None.

            Returns:
                A list of dictionaries with 'title', 'url', and optional 'text', 'published_date',
                'author', 'score' fields.

            Raises:
                ValueError: If the Exa API key is not available.
            """
            if exa_api_key is None:
                raise ValueError("Exa API key is missing.")
            return _execute_exa_search(
                query=query,
                exa_api_key=exa_api_key,
                num_results=num_results,
                search_type=search_type,
                include_text=include_text,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                start_published_date=start_published_date,
                end_published_date=end_published_date,
                category=category,
            )

        super().__init__(
            name="exa_search",
            description="Search the web using Exa's neural search engine. Returns relevant results with optional cleaned page text.",
            func_or_tool=exa_search,
        )
