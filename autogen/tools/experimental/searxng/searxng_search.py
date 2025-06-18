# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""
SearxNG Search Tool
A simple tool to perform web searches using a SearxNG instance.
"""

import logging
from typing import Annotated, Any, List, Optional

import requests

from autogen.doc_utils import export_module
from autogen.tools import Tool

logger = logging.getLogger(__name__)


def _execute_searxng_query(
    query: str,
    max_results: int = 5,
    categories: Optional[List[str]] = None,
    language: Optional[str] = None,
    base_url: str = "https://searxng.site/search",
) -> list[dict[str, Any]]:
    """
    Execute a search query using a SearxNG instance.
    Args:
        query (str): The search query string.
        max_results (int, optional): The maximum number of results to return. Defaults to 5.
        categories (Optional[List[str]]): List of categories to search in.
        language (Optional[str]): Language code.
        base_url (str): SearxNG instance URL.
    Returns:
        list[dict[str, Any]]: A list of search results from SearxNG.
    """
    params = {
        "q": query,
        "format": "json",
        "language": language or "en-US",
        "categories": ",".join(categories) if categories else None,
        "count": max_results,
    }
    params = {k: v for k, v in params.items() if v is not None}
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not isinstance(results, list):
            return []
        # Ensure each result is a dict before returning
        typed_results: list[dict[str, Any]] = []
        for item in results:
            if isinstance(item, dict):
                typed_results.append(item)
        return typed_results
    except Exception as e:
        logger.error(f"SearxNG Search failed: {e}")
        return []


def _searxng_search(
    query: str,
    max_results: int = 5,
    categories: Optional[List[str]] = None,
    language: Optional[str] = None,
    base_url: str = "https://searxng.site/search",
) -> list[dict[str, Any]]:
    """
    Perform a SearxNG search and format the results.
    Args:
        query (str): The search query string.
        max_results (int, optional): The maximum number of results to return. Defaults to 5.
        categories (Optional[List[str]]): List of categories to search in.
        language (Optional[str]): Language code.
        base_url (str): SearxNG instance URL.
    Returns:
        list[dict[str, Any]]: A list of dictionaries with 'title', 'link', and 'snippet'.
    """
    res = _execute_searxng_query(
        query=query,
        max_results=max_results,
        categories=categories,
        language=language,
        base_url=base_url,
    )
    formatted_results: list[dict[str, Any]] = [
        {"title": item.get("title", ""), "link": item.get("url", ""), "snippet": item.get("content", "")}
        for item in res
    ]
    return formatted_results


@export_module("autogen.tools.experimental")
class SearxngSearchTool(Tool):
    """
    SearxngSearchTool is a tool that uses SearxNG to perform a search.

    This tool allows agents to leverage the SearxNG search engine for information retrieval.
    SearxNG does not require an API key by default, making it easy to use.
    """

    def __init__(self, base_url: str = "https://searxng.site/search") -> None:
        """
        Initializes the SearxngSearchTool.
        Args:
            base_url (str): The SearxNG instance URL.
        """
        self.base_url = base_url
        super().__init__(
            name="searxng_search",
            description="Use the SearxNG API to perform a search.",
            func_or_tool=self.searxng_search,
        )

    def searxng_search(
        self,
        query: Annotated[str, "The search query."],
        max_results: Annotated[int, "The number of results to return."] = 5,
        categories: Annotated[Optional[List[str]], "List of categories to search in."] = None,
        language: Annotated[Optional[str], "Language code (e.g., 'en-US')."] = None,
    ) -> list[dict[str, Any]]:
        """
        Performs a search using the SearxNG API and returns formatted results.
        Args:
            query: The search query string.
            max_results: The maximum number of results to return. Defaults to 5.
            categories: List of categories to search in.
            language: Language code.
        Returns:
            A list of dictionaries, each containing 'title', 'link', and 'snippet' of a search result.
        """
        return _searxng_search(
            query=query,
            max_results=max_results,
            categories=categories,
            language=language,
            base_url=self.base_url,
        )
