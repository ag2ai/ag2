# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated, Any
from urllib.parse import urlparse

from ....import_utils import optional_import_block, require_optional_import
from ....llm_config import LLMConfig
from ... import Depends, Tool
from ...dependency_injection import on

logger = logging.getLogger(__name__)
_API_INTEGRATION = "ag2"
_API_INTEGRATION_ENV_VAR = "TF_API_INTEGRATION"
_SAFE_URL_SCHEMES = {"http", "https"}

with optional_import_block():
    from tinyfish import TinyFish


def _safe_url(url: str) -> bool:
    return urlparse(url).scheme.lower() in _SAFE_URL_SCHEMES


@contextmanager
def _tinyfish_api_integration() -> Iterator[None]:
    previous = os.environ.get(_API_INTEGRATION_ENV_VAR)
    os.environ[_API_INTEGRATION_ENV_VAR] = _API_INTEGRATION
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_API_INTEGRATION_ENV_VAR, None)
        else:
            os.environ[_API_INTEGRATION_ENV_VAR] = previous


@require_optional_import(
    [
        "tinyfish",
    ],
    "tinyfish",
)
def _execute_tinyfish_scrape(
    url: str,
    goal: str,
    tinyfish_api_key: str,
) -> dict[str, Any]:
    """Execute a goal-directed scrape using the TinyFish API.

    Args:
        url (str): The URL to scrape.
        goal (str): A natural language description of what to extract.
        tinyfish_api_key (str): The API key for TinyFish.

    Returns:
        dict[str, Any]: The scrape result from TinyFish.
    """
    client = TinyFish(api_key=tinyfish_api_key)
    try:
        with _tinyfish_api_integration():
            response = client.agent.run(url=url, goal=goal)
    finally:
        client.close()

    if response.status == "COMPLETED" and response.result:
        if isinstance(response.result, dict):
            return response.result
        return {"result": response.result}
    elif response.error:
        return {"status": "error", "error": response.error}
    else:
        return {"status": "no_result", "error": "No result returned."}


@require_optional_import(
    [
        "tinyfish",
    ],
    "tinyfish",
)
def _execute_tinyfish_search(
    query: str,
    tinyfish_api_key: str,
    location: str | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    """Execute a TinyFish Search API query.

    Args:
        query (str): The search query string.
        tinyfish_api_key (str): The API key for TinyFish.
        location (str | None): Optional country or location for geo-targeted results.
        language (str | None): Optional language code for result language.

    Returns:
        dict[str, Any]: The search response from TinyFish.
    """
    client = TinyFish(api_key=tinyfish_api_key)
    try:
        response = client.search.query(query=query, location=location, language=language)
    finally:
        client.close()

    return {
        "query": response.query,
        "total_results": response.total_results,
        "results": [
            {
                "position": result.position,
                "site_name": result.site_name,
                "title": result.title,
                "snippet": result.snippet,
                "url": result.url,
            }
            for result in response.results
        ],
    }


@require_optional_import(
    [
        "tinyfish",
    ],
    "tinyfish",
)
def _execute_tinyfish_fetch(
    urls: list[str],
    tinyfish_api_key: str,
    format: str | None = None,
    links: bool | None = None,
    image_links: bool | None = None,
) -> dict[str, Any]:
    """Execute a TinyFish Fetch API request.

    Args:
        urls (list[str]): URLs to fetch and extract.
        tinyfish_api_key (str): The API key for TinyFish.
        format (str | None): Output format: "markdown", "html", or "json".
        links (bool | None): Whether to include page links.
        image_links (bool | None): Whether to include image links.

    Returns:
        dict[str, Any]: The fetch response from TinyFish.
    """
    client = TinyFish(api_key=tinyfish_api_key)
    try:
        with _tinyfish_api_integration():
            response = client.fetch.get_contents(urls=urls, format=format, links=links, image_links=image_links)
    finally:
        client.close()

    return {
        "results": [
            {
                "url": result.url,
                "final_url": result.final_url,
                "title": result.title,
                "description": result.description,
                "language": result.language,
                "author": result.author,
                "published_date": result.published_date,
                "text": result.text,
                "format": result.format,
                "links": list(result.links or []),
                "image_links": list(result.image_links or []),
            }
            for result in response.results
        ],
        "errors": [{"url": error.url, "error": error.error} for error in response.errors],
    }


def _tinyfish_scrape(
    url: str,
    goal: str,
    tinyfish_api_key: str,
) -> dict[str, Any]:
    """Perform a TinyFish scrape and format the results.

    Args:
        url (str): The URL to scrape.
        goal (str): A natural language description of what to extract.
        tinyfish_api_key (str): The API key for TinyFish.

    Returns:
        dict[str, Any]: The scraped data, or an error dict.
    """
    try:
        result = _execute_tinyfish_scrape(
            url=url,
            goal=goal,
            tinyfish_api_key=tinyfish_api_key,
        )
        return {
            "url": url,
            "goal": goal,
            "data": result,
        }
    except Exception as e:
        logger.error(f"TinyFish scrape failed: {e}")
        return {
            "url": url,
            "goal": goal,
            "data": {},
            "error": str(e),
        }


def _tinyfish_search(
    query: str,
    tinyfish_api_key: str,
    location: str | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    """Perform a TinyFish search and format the results.

    Args:
        query (str): The search query string.
        tinyfish_api_key (str): The API key for TinyFish.
        location (str | None): Optional country or location for geo-targeted results.
        language (str | None): Optional language code for result language.

    Returns:
        dict[str, Any]: The search result, or an error dict.
    """
    try:
        return _execute_tinyfish_search(
            query=query,
            tinyfish_api_key=tinyfish_api_key,
            location=location,
            language=language,
        )
    except Exception as e:
        logger.error(f"TinyFish search failed: {e}")
        return {
            "query": query,
            "results": [],
            "total_results": 0,
            "error": str(e),
        }


def _tinyfish_fetch(
    urls: list[str],
    tinyfish_api_key: str,
    format: str | None = None,
    links: bool | None = None,
    image_links: bool | None = None,
) -> dict[str, Any]:
    """Perform a TinyFish fetch and format the results.

    Args:
        urls (list[str]): URLs to fetch and extract.
        tinyfish_api_key (str): The API key for TinyFish.
        format (str | None): Output format: "markdown", "html", or "json".
        links (bool | None): Whether to include page links.
        image_links (bool | None): Whether to include image links.

    Returns:
        dict[str, Any]: The fetch result, or an error dict.
    """
    invalid_urls = [url for url in urls if not _safe_url(url)]
    if invalid_urls:
        return {
            "results": [],
            "errors": [{"url": url, "error": "Only http/https URLs are supported."} for url in invalid_urls],
        }

    try:
        return _execute_tinyfish_fetch(
            urls=urls,
            tinyfish_api_key=tinyfish_api_key,
            format=format,
            links=links,
            image_links=image_links,
        )
    except Exception as e:
        logger.error(f"TinyFish fetch failed: {e}")
        return {
            "results": [],
            "errors": [{"url": url, "error": str(e)} for url in urls],
        }


class TinyFishTool(Tool):
    """TinyFishTool is a tool that uses the TinyFish API to deep-scrape web pages with a natural language goal.

    TinyFish performs goal-directed web scraping — you provide a URL and describe what information
    you want to extract, and it returns structured results.

    This tool requires a TinyFish API key, which can be provided during initialization or set as
    an environment variable ``TINYFISH_API_KEY``.

    Attributes:
        tinyfish_api_key (str): The API key used for authenticating with the TinyFish API.
    """

    def __init__(
        self,
        *,
        llm_config: LLMConfig | dict[str, Any] | None = None,
        tinyfish_api_key: str | None = None,
    ):
        """Initializes the TinyFishTool.

        Args:
            llm_config (Optional[Union[LLMConfig, dict[str, Any]]]): LLM configuration.
                (Currently unused but kept for potential future integration).
            tinyfish_api_key (Optional[str]): The API key for the TinyFish API. If not provided,
                it attempts to read from the ``TINYFISH_API_KEY`` environment variable.

        Raises:
            ValueError: If ``tinyfish_api_key`` is not provided either directly or via the environment variable.
        """
        self.tinyfish_api_key = tinyfish_api_key or os.getenv("TINYFISH_API_KEY")

        if self.tinyfish_api_key is None:
            raise ValueError("tinyfish_api_key must be provided either as an argument or via TINYFISH_API_KEY env var")

        def tinyfish_scrape(
            url: Annotated[str, "The URL to scrape."],
            goal: Annotated[str, "A natural language description of what information to extract from the page."],
            tinyfish_api_key: Annotated[str | None, Depends(on(self.tinyfish_api_key))],
        ) -> dict[str, Any]:
            """Deep-scrape a URL using TinyFish with a natural language goal.

            Pass a URL and describe what you want to extract. TinyFish will navigate the page
            and return structured data matching your goal.

            Args:
                url: The URL to scrape.
                goal: A natural language description of what to extract (e.g.,
                    "Find all team members and their roles",
                    "Extract pricing information and plan details").
                tinyfish_api_key: The API key for TinyFish (injected dependency).

            Returns:
                A dictionary containing the URL, goal, and extracted data.

            Raises:
                ValueError: If the TinyFish API key is not available.
            """
            if tinyfish_api_key is None:
                raise ValueError("TinyFish API key is missing.")
            return _tinyfish_scrape(
                url=url,
                goal=goal,
                tinyfish_api_key=tinyfish_api_key,
            )

        super().__init__(
            name="tinyfish_scrape",
            description="Deep-scrape a URL using TinyFish. Pass a URL and a natural language goal describing what to extract.",
            func_or_tool=tinyfish_scrape,
        )


class TinyFishSearchTool(Tool):
    """TinyFishSearchTool uses the TinyFish Search API to search the web.

    TinyFish Search returns ranked results with title, snippet, site name, and URL.
    This tool requires a TinyFish API key, which can be provided during initialization
    or set as an environment variable ``TINYFISH_API_KEY``.

    Attributes:
        tinyfish_api_key (str): The API key used for authenticating with the TinyFish API.
    """

    def __init__(
        self,
        *,
        llm_config: LLMConfig | dict[str, Any] | None = None,
        tinyfish_api_key: str | None = None,
    ):
        """Initializes the TinyFishSearchTool.

        Args:
            llm_config (Optional[Union[LLMConfig, dict[str, Any]]]): LLM configuration.
                (Currently unused but kept for potential future integration).
            tinyfish_api_key (Optional[str]): The API key for the TinyFish API. If not provided,
                it attempts to read from the ``TINYFISH_API_KEY`` environment variable.

        Raises:
            ValueError: If ``tinyfish_api_key`` is not provided either directly or via the environment variable.
        """
        self.tinyfish_api_key = tinyfish_api_key or os.getenv("TINYFISH_API_KEY")

        if self.tinyfish_api_key is None:
            raise ValueError("tinyfish_api_key must be provided either as an argument or via TINYFISH_API_KEY env var")

        def tinyfish_search(
            query: Annotated[str, "The search query string."],
            tinyfish_api_key: Annotated[str | None, Depends(on(self.tinyfish_api_key))],
            location: Annotated[str | None, "Optional country or location for geo-targeted results."] = None,
            language: Annotated[str | None, "Optional language code for result language."] = None,
        ) -> dict[str, Any]:
            """Search the web using TinyFish Search.

            Args:
                query: The search query string.
                tinyfish_api_key: The API key for TinyFish (injected dependency).
                location: Optional country or location for geo-targeted results.
                language: Optional language code for result language.

            Returns:
                A dictionary containing the query, total result count, and ranked results.

            Raises:
                ValueError: If the TinyFish API key is not available.
            """
            if tinyfish_api_key is None:
                raise ValueError("TinyFish API key is missing.")
            return _tinyfish_search(
                query=query,
                tinyfish_api_key=tinyfish_api_key,
                location=location,
                language=language,
            )

        super().__init__(
            name="tinyfish_search",
            description="Search the web using TinyFish. Returns ranked results with titles, snippets, site names, and URLs.",
            func_or_tool=tinyfish_search,
        )


class TinyFishFetchTool(Tool):
    """TinyFishFetchTool uses the TinyFish Fetch API to extract content from URLs.

    TinyFish Fetch renders pages in a browser and returns clean extracted content,
    metadata, and per-URL errors. This tool requires a TinyFish API key, which can
    be provided during initialization or set as an environment variable
    ``TINYFISH_API_KEY``.

    Attributes:
        tinyfish_api_key (str): The API key used for authenticating with the TinyFish API.
    """

    def __init__(
        self,
        *,
        llm_config: LLMConfig | dict[str, Any] | None = None,
        tinyfish_api_key: str | None = None,
    ):
        """Initializes the TinyFishFetchTool.

        Args:
            llm_config (Optional[Union[LLMConfig, dict[str, Any]]]): LLM configuration.
                (Currently unused but kept for potential future integration).
            tinyfish_api_key (Optional[str]): The API key for the TinyFish API. If not provided,
                it attempts to read from the ``TINYFISH_API_KEY`` environment variable.

        Raises:
            ValueError: If ``tinyfish_api_key`` is not provided either directly or via the environment variable.
        """
        self.tinyfish_api_key = tinyfish_api_key or os.getenv("TINYFISH_API_KEY")

        if self.tinyfish_api_key is None:
            raise ValueError("tinyfish_api_key must be provided either as an argument or via TINYFISH_API_KEY env var")

        def tinyfish_fetch(
            urls: Annotated[list[str], "URLs to fetch and extract. TinyFish supports 1-10 URLs."],
            tinyfish_api_key: Annotated[str | None, Depends(on(self.tinyfish_api_key))],
            format: Annotated[str | None, "Output format: 'markdown', 'html', or 'json'."] = None,
            links: Annotated[bool | None, "Whether to include page links in results."] = None,
            image_links: Annotated[bool | None, "Whether to include image links in results."] = None,
        ) -> dict[str, Any]:
            """Fetch and extract content from URLs using TinyFish Fetch.

            Args:
                urls: URLs to fetch and extract.
                tinyfish_api_key: The API key for TinyFish (injected dependency).
                format: Output format: "markdown", "html", or "json".
                links: Whether to include page links in results.
                image_links: Whether to include image links in results.

            Returns:
                A dictionary containing successful fetch results and per-URL errors.

            Raises:
                ValueError: If the TinyFish API key is not available.
            """
            if tinyfish_api_key is None:
                raise ValueError("TinyFish API key is missing.")
            return _tinyfish_fetch(
                urls=urls,
                tinyfish_api_key=tinyfish_api_key,
                format=format,
                links=links,
                image_links=image_links,
            )

        super().__init__(
            name="tinyfish_fetch",
            description="Fetch and extract clean content from URLs using TinyFish. Returns extracted content and per-URL errors.",
            func_or_tool=tinyfish_fetch,
        )
