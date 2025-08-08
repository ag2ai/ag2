# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Annotated, Any

from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import
from ....llm_config import LLMConfig
from ... import Depends, Tool
from ...dependency_injection import on

logger = logging.getLogger(__name__)

with optional_import_block():
    from firecrawl import FirecrawlApp, ScrapeOptions


@require_optional_import(
    [
        "firecrawl-py",
    ],
    "firecrawl",
)
def _execute_firecrawl_scrape(
    url: str,
    firecrawl_api_key: str,
    firecrawl_api_url: str | None = None,
    formats: list[str] | None = None,
    include_tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    headers: dict[str, str] | None = None,
    wait_for: int | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Execute a scrape operation using the Firecrawl API.

    Args:
        url (str): The URL to scrape.
        firecrawl_api_key (str): The API key for Firecrawl.
        firecrawl_api_url (str, optional): The base URL for the Firecrawl API. Defaults to None (uses default or env var).
        formats (list[str], optional): Output formats (e.g., ['markdown', 'html']). Defaults to ['markdown'].
        include_tags (list[str], optional): HTML tags to include. Defaults to None.
        exclude_tags (list[str], optional): HTML tags to exclude. Defaults to None.
        headers (dict[str, str], optional): HTTP headers to use. Defaults to None.
        wait_for (int, optional): Time to wait for page load in milliseconds. Defaults to None.
        timeout (int, optional): Request timeout in milliseconds. Defaults to None.

    Returns:
        dict[str, Any]: The scrape result from Firecrawl.
    """
    if formats is None:
        formats = ["markdown"]

    app = FirecrawlApp(api_key=firecrawl_api_key, api_url=firecrawl_api_url)

    result = app.scrape_url(
        url=url,
        formats=formats,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        headers=headers,
        wait_for=wait_for,
        timeout=timeout,
    )
    return dict(result)


@require_optional_import(
    [
        "firecrawl-py",
    ],
    "firecrawl",
)
def _execute_firecrawl_crawl(
    url: str,
    firecrawl_api_key: str,
    firecrawl_api_url: str | None = None,
    limit: int = 5,
    formats: list[str] | None = None,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    max_depth: int | None = None,
    allow_backward_crawling: bool = False,
    allow_external_content_links: bool = False,
) -> dict[str, Any]:
    """Execute a crawl operation using the Firecrawl API.

    Args:
        url (str): The starting URL to crawl.
        firecrawl_api_key (str): The API key for Firecrawl.
        firecrawl_api_url (str, optional): The base URL for the Firecrawl API. Defaults to None (uses default or env var).
        limit (int, optional): Maximum number of pages to crawl. Defaults to 5.
        formats (list[str], optional): Output formats (e.g., ['markdown', 'html']). Defaults to ['markdown'].
        include_paths (list[str], optional): URL patterns to include. Defaults to None.
        exclude_paths (list[str], optional): URL patterns to exclude. Defaults to None.
        max_depth (int, optional): Maximum crawl depth. Defaults to None.
        allow_backward_crawling (bool, optional): Allow crawling backward links. Defaults to False.
        allow_external_content_links (bool, optional): Allow external links. Defaults to False.

    Returns:
        dict[str, Any]: The crawl result from Firecrawl.
    """
    if formats is None:
        formats = ["markdown"]

    app = FirecrawlApp(api_key=firecrawl_api_key, api_url=firecrawl_api_url)

    # Build scrape options for crawling
    scrape_options = None
    if formats:
        scrape_options = ScrapeOptions(formats=formats)

    result = app.crawl_url(
        url=url,
        limit=limit,
        scrape_options=scrape_options,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        max_depth=max_depth,
        allow_backward_links=allow_backward_crawling,
        allow_external_links=allow_external_content_links,
    )
    return dict(result)


@require_optional_import(
    [
        "firecrawl-py",
    ],
    "firecrawl",
)
def _execute_firecrawl_map(
    url: str,
    firecrawl_api_key: str,
    firecrawl_api_url: str | None = None,
    search: str | None = None,
    ignore_sitemap: bool = False,
    include_subdomains: bool = False,
    limit: int = 5000,
) -> dict[str, Any]:
    """Execute a map operation using the Firecrawl API to get URLs from a website.

    Args:
        url (str): The website URL to map.
        firecrawl_api_key (str): The API key for Firecrawl.
        firecrawl_api_url (str, optional): The base URL for the Firecrawl API. Defaults to None (uses default or env var).
        search (str, optional): Search term to filter URLs. Defaults to None.
        ignore_sitemap (bool, optional): Whether to ignore the sitemap. Defaults to False.
        include_subdomains (bool, optional): Whether to include subdomains. Defaults to False.
        limit (int, optional): Maximum number of URLs to return. Defaults to 5000.

    Returns:
        dict[str, Any]: The map result from Firecrawl.
    """
    app = FirecrawlApp(api_key=firecrawl_api_key, api_url=firecrawl_api_url)

    result = app.map_url(
        url=url,
        search=search,
        ignore_sitemap=ignore_sitemap,
        include_subdomains=include_subdomains,
        limit=limit,
    )
    return dict(result)


@require_optional_import(
    [
        "firecrawl-py",
    ],
    "firecrawl",
)
def _execute_firecrawl_search(
    query: str,
    firecrawl_api_key: str,
    firecrawl_api_url: str | None = None,
    limit: int = 5,
    tbs: str | None = None,
    filter: str | None = None,
    lang: str = "en",
    country: str = "us",
    location: str | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Execute a search operation using the Firecrawl API.

    Args:
        query (str): The search query string.
        firecrawl_api_key (str): The API key for Firecrawl.
        firecrawl_api_url (str, optional): The base URL for the Firecrawl API. Defaults to None (uses default or env var).
        limit (int, optional): Maximum number of results to return. Defaults to 5.
        tbs (str, optional): Time filter (e.g., "qdr:d" for past day). Defaults to None.
        filter (str, optional): Custom result filter. Defaults to None.
        lang (str, optional): Language code. Defaults to "en".
        country (str, optional): Country code. Defaults to "us".
        location (str, optional): Geo-targeting location. Defaults to None.
        timeout (int, optional): Request timeout in milliseconds. Defaults to None.

    Returns:
        dict[str, Any]: The search result from Firecrawl.
    """
    app = FirecrawlApp(api_key=firecrawl_api_key, api_url=firecrawl_api_url)

    result = app.search(
        query=query,
        limit=limit,
        tbs=tbs,
        filter=filter,
        lang=lang,
        country=country,
        location=location,
        timeout=timeout,
    )
    return dict(result)


@require_optional_import(
    [
        "firecrawl-py",
    ],
    "firecrawl",
)
def _execute_firecrawl_deep_research(
    query: str,
    firecrawl_api_key: str,
    firecrawl_api_url: str | None = None,
    max_depth: int = 7,
    time_limit: int = 270,
    max_urls: int = 20,
    analysis_prompt: str | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Execute a deep research operation using the Firecrawl API.

    Args:
        query (str): The research query or topic to investigate.
        firecrawl_api_key (str): The API key for Firecrawl.
        firecrawl_api_url (str, optional): The base URL for the Firecrawl API. Defaults to None (uses default or env var).
        max_depth (int, optional): Maximum depth of research exploration. Defaults to 7.
        time_limit (int, optional): Time limit in seconds for research. Defaults to 270.
        max_urls (int, optional): Maximum number of URLs to process. Defaults to 20.
        analysis_prompt (str, optional): Custom prompt for analysis. Defaults to None.
        system_prompt (str, optional): Custom system prompt. Defaults to None.

    Returns:
        dict[str, Any]: The deep research result from Firecrawl.
    """
    app = FirecrawlApp(api_key=firecrawl_api_key, api_url=firecrawl_api_url)

    result = app.deep_research(
        query=query,
        max_depth=max_depth,
        time_limit=time_limit,
        max_urls=max_urls,
        analysis_prompt=analysis_prompt,
        system_prompt=system_prompt,
    )
    return dict(result)


def _firecrawl_scrape(
    url: str,
    firecrawl_api_key: str,
    firecrawl_api_url: str | None = None,
    formats: list[str] | None = None,
    include_tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    headers: dict[str, str] | None = None,
    wait_for: int | None = None,
    timeout: int | None = None,
) -> list[dict[str, Any]]:
    """Perform a Firecrawl scrape and format the results.

    Args:
        url (str): The URL to scrape.
        firecrawl_api_key (str): The API key for Firecrawl.
        firecrawl_api_url (str, optional): The base URL for the Firecrawl API. Defaults to None.
        formats (list[str], optional): Output formats. Defaults to ['markdown'].
        include_tags (list[str], optional): HTML tags to include. Defaults to None.
        exclude_tags (list[str], optional): HTML tags to exclude. Defaults to None.
        headers (dict[str, str], optional): HTTP headers to use. Defaults to None.
        wait_for (int, optional): Time to wait for page load in milliseconds. Defaults to None.
        timeout (int, optional): Request timeout in milliseconds. Defaults to None.

    Returns:
        list[dict[str, Any]]: A list containing the scraped content.
    """
    try:
        result = _execute_firecrawl_scrape(
            url=url,
            firecrawl_api_key=firecrawl_api_key,
            firecrawl_api_url=firecrawl_api_url,
            formats=formats,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            headers=headers,
            wait_for=wait_for,
            timeout=timeout,
        )

        # Format the result to match expected output
        formatted_result = {
            "title": result.get("metadata", {}).get("title", ""),
            "url": url,
            "content": result.get("markdown", result.get("html", "")),
            "metadata": result.get("metadata", {}),
        }

        return [formatted_result]
    except Exception as e:
        logger.error(f"Firecrawl scrape failed: {e}")
        return []


def _firecrawl_crawl(
    url: str,
    firecrawl_api_key: str,
    firecrawl_api_url: str | None = None,
    limit: int = 5,
    formats: list[str] | None = None,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    max_depth: int | None = None,
    allow_backward_crawling: bool = False,
    allow_external_content_links: bool = False,
) -> list[dict[str, Any]]:
    """Perform a Firecrawl crawl and format the results.

    Args:
        url (str): The starting URL to crawl.
        firecrawl_api_key (str): The API key for Firecrawl.
        firecrawl_api_url (str, optional): The base URL for the Firecrawl API. Defaults to None.
        limit (int, optional): Maximum number of pages to crawl. Defaults to 5.
        formats (list[str], optional): Output formats. Defaults to ['markdown'].
        include_paths (list[str], optional): URL patterns to include. Defaults to None.
        exclude_paths (list[str], optional): URL patterns to exclude. Defaults to None.
        max_depth (int, optional): Maximum crawl depth. Defaults to None.
        allow_backward_crawling (bool, optional): Allow crawling backward links. Defaults to False.
        allow_external_content_links (bool, optional): Allow external links. Defaults to False.

    Returns:
        list[dict[str, Any]]: A list of crawled pages with content.
    """
    try:
        result = _execute_firecrawl_crawl(
            url=url,
            firecrawl_api_key=firecrawl_api_key,
            firecrawl_api_url=firecrawl_api_url,
            limit=limit,
            formats=formats,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            max_depth=max_depth,
            allow_backward_crawling=allow_backward_crawling,
            allow_external_content_links=allow_external_content_links,
        )

        # Format the results
        formatted_results = []
        data = result.get("data", [])

        for item in data:
            formatted_result = {
                "title": item.get("metadata", {}).get("title", ""),
                "url": item.get("metadata", {}).get("sourceURL", ""),
                "content": item.get("markdown", item.get("html", "")),
                "metadata": item.get("metadata", {}),
            }
            formatted_results.append(formatted_result)

        return formatted_results
    except Exception as e:
        logger.error(f"Firecrawl crawl failed: {e}")
        return []


def _firecrawl_map(
    url: str,
    firecrawl_api_key: str,
    firecrawl_api_url: str | None = None,
    search: str | None = None,
    ignore_sitemap: bool = False,
    include_subdomains: bool = False,
    limit: int = 5000,
) -> list[dict[str, Any]]:
    """Perform a Firecrawl map operation and format the results.

    Args:
        url (str): The website URL to map.
        firecrawl_api_key (str): The API key for Firecrawl.
        firecrawl_api_url (str, optional): The base URL for the Firecrawl API. Defaults to None.
        search (str, optional): Search term to filter URLs. Defaults to None.
        ignore_sitemap (bool, optional): Whether to ignore the sitemap. Defaults to False.
        include_subdomains (bool, optional): Whether to include subdomains. Defaults to False.
        limit (int, optional): Maximum number of URLs to return. Defaults to 5000.

    Returns:
        list[dict[str, Any]]: A list of URLs found on the website.
    """
    try:
        result = _execute_firecrawl_map(
            url=url,
            firecrawl_api_key=firecrawl_api_key,
            firecrawl_api_url=firecrawl_api_url,
            search=search,
            ignore_sitemap=ignore_sitemap,
            include_subdomains=include_subdomains,
            limit=limit,
        )

        # Format the results
        formatted_results = []
        links = result.get("links", [])

        for link in links:
            formatted_result = {
                "url": link,
                "title": "",  # Map operation doesn't provide titles
                "content": "",  # Map operation doesn't provide content
            }
            formatted_results.append(formatted_result)

        return formatted_results
    except Exception as e:
        logger.error(f"Firecrawl map failed: {e}")
        return []


def _firecrawl_search(
    query: str,
    firecrawl_api_key: str,
    firecrawl_api_url: str | None = None,
    limit: int = 5,
    tbs: str | None = None,
    filter: str | None = None,
    lang: str = "en",
    country: str = "us",
    location: str | None = None,
    timeout: int | None = None,
) -> list[dict[str, Any]]:
    """Perform a Firecrawl search and format the results.

    Args:
        query (str): The search query string.
        firecrawl_api_key (str): The API key for Firecrawl.
        firecrawl_api_url (str, optional): The base URL for the Firecrawl API. Defaults to None.
        limit (int, optional): Maximum number of results to return. Defaults to 5.
        tbs (str, optional): Time filter (e.g., "qdr:d" for past day). Defaults to None.
        filter (str, optional): Custom result filter. Defaults to None.
        lang (str, optional): Language code. Defaults to "en".
        country (str, optional): Country code. Defaults to "us".
        location (str, optional): Geo-targeting location. Defaults to None.
        timeout (int, optional): Request timeout in milliseconds. Defaults to None.

    Returns:
        list[dict[str, Any]]: A list of search results with content.
    """
    try:
        result = _execute_firecrawl_search(
            query=query,
            firecrawl_api_key=firecrawl_api_key,
            firecrawl_api_url=firecrawl_api_url,
            limit=limit,
            tbs=tbs,
            filter=filter,
            lang=lang,
            country=country,
            location=location,
            timeout=timeout,
        )

        # Format the results
        formatted_results = []
        data = result.get("data", [])

        for item in data:
            formatted_result = {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("markdown", item.get("html", "")),
                "description": item.get("description", ""),
                "metadata": item.get("metadata", {}),
            }
            formatted_results.append(formatted_result)

        return formatted_results
    except Exception as e:
        logger.error(f"Firecrawl search failed: {e}")
        return []


def _firecrawl_deep_research(
    query: str,
    firecrawl_api_key: str,
    firecrawl_api_url: str | None = None,
    max_depth: int = 7,
    time_limit: int = 270,
    max_urls: int = 20,
    analysis_prompt: str | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Perform a Firecrawl deep research operation and format the results.

    Args:
        query (str): The research query or topic to investigate.
        firecrawl_api_key (str): The API key for Firecrawl.
        firecrawl_api_url (str, optional): The base URL for the Firecrawl API. Defaults to None.
        max_depth (int, optional): Maximum depth of research exploration. Defaults to 7.
        time_limit (int, optional): Time limit in seconds for research. Defaults to 270.
        max_urls (int, optional): Maximum number of URLs to process. Defaults to 20.
        analysis_prompt (str, optional): Custom prompt for analysis. Defaults to None.
        system_prompt (str, optional): Custom system prompt. Defaults to None.

    Returns:
        dict[str, Any]: The deep research result with analysis, sources, and summaries.
    """
    try:
        result = _execute_firecrawl_deep_research(
            query=query,
            firecrawl_api_key=firecrawl_api_key,
            firecrawl_api_url=firecrawl_api_url,
            max_depth=max_depth,
            time_limit=time_limit,
            max_urls=max_urls,
            analysis_prompt=analysis_prompt,
            system_prompt=system_prompt,
        )

        # Format the result - deep research returns a comprehensive analysis
        formatted_result = {
            "query": query,
            "status": result.get("status", ""),
            "data": result.get("data", {}),
            "sources": result.get("sources", []),
            "summaries": result.get("summaries", []),
            "activities": result.get("activities", []),
            "success": result.get("success", False),
            "error": result.get("error", ""),
        }

        return formatted_result
    except Exception as e:
        logger.error(f"Firecrawl deep research failed: {e}")
        return {
            "query": query,
            "status": "failed",
            "data": {},
            "sources": [],
            "summaries": [],
            "activities": [],
            "success": False,
            "error": str(e),
        }


@export_module("autogen.tools.experimental")
class FirecrawlTool(Tool):
    """FirecrawlTool is a tool that uses the Firecrawl API to scrape, crawl, map, search, and research websites.

    This tool allows agents to leverage Firecrawl for web content extraction, discovery, and research.
    It requires a Firecrawl API key, which can be provided during initialization or set as
    an environment variable `FIRECRAWL_API_KEY`.

    The tool provides five main functionalities:
    - Scrape: Extract content from a single URL
    - Crawl: Recursively crawl a website starting from a URL
    - Map: Discover URLs from a website
    - Search: Search the web for content using Firecrawl's search capabilities
    - Deep Research: Perform comprehensive research on a topic with analysis and insights

    Attributes:
        firecrawl_api_key (str): The API key used for authenticating with the Firecrawl API.
        firecrawl_api_url (str, optional): The base URL for the Firecrawl API. Defaults to None.
    """

    def __init__(
        self,
        *,
        llm_config: LLMConfig | dict[str, Any] | None = None,
        firecrawl_api_key: str | None = None,
        firecrawl_api_url: str | None = None,
    ):
        """Initializes the FirecrawlTool.

        Args:
            llm_config (Optional[Union[LLMConfig, dict[str, Any]]]): LLM configuration. (Currently unused but kept for potential future integration).
            firecrawl_api_key (Optional[str]): The API key for the Firecrawl API. If not provided,
                it attempts to read from the `FIRECRAWL_API_KEY` environment variable.
            firecrawl_api_url (Optional[str]): The base URL for the Firecrawl API. If not provided,
                it attempts to read from the `FIRECRAWL_API_URL` environment variable, or defaults
                to the public Firecrawl API. Use this parameter to connect to self-hosted Firecrawl instances.

        Raises:
            ValueError: If `firecrawl_api_key` is not provided either directly or via the environment variable.
        """
        self.firecrawl_api_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
        self.firecrawl_api_url = firecrawl_api_url or os.getenv("FIRECRAWL_API_URL")

        if self.firecrawl_api_key is None:
            raise ValueError(
                "firecrawl_api_key must be provided either as an argument or via FIRECRAWL_API_KEY env var"
            )

        def firecrawl_scrape(
            url: Annotated[str, "The URL to scrape."],
            firecrawl_api_key: Annotated[str | None, Depends(on(self.firecrawl_api_key))],
            firecrawl_api_url: Annotated[str | None, Depends(on(self.firecrawl_api_url))],
            formats: Annotated[list[str] | None, "Output formats (e.g., ['markdown', 'html'])"] = None,
            include_tags: Annotated[list[str] | None, "HTML tags to include"] = None,
            exclude_tags: Annotated[list[str] | None, "HTML tags to exclude"] = None,
            headers: Annotated[dict[str, str] | None, "HTTP headers to use"] = None,
            wait_for: Annotated[int | None, "Time to wait for page load in milliseconds"] = None,
            timeout: Annotated[int | None, "Request timeout in milliseconds"] = None,
        ) -> list[dict[str, Any]]:
            """Scrapes a single URL and returns the content.

            Args:
                url: The URL to scrape.
                firecrawl_api_key: The API key for Firecrawl (injected dependency).
                firecrawl_api_url: The base URL for the Firecrawl API (injected dependency).
                formats: Output formats (e.g., ['markdown', 'html']). Defaults to ['markdown'].
                include_tags: HTML tags to include. Defaults to None.
                exclude_tags: HTML tags to exclude. Defaults to None.
                headers: HTTP headers to use. Defaults to None.
                wait_for: Time to wait for page load in milliseconds. Defaults to None.
                timeout: Request timeout in milliseconds. Defaults to None.

            Returns:
                A list containing the scraped content with title, url, content, and metadata.

            Raises:
                ValueError: If the Firecrawl API key is not available.
            """
            if firecrawl_api_key is None:
                raise ValueError("Firecrawl API key is missing.")
            return _firecrawl_scrape(
                url=url,
                firecrawl_api_key=firecrawl_api_key,
                firecrawl_api_url=firecrawl_api_url,
                formats=formats,
                include_tags=include_tags,
                exclude_tags=exclude_tags,
                headers=headers,
                wait_for=wait_for,
                timeout=timeout,
            )

        def firecrawl_crawl(
            url: Annotated[str, "The starting URL to crawl."],
            firecrawl_api_key: Annotated[str | None, Depends(on(self.firecrawl_api_key))],
            firecrawl_api_url: Annotated[str | None, Depends(on(self.firecrawl_api_url))],
            limit: Annotated[int, "Maximum number of pages to crawl"] = 5,
            formats: Annotated[list[str] | None, "Output formats (e.g., ['markdown', 'html'])"] = None,
            include_paths: Annotated[list[str] | None, "URL patterns to include"] = None,
            exclude_paths: Annotated[list[str] | None, "URL patterns to exclude"] = None,
            max_depth: Annotated[int | None, "Maximum crawl depth"] = None,
            allow_backward_crawling: Annotated[bool | None, "Allow crawling backward links"] = False,
            allow_external_content_links: Annotated[bool | None, "Allow external links"] = False,
        ) -> list[dict[str, Any]]:
            """Crawls a website starting from a URL and returns the content from multiple pages.

            Args:
                url: The starting URL to crawl.
                firecrawl_api_key: The API key for Firecrawl (injected dependency).
                firecrawl_api_url: The base URL for the Firecrawl API (injected dependency).
                limit: Maximum number of pages to crawl. Defaults to 5.
                formats: Output formats (e.g., ['markdown', 'html']). Defaults to ['markdown'].
                include_paths: URL patterns to include. Defaults to None.
                exclude_paths: URL patterns to exclude. Defaults to None.
                max_depth: Maximum crawl depth. Defaults to None.
                allow_backward_crawling: Allow crawling backward links. Defaults to False.
                allow_external_content_links: Allow external links. Defaults to False.

            Returns:
                A list of crawled pages with title, url, content, and metadata for each page.

            Raises:
                ValueError: If the Firecrawl API key is not available.
            """
            if firecrawl_api_key is None:
                raise ValueError("Firecrawl API key is missing.")
            return _firecrawl_crawl(
                url=url,
                firecrawl_api_key=firecrawl_api_key,
                firecrawl_api_url=firecrawl_api_url,
                limit=limit,
                formats=formats,
                include_paths=include_paths,
                exclude_paths=exclude_paths,
                max_depth=max_depth,
                allow_backward_crawling=allow_backward_crawling or False,
                allow_external_content_links=allow_external_content_links or False,
            )

        def firecrawl_map(
            url: Annotated[str, "The website URL to map."],
            firecrawl_api_key: Annotated[str | None, Depends(on(self.firecrawl_api_key))],
            firecrawl_api_url: Annotated[str | None, Depends(on(self.firecrawl_api_url))],
            search: Annotated[str | None, "Search term to filter URLs"] = None,
            ignore_sitemap: Annotated[bool | None, "Whether to ignore the sitemap"] = False,
            include_subdomains: Annotated[bool | None, "Whether to include subdomains"] = False,
            limit: Annotated[int, "Maximum number of URLs to return"] = 5000,
        ) -> list[dict[str, Any]]:
            """Maps a website to discover URLs.

            Args:
                url: The website URL to map.
                firecrawl_api_key: The API key for Firecrawl (injected dependency).
                firecrawl_api_url: The base URL for the Firecrawl API (injected dependency).
                search: Search term to filter URLs. Defaults to None.
                ignore_sitemap: Whether to ignore the sitemap. Defaults to False.
                include_subdomains: Whether to include subdomains. Defaults to False.
                limit: Maximum number of URLs to return. Defaults to 5000.

            Returns:
                A list of URLs found on the website.

            Raises:
                ValueError: If the Firecrawl API key is not available.
            """
            if firecrawl_api_key is None:
                raise ValueError("Firecrawl API key is missing.")
            return _firecrawl_map(
                url=url,
                firecrawl_api_key=firecrawl_api_key,
                firecrawl_api_url=firecrawl_api_url,
                search=search,
                ignore_sitemap=ignore_sitemap or False,
                include_subdomains=include_subdomains or False,
                limit=limit,
            )

        def firecrawl_search(
            query: Annotated[str, "The search query string."],
            firecrawl_api_key: Annotated[str | None, Depends(on(self.firecrawl_api_key))],
            firecrawl_api_url: Annotated[str | None, Depends(on(self.firecrawl_api_url))],
            limit: Annotated[int, "Maximum number of results to return"] = 5,
            tbs: Annotated[str | None, "Time filter (e.g., 'qdr:d' for past day)"] = None,
            filter: Annotated[str | None, "Custom result filter"] = None,
            lang: Annotated[str | None, "Language code"] = "en",
            country: Annotated[str | None, "Country code"] = "us",
            location: Annotated[str | None, "Geo-targeting location"] = None,
            timeout: Annotated[int | None, "Request timeout in milliseconds"] = None,
        ) -> list[dict[str, Any]]:
            """Executes a search operation using the Firecrawl API.

            Args:
                query: The search query string.
                firecrawl_api_key: The API key for Firecrawl (injected dependency).
                firecrawl_api_url: The base URL for the Firecrawl API (injected dependency).
                limit: Maximum number of results to return. Defaults to 5.
                tbs: Time filter (e.g., "qdr:d" for past day). Defaults to None.
                filter: Custom result filter. Defaults to None.
                lang: Language code. Defaults to "en".
                country: Country code. Defaults to "us".
                location: Geo-targeting location. Defaults to None.
                timeout: Request timeout in milliseconds. Defaults to None.

            Returns:
                A list of search results with title, url, content, and metadata.

            Raises:
                ValueError: If the Firecrawl API key is not available.
            """
            if firecrawl_api_key is None:
                raise ValueError("Firecrawl API key is missing.")
            return _firecrawl_search(
                query=query,
                firecrawl_api_key=firecrawl_api_key,
                firecrawl_api_url=firecrawl_api_url,
                limit=limit,
                tbs=tbs,
                filter=filter,
                lang=lang or "en",
                country=country or "us",
                location=location,
                timeout=timeout,
            )

        def firecrawl_deep_research(
            query: Annotated[str, "The research query or topic to investigate."],
            firecrawl_api_key: Annotated[str | None, Depends(on(self.firecrawl_api_key))],
            firecrawl_api_url: Annotated[str | None, Depends(on(self.firecrawl_api_url))],
            max_depth: Annotated[int, "Maximum depth of research exploration"] = 7,
            time_limit: Annotated[int, "Time limit in seconds for research"] = 270,
            max_urls: Annotated[int, "Maximum number of URLs to process"] = 20,
            analysis_prompt: Annotated[str | None, "Custom prompt for analysis"] = None,
            system_prompt: Annotated[str | None, "Custom system prompt"] = None,
        ) -> dict[str, Any]:
            """Executes a deep research operation using the Firecrawl API.

            Args:
                query: The research query or topic to investigate.
                firecrawl_api_key: The API key for Firecrawl (injected dependency).
                firecrawl_api_url: The base URL for the Firecrawl API (injected dependency).
                max_depth: Maximum depth of research exploration. Defaults to 7.
                time_limit: Time limit in seconds for research. Defaults to 270.
                max_urls: Maximum number of URLs to process. Defaults to 20.
                analysis_prompt: Custom prompt for analysis. Defaults to None.
                system_prompt: Custom system prompt. Defaults to None.

            Returns:
                The deep research result from Firecrawl.

            Raises:
                ValueError: If the Firecrawl API key is not available.
            """
            if firecrawl_api_key is None:
                raise ValueError("Firecrawl API key is missing.")
            return _execute_firecrawl_deep_research(
                query=query,
                firecrawl_api_key=firecrawl_api_key,
                firecrawl_api_url=firecrawl_api_url,
                max_depth=max_depth,
                time_limit=time_limit,
                max_urls=max_urls,
                analysis_prompt=analysis_prompt,
                system_prompt=system_prompt,
            )

        # Default to scrape functionality for the main tool
        super().__init__(
            name="firecrawl_scrape",
            description="Use the Firecrawl API to scrape content from a single URL.",
            func_or_tool=firecrawl_scrape,
        )

        # Store additional methods for manual access
        self.scrape = firecrawl_scrape
        self.crawl = firecrawl_crawl
        self.map = firecrawl_map
        self.search = firecrawl_search
        self.deep_research = firecrawl_deep_research
