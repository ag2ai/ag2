# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Firecrawl search and scrape extension for AG2.

Provides a toolkit that lets agents search the web with Firecrawl Search,
optionally getting each result's page content as markdown in the same call,
and read a single URL as clean markdown, HTML, or links with Firecrawl Scrape.

Maintainer: JuampiHernandez
Docs: https://docs.ag2.ai/docs/user-guide/extensions/tools/search/firecrawl/
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypeAlias
from urllib.parse import urlparse

from firecrawl.v2.client_async import AsyncFirecrawlClient
from firecrawl.v2.types import Document
from pydantic import Field

from ag2.annotations import Context, Variable
from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool

SearchSource: TypeAlias = Literal["web", "news"]
SearchCategory: TypeAlias = Literal["github", "pdf"]
ScrapeFormat: TypeAlias = Literal["markdown", "html", "links"]
Proxy: TypeAlias = Literal["basic", "stealth", "enhanced", "auto"]

_ORIGIN = "ag2"
_SAFE_URL_SCHEMES = {"http", "https"}


def _safe_url(url: str) -> bool:
    return urlparse(url).scheme.lower() in _SAFE_URL_SCHEMES


def _as_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


@dataclass(slots=True)
class FirecrawlSearchResult:
    url: str
    title: str | None = None
    description: str | None = None
    position: int | None = None
    markdown: str | None = None


@dataclass(slots=True)
class FirecrawlNewsResult:
    url: str
    title: str | None = None
    snippet: str | None = None
    date: str | None = None
    position: int | None = None
    markdown: str | None = None


@dataclass(slots=True)
class FirecrawlSearchResponse:
    query: str
    web: list[FirecrawlSearchResult] = field(default_factory=list)
    news: list[FirecrawlNewsResult] = field(default_factory=list)


@dataclass(slots=True)
class FirecrawlScrapeResult:
    url: str
    title: str | None = None
    description: str | None = None
    language: str | None = None
    status_code: int | None = None
    markdown: str | None = None
    html: str | None = None
    links: list[str] = field(default_factory=list)


class FirecrawlToolkit(Toolkit):
    """Toolkit that exposes Firecrawl Search and Scrape as agent tools.

    The two tools mirror Firecrawl's public APIs:
      - ``firecrawl_search``: web and news search; with ``scrape_results=True``
        each result also carries the main content of its page as markdown
      - ``firecrawl_scrape``: read one URL as markdown, HTML, or a list of links

    By default, passing the whole toolkit to an agent registers both tools.
    To use a subset, or to customise per-tool parameters, call the factory
    methods directly and pass the returned tools to the agent::

        toolkit = FirecrawlToolkit(api_key=...)

        # both tools
        agent = Agent("a", config=config, tools=[toolkit])

        # only search, returning page markdown for each result
        agent = Agent("a", config=config, tools=[toolkit.search(limit=3, scrape_results=True)])

    The Firecrawl SDK reads ``FIRECRAWL_API_KEY`` from the environment when
    ``api_key`` is omitted. The client is created when a tool runs, so building
    the toolkit never requires a key.

    ``http_timeout`` (seconds) bounds each HTTP request made by the SDK client.
    The ``timeout`` options on ``search()`` and ``scrape()`` are Firecrawl API
    timeouts in milliseconds.
    """

    __slots__ = ("_api_key", "_api_url", "_http_timeout", "_max_retries")

    def __init__(
        self,
        api_key: str | None = None,
        *,
        api_url: str | None = None,
        http_timeout: float | None = None,
        max_retries: int | None = None,
        limit: int | Variable | None = None,
        scrape_results: bool | Variable | None = None,
        formats: Sequence[ScrapeFormat] | Variable | None = None,
        only_main_content: bool | Variable | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        self._api_key = api_key
        self._api_url = api_url
        self._http_timeout = http_timeout
        self._max_retries = max_retries

        super().__init__(
            self.search(limit=limit, scrape_results=scrape_results),
            self.scrape(formats=formats, only_main_content=only_main_content),
            name="firecrawl_toolkit",
            middleware=middleware,
        )

    def search(
        self,
        *,
        limit: int | Variable | None = None,
        sources: Sequence[SearchSource] | Variable | None = None,
        categories: Sequence[SearchCategory] | Variable | None = None,
        include_domains: Sequence[str] | Variable | None = None,
        exclude_domains: Sequence[str] | Variable | None = None,
        tbs: str | Variable | None = None,
        location: str | Variable | None = None,
        country: str | Variable | None = None,
        ignore_invalid_urls: bool | Variable | None = None,
        timeout: int | Variable | None = None,
        scrape_results: bool | Variable | None = None,
        scrape_only_main_content: bool | Variable | None = None,
        scrape_max_age: int | Variable | None = None,
        name: str = "firecrawl_search",
        description: str = (
            "Search the web using Firecrawl. Returns ranked results with URLs, titles, and descriptions. "
            "Results may also include the main content of each page as markdown."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Create the ``firecrawl_search`` tool.

        Args:
            limit: Maximum number of results per source. Firecrawl returns 5 when omitted.
            sources: Result types to return: ``"web"``, ``"news"``.
                Firecrawl searches ``"web"`` when omitted.
            categories: Narrow web results to ``"github"`` or ``"pdf"``.
            include_domains: Only return results from these domains.
                Cannot be combined with ``exclude_domains``.
            exclude_domains: Never return results from these domains.
            tbs: Time filter for web results, for example ``"qdr:d"`` for the past day.
            location: Location to search from, for example ``"Germany"``.
            country: ISO country code, for example ``"US"``.
            ignore_invalid_urls: Drop results whose URLs Firecrawl cannot scrape.
            timeout: Firecrawl API timeout in milliseconds.
            scrape_results: When true, each result includes the main content of
                its page as markdown. Each result is then a full page, so keep
                ``limit`` low to bound the size of the tool result.
            scrape_only_main_content: With ``scrape_results``, drop headers,
                navigation, and footers. Firecrawl defaults to ``True``.
            scrape_max_age: With ``scrape_results``, accept cached pages up to
                this many milliseconds old. ``0`` always fetches fresh pages.
            name: Tool name exposed to the model.
            description: Tool description exposed to the model.
            middleware: Tool middleware applied to this tool.
        """
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def firecrawl_search(
            query: Annotated[str, Field(description="The search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Search the web using Firecrawl and return ranked results."""
            params: dict[str, Any] = {
                "limit": resolve_variable(limit, ctx, param_name="limit"),
                "sources": _as_list(resolve_variable(sources, ctx, param_name="sources")),
                "categories": _as_list(resolve_variable(categories, ctx, param_name="categories")),
                "include_domains": _as_list(resolve_variable(include_domains, ctx, param_name="include_domains")),
                "exclude_domains": _as_list(resolve_variable(exclude_domains, ctx, param_name="exclude_domains")),
                "tbs": resolve_variable(tbs, ctx, param_name="tbs"),
                "location": resolve_variable(location, ctx, param_name="location"),
                "country": resolve_variable(country, ctx, param_name="country"),
                "ignore_invalid_urls": resolve_variable(ignore_invalid_urls, ctx, param_name="ignore_invalid_urls"),
                "timeout": resolve_variable(timeout, ctx, param_name="timeout"),
            }
            if resolve_variable(scrape_results, ctx, param_name="scrape_results"):
                scrape_options: dict[str, Any] = {
                    "formats": ["markdown"],
                    "only_main_content": resolve_variable(
                        scrape_only_main_content, ctx, param_name="scrape_only_main_content"
                    ),
                    "max_age": resolve_variable(scrape_max_age, ctx, param_name="scrape_max_age"),
                }
                params["scrape_options"] = {k: v for k, v in scrape_options.items() if v is not None}
            kwargs = {k: v for k, v in params.items() if v is not None}

            client = AsyncFirecrawlClient(**client_kwargs)
            try:
                raw = await client.search(query, **kwargs)
            finally:
                await client.async_http_client.close()

            return ToolResult(
                FirecrawlSearchResponse(
                    query=query,
                    web=[_web_result(r, i) for i, r in enumerate(raw.web or [], start=1)],
                    news=[_news_result(r, i) for i, r in enumerate(raw.news or [], start=1)],
                )
            )

        return firecrawl_search

    def scrape(
        self,
        *,
        formats: Sequence[ScrapeFormat] | Variable | None = None,
        only_main_content: bool | Variable | None = None,
        include_tags: Sequence[str] | Variable | None = None,
        exclude_tags: Sequence[str] | Variable | None = None,
        headers: Mapping[str, str] | Variable | None = None,
        wait_for: int | Variable | None = None,
        mobile: bool | Variable | None = None,
        proxy: Proxy | Variable | None = None,
        block_ads: bool | Variable | None = None,
        max_age: int | Variable | None = None,
        store_in_cache: bool | Variable | None = None,
        timeout: int | Variable | None = None,
        name: str = "firecrawl_scrape",
        description: str = (
            "Read a single web page using Firecrawl. Returns the page content (markdown by default) "
            "with the page title, description, and status code."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Create the ``firecrawl_scrape`` tool.

        Args:
            formats: Output formats to return: ``"markdown"``, ``"html"``, ``"links"``.
                Defaults to ``["markdown"]``.
            only_main_content: Drop headers, navigation, and footers. Firecrawl
                defaults to ``True``.
            include_tags: Only keep these HTML tags, classes, or ids.
            exclude_tags: Remove these HTML tags, classes, or ids.
            headers: Extra HTTP headers to send with the page request.
            wait_for: Milliseconds to wait for the page to render before reading it.
            mobile: Emulate a mobile device.
            proxy: Proxy type: ``"basic"``, ``"stealth"``, ``"enhanced"``, or ``"auto"``.
            block_ads: Block ads and cookie pop-ups. Firecrawl defaults to ``True``.
            max_age: Accept a cached copy up to this many milliseconds old.
                Firecrawl uses a cached copy up to 2 days old by default;
                ``0`` always fetches a fresh copy.
            store_in_cache: Whether Firecrawl may cache this page.
            timeout: Firecrawl API timeout in milliseconds.
            name: Tool name exposed to the model.
            description: Tool description exposed to the model.
            middleware: Tool middleware applied to this tool.
        """
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def firecrawl_scrape(
            url: Annotated[str, Field(description="The http or https URL of the page to read.")],
            ctx: Context,
        ) -> ToolResult:
            """Scrape one web page using Firecrawl and return its content."""
            if not _safe_url(url):
                return ToolResult({"error": f"Only http/https URLs are supported; rejected: {url!r}"})

            resolved_headers = resolve_variable(headers, ctx, param_name="headers")
            params: dict[str, Any] = {
                "formats": _as_list(resolve_variable(formats, ctx, param_name="formats")) or ["markdown"],
                "only_main_content": resolve_variable(only_main_content, ctx, param_name="only_main_content"),
                "include_tags": _as_list(resolve_variable(include_tags, ctx, param_name="include_tags")),
                "exclude_tags": _as_list(resolve_variable(exclude_tags, ctx, param_name="exclude_tags")),
                "headers": dict(resolved_headers) if resolved_headers is not None else None,
                "wait_for": resolve_variable(wait_for, ctx, param_name="wait_for"),
                "mobile": resolve_variable(mobile, ctx, param_name="mobile"),
                "proxy": resolve_variable(proxy, ctx, param_name="proxy"),
                "block_ads": resolve_variable(block_ads, ctx, param_name="block_ads"),
                "max_age": resolve_variable(max_age, ctx, param_name="max_age"),
                "store_in_cache": resolve_variable(store_in_cache, ctx, param_name="store_in_cache"),
                "timeout": resolve_variable(timeout, ctx, param_name="timeout"),
            }
            kwargs = {k: v for k, v in params.items() if v is not None}

            client = AsyncFirecrawlClient(**client_kwargs)
            try:
                doc = await client.scrape(url, **kwargs)
            finally:
                await client.async_http_client.close()

            metadata = doc.metadata_typed
            return ToolResult(
                FirecrawlScrapeResult(
                    url=metadata.url or metadata.source_url or url,
                    title=metadata.title,
                    description=metadata.description,
                    language=metadata.language,
                    status_code=metadata.status_code,
                    markdown=doc.markdown,
                    html=doc.html,
                    links=list(doc.links or []),
                )
            )

        return firecrawl_scrape

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"api_key": self._api_key, "origin": _ORIGIN}
        if self._api_url is not None:
            kwargs["api_url"] = self._api_url
        if self._http_timeout is not None:
            kwargs["timeout"] = self._http_timeout
        if self._max_retries is not None:
            kwargs["max_retries"] = self._max_retries
        return kwargs


# With scrape options the SDK returns each result as a ``Document``, which keeps
# only the page metadata (no search ``position``, ``date`` or snippet). Results
# keep their rank order, so the list index is used as the position.
# ``item`` is ``SearchResultWeb``/``SearchResultNews`` or ``Document``; firecrawl-py
# ships no type information, so it is typed as ``Any``.
def _web_result(item: Any, rank: int) -> FirecrawlSearchResult:
    if isinstance(item, Document):
        metadata = item.metadata_typed
        return FirecrawlSearchResult(
            url=metadata.url or metadata.source_url or "",
            title=metadata.title,
            description=metadata.description,
            position=rank,
            markdown=item.markdown,
        )
    return FirecrawlSearchResult(
        url=item.url,
        title=item.title,
        description=item.description,
        position=item.position,
    )


def _news_result(item: Any, rank: int) -> FirecrawlNewsResult:
    if isinstance(item, Document):
        metadata = item.metadata_typed
        return FirecrawlNewsResult(
            url=metadata.url or metadata.source_url or "",
            title=metadata.title,
            snippet=metadata.description,
            position=rank,
            markdown=item.markdown,
        )
    return FirecrawlNewsResult(
        url=item.url or "",
        title=item.title,
        snippet=item.snippet,
        date=item.date,
        position=item.position,
    )
