# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""TinyFish search and fetch extension for AG2 Beta.

Provides a toolkit that lets agents query TinyFish Search for ranked web
results and use TinyFish Fetch to extract browser-rendered page content.

Maintainer: pranavjana
Docs: https://docs.ag2.ai/docs/beta/extensions/tools/search/tinyfish
"""

import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypeAlias
from urllib.parse import urlparse

from pydantic import Field
from tinyfish import AsyncTinyFish

from ag2.annotations import Context, Variable
from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool

FetchFormat: TypeAlias = Literal["markdown", "html", "json"]
_API_INTEGRATION = "ag2"
_API_INTEGRATION_ENV_VAR = "TF_API_INTEGRATION"
_SAFE_URL_SCHEMES = {"http", "https"}


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


@dataclass(slots=True)
class TinyFishSearchResult:
    position: int
    site_name: str
    title: str
    snippet: str
    url: str


@dataclass(slots=True)
class TinyFishSearchResponse:
    query: str
    results: list[TinyFishSearchResult] = field(default_factory=list)
    total_results: int = 0


@dataclass(slots=True)
class TinyFishFetchResult:
    url: str
    format: str
    final_url: str | None = None
    title: str | None = None
    description: str | None = None
    language: str | None = None
    author: str | None = None
    published_date: str | None = None
    text: Any | None = None
    links: list[str] = field(default_factory=list)
    image_links: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TinyFishFetchError:
    url: str
    error: str


@dataclass(slots=True)
class TinyFishFetchResponse:
    results: list[TinyFishFetchResult] = field(default_factory=list)
    errors: list[TinyFishFetchError] = field(default_factory=list)


class TinyFishSearchToolkit(Toolkit):
    """Toolkit that exposes TinyFish Search and Fetch APIs as agent tools.

    The two tools mirror TinyFish's public APIs:
      - ``tinyfish_search``: web search with ranked titles, snippets, and URLs
      - ``tinyfish_fetch``: browser-rendered content extraction for up to 10 URLs

    By default, passing the whole toolkit to an agent registers both tools.
    To use a subset, or to customise per-tool parameters, call the factory
    methods directly and pass the returned tools to the agent::

        toolkit = TinyFishSearchToolkit(api_key=...)

        # both tools
        agent = Agent("a", config=config, tools=[toolkit])

        # only search, with custom defaults
        agent = Agent("a", config=config, tools=[toolkit.search(location="US", language="en")])

    The constructor reads ``TINYFISH_API_KEY`` from the environment when
    ``api_key`` is omitted (handled by the underlying TinyFish SDK).
    """

    __slots__ = ("_api_key", "_base_url", "_timeout", "_max_retries")

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        location: str | Variable | None = None,
        language: str | Variable | None = None,
        format: FetchFormat | Variable | None = None,
        links: bool | Variable | None = None,
        image_links: bool | Variable | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries

        super().__init__(
            self.search(location=location, language=language),
            self.fetch(format=format, links=links, image_links=image_links),
            name="tinyfish_search_toolkit",
            middleware=middleware,
        )

    def search(
        self,
        *,
        location: str | Variable | None = None,
        language: str | Variable | None = None,
        name: str = "tinyfish_search",
        description: str = (
            "Search the web using TinyFish. Returns ranked results with titles, snippets, site names, and URLs."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def tinyfish_search(
            query: Annotated[str, Field(description="The search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Search the web using TinyFish and return ranked results."""
            params: dict[str, Any] = {
                "location": resolve_variable(location, ctx, param_name="location"),
                "language": resolve_variable(language, ctx, param_name="language"),
            }
            kwargs = {k: v for k, v in params.items() if v is not None}

            client = AsyncTinyFish(**client_kwargs)
            try:
                raw = await client.search.query(query=query, **kwargs)
            finally:
                await client.close()

            return ToolResult(
                TinyFishSearchResponse(
                    query=raw.query,
                    total_results=raw.total_results,
                    results=[
                        TinyFishSearchResult(
                            position=r.position,
                            site_name=r.site_name,
                            title=r.title,
                            snippet=r.snippet,
                            url=r.url,
                        )
                        for r in raw.results
                    ],
                )
            )

        return tinyfish_search

    def fetch(
        self,
        *,
        format: FetchFormat | Variable | None = None,
        links: bool | Variable | None = None,
        image_links: bool | Variable | None = None,
        name: str = "tinyfish_fetch",
        description: str = (
            "Fetch and extract clean content from URLs using TinyFish. "
            "Renders pages in a browser and returns extracted content plus per-URL errors."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def tinyfish_fetch(
            urls: Annotated[list[str], Field(description="URLs to fetch content for. TinyFish supports 1-10 URLs.")],
            ctx: Context,
        ) -> ToolResult:
            """Fetch web pages using TinyFish and return extracted content."""
            invalid_urls = [url for url in urls if not _safe_url(url)]
            if invalid_urls:
                return ToolResult({"error": f"Only http/https URLs are supported; rejected: {invalid_urls}"})

            params: dict[str, Any] = {
                "format": resolve_variable(format, ctx, param_name="format"),
                "links": resolve_variable(links, ctx, param_name="links"),
                "image_links": resolve_variable(image_links, ctx, param_name="image_links"),
            }
            kwargs = {k: v for k, v in params.items() if v is not None}

            client = AsyncTinyFish(**client_kwargs)
            try:
                with _tinyfish_api_integration():
                    raw = await client.fetch.get_contents(urls=urls, **kwargs)
            finally:
                await client.close()

            return ToolResult(
                TinyFishFetchResponse(
                    results=[
                        TinyFishFetchResult(
                            url=r.url,
                            final_url=r.final_url,
                            title=r.title,
                            description=r.description,
                            language=r.language,
                            author=r.author,
                            published_date=r.published_date,
                            text=r.text,
                            format=r.format,
                            links=list(r.links or []),
                            image_links=list(r.image_links or []),
                        )
                        for r in raw.results
                    ],
                    errors=[TinyFishFetchError(url=e.url, error=e.error) for e in raw.errors],
                )
            )

        return tinyfish_fetch

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout
        if self._max_retries is not None:
            kwargs["max_retries"] = self._max_retries
        return kwargs
