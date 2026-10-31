# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""SearchApi search extension for AG2.

SearchApi serves live Google web, Google News and Google Scholar results
from one REST endpoint, switched with the ``engine`` parameter. This
extension wraps the three engines as agent tools.

Maintainer: SearchApi (Gabs Ferreira, @gabsferreira)
Docs: https://docs.ag2.ai/docs/user-guide/extensions/tools/search/searchapi/
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any

import httpx
from pydantic import Field

from ag2.annotations import Context, Variable
from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool

# Every engine answers on the same path; ``engine`` selects the vertical.
_SEARCH_PATH = "/api/v1/search"
# Google ships web and news links as ``google.com/goto`` redirects; SearchApi
# follows them when asked, so the model gets the destination URL.
_RESOLVED_LINKS = "resolved"
_USER_AGENT = "ag2"


@dataclass(slots=True)
class SearchApiWebResult:
    title: str
    link: str
    snippet: str = ""
    position: int | None = None


@dataclass(slots=True)
class SearchApiWebSearchResponse:
    query: str
    results: list[SearchApiWebResult] = field(default_factory=list)


@dataclass(slots=True)
class SearchApiNewsResult:
    title: str
    link: str
    snippet: str = ""
    source: str = ""
    date: str = ""


@dataclass(slots=True)
class SearchApiNewsSearchResponse:
    query: str
    results: list[SearchApiNewsResult] = field(default_factory=list)


@dataclass(slots=True)
class SearchApiScholarResult:
    title: str
    link: str
    snippet: str = ""
    publication: str = ""
    authors: list[str] = field(default_factory=list)
    pdf_link: str = ""
    citations: int | None = None


@dataclass(slots=True)
class SearchApiScholarSearchResponse:
    query: str
    results: list[SearchApiScholarResult] = field(default_factory=list)


def _text(raw: Any, key: str) -> str:
    """Return ``raw[key]`` when *raw* is a mapping holding a string there, else ``""``."""
    if not isinstance(raw, dict):
        return ""
    value = raw.get(key)
    return value if isinstance(value, str) else ""


def _int(value: Any) -> int | None:
    """Return *value* when it is a real integer, else ``None`` (``bool`` is not an integer here)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _items(raw: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Return the mappings inside the list at ``raw[key]``, skipping anything else."""
    items = raw.get(key)
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _nested(raw: Any, *keys: str) -> Any:
    """Walk *keys* through nested mappings, returning ``None`` as soon as one is missing."""
    for key in keys:
        if not isinstance(raw, dict):
            return None
        raw = raw.get(key)
    return raw


def _names(raw: dict[str, Any], key: str) -> list[str]:
    """Return the non-empty ``name`` strings of the mappings listed at ``raw[key]``."""
    return [name for item in _items(raw, key) if (name := _text(item, "name"))]


def _pdf_link(raw: Any) -> str:
    """Return the ``resource.link`` of a Scholar row, or ``""`` when Google serves a ``javascript:`` stub there."""
    link = _text(raw, "link")
    return "" if "javascript:" in link.lower() else link


def _cap(raw: dict[str, Any], limit: Any) -> list[dict[str, Any]]:
    """Return the ``organic_results`` rows, cut to *limit* when it is a positive integer."""
    rows = _items(raw, "organic_results")
    count = _int(limit)
    return rows[:count] if count is not None and count > 0 else rows


async def _get(
    client_kwargs: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Run one GET against the SearchApi search endpoint.

    Args:
        client_kwargs: Keyword arguments for the ``httpx.AsyncClient`` to open.
        params: Query parameters. Entries with a ``None`` value are dropped.

    Returns:
        The decoded JSON object, or an empty dict when the payload is not an object.

    Raises:
        httpx.HTTPStatusError: If SearchApi answers with a non-2xx status code.
    """
    async with httpx.AsyncClient(**client_kwargs) as client:
        response = await client.get(
            _SEARCH_PATH,
            params={key: value for key, value in params.items() if value is not None},
        )
        response.raise_for_status()
        raw = response.json()

    return raw if isinstance(raw, dict) else {}


class SearchApiSearchToolkit(Toolkit):
    """Toolkit that searches Google web, news and scholar results through the SearchApi REST API.

    Passing the toolkit to an agent registers ``searchapi_web_search``,
    ``searchapi_news_search`` and ``searchapi_scholar_search``. To use a
    subset, or to customise per-tool defaults, call the factory methods and
    pass the returned tools to the agent::

        toolkit = SearchApiSearchToolkit(api_key=...)

        # all three tools
        agent = Agent("a", config=config, tools=[toolkit])

        # only web search, with custom defaults
        agent = Agent("a", config=config, tools=[toolkit.web(num=5, gl="us")])

    Optional defaults can be fixed when the toolkit or tool is constructed,
    or supplied through AG2 ``Variable`` values at runtime.

    A SearchApi ``api_key`` is required.
    """

    __slots__ = ("_api_key", "_base_url", "_timeout", "_proxy", "_verify")

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://www.searchapi.io",
        timeout: float = 60.0,
        proxy: str | None = None,
        verify: bool = True,
        num: int | Variable | None = None,
        gl: str | Variable | None = None,
        hl: str | Variable | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        """Build the toolkit and its three default tools.

        Args:
            api_key: SearchApi API key, sent as a ``Authorization: Bearer`` header.
            base_url: SearchApi root. Trailing slashes are stripped.
            timeout: Per-request timeout in seconds.
            proxy: Proxy URL for the outgoing HTTP connection.
            verify: Whether to verify the server's TLS certificate.
            num: Default number of results. Scholar sends it to the API (Google
                Scholar serves up to 20 per page); web and news cut the page
                client-side, because Google no longer honours ``num`` there and
                answers with one page of results.
            gl: Default Google country code, forwarded as a query parameter to
                the web and news tools.
            hl: Default Google interface language, forwarded as a query parameter.
            middleware: Middleware applied to every tool in the toolkit.

        Raises:
            ValueError: If *api_key* is empty.
        """
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._proxy = proxy
        self._verify = verify

        super().__init__(
            self.web(num=num, gl=gl, hl=hl),
            self.news(num=num, gl=gl, hl=hl),
            self.scholar(num=num, hl=hl),
            name="searchapi_search_toolkit",
            middleware=middleware,
        )

    def web(
        self,
        *,
        num: int | Variable | None = None,
        gl: str | Variable | None = None,
        hl: str | Variable | None = None,
        name: str = "searchapi_web_search",
        description: str = (
            "Search the web with Google through SearchApi. Returns ranked results with titles, snippets, and URLs."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the Google web search tool.

        Args:
            num: Maximum number of results to return. Google serves one page of
                results regardless of ``num``, so the page is cut client-side.
            gl: Google country code, forwarded as a query parameter.
            hl: Google interface language, forwarded as a query parameter.
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A ``FunctionTool`` that queries SearchApi's ``google`` engine with
            ``link=resolved``, so each ``link`` is the destination URL rather
            than Google's ``google.com/goto`` redirect.
        """
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def searchapi_web_search(
            query: Annotated[str, Field(description="The web search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Search Google web results through SearchApi."""
            limit = resolve_variable(num, ctx, param_name="num")
            raw = await _get(
                client_kwargs,
                {
                    "engine": "google",
                    "q": query,
                    "link": _RESOLVED_LINKS,
                    "gl": resolve_variable(gl, ctx, param_name="gl"),
                    "hl": resolve_variable(hl, ctx, param_name="hl"),
                },
            )
            return ToolResult(
                SearchApiWebSearchResponse(
                    query=query,
                    results=[
                        SearchApiWebResult(
                            title=_text(item, "title"),
                            link=_text(item, "link"),
                            snippet=_text(item, "snippet"),
                            position=_int(item.get("position")),
                        )
                        for item in _cap(raw, limit)
                    ],
                )
            )

        return searchapi_web_search

    def news(
        self,
        *,
        num: int | Variable | None = None,
        gl: str | Variable | None = None,
        hl: str | Variable | None = None,
        name: str = "searchapi_news_search",
        description: str = (
            "Search Google News through SearchApi. Returns recent articles with titles, sources, publish dates, "
            "snippets, and URLs."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the Google News search tool.

        Args:
            num: Maximum number of articles to return. Google News serves one
                page of results regardless of ``num``, so the page is cut
                client-side.
            gl: Google country code, forwarded as a query parameter.
            hl: Google interface language, forwarded as a query parameter.
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A ``FunctionTool`` that queries SearchApi's ``google_news`` engine
            with ``link=resolved``, so each ``link`` is the publisher's URL
            rather than Google's ``google.com/goto`` redirect.
        """
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def searchapi_news_search(
            query: Annotated[str, Field(description="The news search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Search Google News through SearchApi."""
            limit = resolve_variable(num, ctx, param_name="num")
            raw = await _get(
                client_kwargs,
                {
                    "engine": "google_news",
                    "q": query,
                    "link": _RESOLVED_LINKS,
                    "gl": resolve_variable(gl, ctx, param_name="gl"),
                    "hl": resolve_variable(hl, ctx, param_name="hl"),
                },
            )
            return ToolResult(
                SearchApiNewsSearchResponse(
                    query=query,
                    results=[
                        SearchApiNewsResult(
                            title=_text(item, "title"),
                            link=_text(item, "link"),
                            snippet=_text(item, "snippet"),
                            source=_text(item, "source"),
                            date=_text(item, "date"),
                        )
                        for item in _cap(raw, limit)
                    ],
                )
            )

        return searchapi_news_search

    def scholar(
        self,
        *,
        num: int | Variable | None = None,
        hl: str | Variable | None = None,
        name: str = "searchapi_scholar_search",
        description: str = (
            "Search Google Scholar through SearchApi. Returns academic articles with titles, authors, publication "
            "details, citation counts, and URLs."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the Google Scholar search tool.

        Args:
            num: Number of results to request, forwarded as a query parameter
                (Google Scholar serves up to 20 per page).
            hl: Google interface language, forwarded as a query parameter.
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A ``FunctionTool`` that queries SearchApi's ``google_scholar`` engine.
        """
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def searchapi_scholar_search(
            query: Annotated[str, Field(description="The academic search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Search Google Scholar through SearchApi."""
            raw = await _get(
                client_kwargs,
                {
                    "engine": "google_scholar",
                    "q": query,
                    "num": resolve_variable(num, ctx, param_name="num"),
                    "hl": resolve_variable(hl, ctx, param_name="hl"),
                },
            )
            return ToolResult(
                SearchApiScholarSearchResponse(
                    query=query,
                    results=[
                        SearchApiScholarResult(
                            title=_text(item, "title"),
                            link=_text(item, "link"),
                            snippet=_text(item, "snippet"),
                            publication=_text(item, "publication"),
                            authors=_names(item, "authors"),
                            pdf_link=_pdf_link(item.get("resource")),
                            citations=_int(_nested(item, "inline_links", "cited_by", "total")),
                        )
                        for item in _items(raw, "organic_results")
                    ],
                )
            )

        return searchapi_scholar_search

    def _client_kwargs(self) -> dict[str, Any]:
        """Snapshot the connection settings as ``httpx.AsyncClient`` keyword arguments."""
        return {
            "base_url": self._base_url,
            "headers": {
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
                "User-Agent": _USER_AGENT,
            },
            "timeout": self._timeout,
            "proxy": self._proxy,
            "verify": self._verify,
        }
