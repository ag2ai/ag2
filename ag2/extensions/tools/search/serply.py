# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Serply search extension for AG2.

Serply exposes Google web, Google News and Google Scholar results through one
REST API. This extension wraps the three endpoints as agent tools.

Maintainer: googio
Docs: https://docs.ag2.ai/docs/user-guide/extensions/tools/search/serply/
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypeAlias

import httpx
from pydantic import Field

from ag2.annotations import Context, Variable
from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool

ProxyLocation: TypeAlias = Literal["EU", "CA", "US", "IE", "GB", "FR", "DE", "SE", "IN", "JP", "KR", "SG", "AU", "BR"]

# Serply sits behind Cloudflare, which blocks httpx's default ``python-httpx/*``
# User-Agent. Unrelated to Serply's own ``X-User-Agent`` header, which Serply
# documents as a desktop/mobile switch but does not currently act on.
_USER_AGENT = "ag2-serply-extension"

# Google News ignores `num` server-side — it answers with its whole feed, seen at
# 49-105 entries whose links are ~284-character opaque redirects. Left uncapped a
# single news call lands ~15k tokens in the model's context, so the cap is applied
# here instead.
_NEWS_MAX_RESULTS = 10


@dataclass(slots=True)
class SerplyWebResult:
    title: str
    link: str
    description: str = ""
    position: int | None = None


@dataclass(slots=True)
class SerplyWebSearchResponse:
    query: str
    results: list[SerplyWebResult] = field(default_factory=list)


@dataclass(slots=True)
class SerplyNewsResult:
    title: str
    link: str
    published: str = ""
    source: str = ""


@dataclass(slots=True)
class SerplyNewsSearchResponse:
    query: str
    results: list[SerplyNewsResult] = field(default_factory=list)


@dataclass(slots=True)
class SerplyScholarResult:
    title: str
    link: str
    authors: str = ""
    pdf_link: str = ""
    citations: int | None = None


@dataclass(slots=True)
class SerplyScholarSearchResponse:
    query: str
    results: list[SerplyScholarResult] = field(default_factory=list)


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


async def _get(
    client_kwargs: dict[str, Any],
    path: str,
    params: dict[str, Any],
    headers: dict[str, Any],
) -> dict[str, Any]:
    """Run one GET against the Serply REST API.

    Args:
        client_kwargs: Keyword arguments for the ``httpx.AsyncClient`` to open.
        path: Endpoint path, resolved against the client's base URL.
        params: Query parameters. Entries with a ``None`` value are dropped.
        headers: Per-request headers. Entries with a ``None`` value are dropped.

    Returns:
        The decoded JSON object, or an empty dict when the payload is not an object.

    Raises:
        httpx.HTTPStatusError: If Serply answers with a non-2xx status code.
    """
    async with httpx.AsyncClient(**client_kwargs) as client:
        response = await client.get(
            path,
            params={key: value for key, value in params.items() if value is not None},
            headers={key: value for key, value in headers.items() if value is not None},
        )
        response.raise_for_status()
        raw = response.json()

    return raw if isinstance(raw, dict) else {}


class SerplySearchToolkit(Toolkit):
    """Toolkit that searches Google web, news and scholar results through the Serply REST API.

    Passing the toolkit to an agent registers ``serply_web_search``,
    ``serply_news_search`` and ``serply_scholar_search``. To use a subset, or
    to customise per-tool defaults, call the factory methods and pass the
    returned tools to the agent::

        toolkit = SerplySearchToolkit(api_key=...)

        # all three tools
        agent = Agent("a", config=config, tools=[toolkit])

        # only web search, with custom defaults
        agent = Agent("a", config=config, tools=[toolkit.web(num=5, gl="us")])

    Optional defaults can be fixed when the toolkit or tool is constructed,
    or supplied through AG2 ``Variable`` values at runtime.

    A Serply ``api_key`` is required.
    """

    __slots__ = ("_api_key", "_base_url", "_timeout", "_proxy", "_verify")

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.serply.io",
        timeout: float = 60.0,
        proxy: str | None = None,
        verify: bool = True,
        num: int | Variable | None = None,
        gl: str | Variable | None = None,
        hl: str | Variable | None = None,
        proxy_location: ProxyLocation | Variable | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        """Build the toolkit and its three default tools.

        Args:
            api_key: Serply API key, sent as the ``X-Api-Key`` header.
            base_url: Serply API root. Trailing slashes are stripped.
            timeout: Per-request timeout in seconds.
            proxy: Proxy URL for the outgoing HTTP connection.
            verify: Whether to verify the server's TLS certificate.
            num: Default number of results. Web and scholar send it to the API;
                news truncates the feed client-side, because Google News ignores
                the parameter and answers with its whole feed.
            gl: Default Google country code, forwarded as a query parameter.
            hl: Default Google interface language, forwarded as a query parameter.
            proxy_location: Default region Serply searches from, sent as the
                ``X-Proxy-Location`` header. This is Serply's documented way to
                target results geographically, and it is best-effort: Serply may
                answer from a nearby region instead of the one asked for.
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
            self.web(num=num, gl=gl, hl=hl, proxy_location=proxy_location),
            self.news(num=num, gl=gl, hl=hl, proxy_location=proxy_location),
            self.scholar(num=num, gl=gl, hl=hl, proxy_location=proxy_location),
            name="serply_search_toolkit",
            middleware=middleware,
        )

    def web(
        self,
        *,
        num: int | Variable | None = None,
        gl: str | Variable | None = None,
        hl: str | Variable | None = None,
        proxy_location: ProxyLocation | Variable | None = None,
        name: str = "serply_web_search",
        description: str = (
            "Search the web with Google through Serply. Returns ranked results with titles, snippets, and URLs."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the Google web search tool.

        Args:
            num: Number of results to request.
            gl: Google country code, forwarded as a query parameter.
            hl: Google interface language, forwarded as a query parameter.
            proxy_location: Region to search from (``X-Proxy-Location`` header).
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A ``FunctionTool`` that queries Serply's ``/v1/search/`` endpoint.
        """
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def serply_web_search(
            query: Annotated[str, Field(description="The web search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Search Google web results through Serply."""
            raw = await _get(
                client_kwargs,
                "/v1/search/",
                {
                    "q": query,
                    "num": resolve_variable(num, ctx, param_name="num"),
                    "gl": resolve_variable(gl, ctx, param_name="gl"),
                    "hl": resolve_variable(hl, ctx, param_name="hl"),
                },
                {"X-Proxy-Location": resolve_variable(proxy_location, ctx, param_name="proxy_location")},
            )
            return ToolResult(
                SerplyWebSearchResponse(
                    query=query,
                    results=[
                        SerplyWebResult(
                            title=_text(item, "title"),
                            link=_text(item, "link"),
                            description=_text(item, "description"),
                            position=_int(item.get("position")),
                        )
                        for item in _items(raw, "results")
                    ],
                )
            )

        return serply_web_search

    def news(
        self,
        *,
        num: int | Variable | None = None,
        gl: str | Variable | None = None,
        hl: str | Variable | None = None,
        proxy_location: ProxyLocation | Variable | None = None,
        name: str = "serply_news_search",
        description: str = (
            "Search Google News through Serply. Returns recent articles with titles, sources, publish dates, and URLs."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the Google News search tool.

        Args:
            num: Maximum number of articles to return. Google News ignores this
                server-side, so the feed is truncated client-side. Defaults to
                ``_NEWS_MAX_RESULTS``.
            gl: Google country code, forwarded as a query parameter.
            hl: Google interface language, forwarded as a query parameter.
            proxy_location: Region to search from (``X-Proxy-Location`` header).
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A ``FunctionTool`` that queries Serply's ``/v1/news/`` endpoint.
        """
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def serply_news_search(
            query: Annotated[str, Field(description="The news search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Search Google News through Serply."""
            limit = _int(resolve_variable(num, ctx, param_name="num"))
            raw = await _get(
                client_kwargs,
                "/v1/news/",
                {
                    "q": query,
                    "gl": resolve_variable(gl, ctx, param_name="gl"),
                    "hl": resolve_variable(hl, ctx, param_name="hl"),
                },
                {"X-Proxy-Location": resolve_variable(proxy_location, ctx, param_name="proxy_location")},
            )
            return ToolResult(
                SerplyNewsSearchResponse(
                    query=query,
                    results=[
                        SerplyNewsResult(
                            title=_text(item, "title"),
                            link=_text(item, "link"),
                            published=_text(item, "published"),
                            source=_text(item.get("source"), "title"),
                        )
                        for item in _items(raw, "entries")[: limit if limit and limit > 0 else _NEWS_MAX_RESULTS]
                    ],
                )
            )

        return serply_news_search

    def scholar(
        self,
        *,
        num: int | Variable | None = None,
        gl: str | Variable | None = None,
        hl: str | Variable | None = None,
        proxy_location: ProxyLocation | Variable | None = None,
        name: str = "serply_scholar_search",
        description: str = (
            "Search Google Scholar through Serply. Returns academic articles with titles, authors, "
            "citation counts, and URLs."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the Google Scholar search tool.

        Args:
            num: Number of results to request.
            gl: Google country code, forwarded as a query parameter.
            hl: Google interface language, forwarded as a query parameter.
            proxy_location: Region to search from (``X-Proxy-Location`` header).
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A ``FunctionTool`` that queries Serply's ``/v1/scholar/`` endpoint.
        """
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def serply_scholar_search(
            query: Annotated[str, Field(description="The academic search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Search Google Scholar through Serply."""
            raw = await _get(
                client_kwargs,
                "/v1/scholar/",
                {
                    "q": query,
                    "num": resolve_variable(num, ctx, param_name="num"),
                    "gl": resolve_variable(gl, ctx, param_name="gl"),
                    "hl": resolve_variable(hl, ctx, param_name="hl"),
                },
                {"X-Proxy-Location": resolve_variable(proxy_location, ctx, param_name="proxy_location")},
            )
            return ToolResult(
                SerplyScholarSearchResponse(
                    query=query,
                    results=[
                        SerplyScholarResult(
                            title=_text(item, "title"),
                            link=_text(item, "link"),
                            authors=_text(item.get("author"), "names"),
                            pdf_link=_text(item.get("doc"), "link"),
                            citations=_int(_nested(item, "extras", "citations", "count")),
                        )
                        for item in _items(raw, "articles")
                    ],
                )
            )

        return serply_scholar_search

    def _client_kwargs(self) -> dict[str, Any]:
        """Snapshot the connection settings as ``httpx.AsyncClient`` keyword arguments."""
        return {
            "base_url": self._base_url,
            "headers": {"X-Api-Key": self._api_key, "User-Agent": _USER_AGENT},
            "timeout": self._timeout,
            "proxy": self._proxy,
            "verify": self._verify,
        }
