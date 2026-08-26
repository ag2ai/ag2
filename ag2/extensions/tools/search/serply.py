# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Serply search extension for AG2.

Serply exposes Google web, Google News and Google Scholar results through one
REST API. This extension wraps the three endpoints as agent tools.

Maintainer: googio
Docs: https://docs.ag2.ai/docs/user-guide/extensions/tools/search/serply/
"""

import html
import re
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

# Serply sits behind Cloudflare, which rejects requests without a User-Agent.
_USER_AGENT = "ag2-serply-extension"
_TAG_RE = re.compile(r"<[^>]+>")


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
    summary: str = ""


@dataclass(slots=True)
class SerplyNewsSearchResponse:
    query: str
    results: list[SerplyNewsResult] = field(default_factory=list)


@dataclass(slots=True)
class SerplyScholarResult:
    title: str
    link: str
    authors: str = ""
    description: str = ""
    citations: int | None = None


@dataclass(slots=True)
class SerplyScholarSearchResponse:
    query: str
    results: list[SerplyScholarResult] = field(default_factory=list)


def _text(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    return value if isinstance(value, str) else ""


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _items(raw: dict[str, Any], key: str) -> list[dict[str, Any]]:
    items = raw.get(key)
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _plain_text(fragment: str) -> str:
    """Flatten the HTML fragment Google News uses for entry summaries."""
    return " ".join(_TAG_RE.sub(" ", html.unescape(fragment)).split())


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

    __slots__ = ("_api_key", "_base_url", "_timeout")

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.serply.io",
        timeout: float = 60.0,
        num: int | Variable | None = None,
        gl: str | Variable | None = None,
        hl: str | Variable | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

        super().__init__(
            self.web(num=num, gl=gl, hl=hl),
            self.news(gl=gl, hl=hl),
            self.scholar(num=num, gl=gl, hl=hl),
            name="serply_search_toolkit",
            middleware=middleware,
        )

    def web(
        self,
        *,
        num: int | Variable | None = None,
        gl: str | Variable | None = None,
        hl: str | Variable | None = None,
        name: str = "serply_web_search",
        description: str = (
            "Search the web with Google through Serply. Returns ranked results with titles, snippets, and URLs."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        @tool(name=name, description=description, middleware=middleware)
        async def serply_web_search(
            query: Annotated[str, Field(description="The web search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Search Google web results through Serply."""
            raw = await self._get(
                "/v1/search/",
                {
                    "q": query,
                    "num": resolve_variable(num, ctx, param_name="num"),
                    "gl": resolve_variable(gl, ctx, param_name="gl"),
                    "hl": resolve_variable(hl, ctx, param_name="hl"),
                },
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
        gl: str | Variable | None = None,
        hl: str | Variable | None = None,
        name: str = "serply_news_search",
        description: str = (
            "Search Google News through Serply. Returns recent articles with titles, sources, publish dates, and URLs."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        @tool(name=name, description=description, middleware=middleware)
        async def serply_news_search(
            query: Annotated[str, Field(description="The news search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Search Google News through Serply."""
            raw = await self._get(
                "/v1/news/",
                {
                    "q": query,
                    "gl": resolve_variable(gl, ctx, param_name="gl"),
                    "hl": resolve_variable(hl, ctx, param_name="hl"),
                },
            )
            results = []
            for item in _items(raw, "entries"):
                source = item.get("source")
                results.append(
                    SerplyNewsResult(
                        title=_text(item, "title"),
                        link=_text(item, "link"),
                        published=_text(item, "published"),
                        source=_text(source, "title") if isinstance(source, dict) else "",
                        summary=_plain_text(_text(item, "summary")),
                    )
                )
            return ToolResult(SerplyNewsSearchResponse(query=query, results=results))

        return serply_news_search

    def scholar(
        self,
        *,
        num: int | Variable | None = None,
        gl: str | Variable | None = None,
        hl: str | Variable | None = None,
        name: str = "serply_scholar_search",
        description: str = (
            "Search Google Scholar through Serply. Returns academic articles with titles, authors, "
            "citation counts, and URLs."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        @tool(name=name, description=description, middleware=middleware)
        async def serply_scholar_search(
            query: Annotated[str, Field(description="The academic search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Search Google Scholar through Serply."""
            raw = await self._get(
                "/v1/scholar/",
                {
                    "q": query,
                    "num": resolve_variable(num, ctx, param_name="num"),
                    "gl": resolve_variable(gl, ctx, param_name="gl"),
                    "hl": resolve_variable(hl, ctx, param_name="hl"),
                },
            )
            results = []
            for item in _items(raw, "articles"):
                author = item.get("author")
                extras = item.get("extras")
                citations = extras.get("citations") if isinstance(extras, dict) else None
                results.append(
                    SerplyScholarResult(
                        title=_text(item, "title"),
                        link=_text(item, "link"),
                        authors=_text(author, "names") if isinstance(author, dict) else "",
                        description=_text(item, "description"),
                        citations=_int(citations.get("count")) if isinstance(citations, dict) else None,
                    )
                )
            return ToolResult(SerplyScholarSearchResponse(query=query, results=results))

        return serply_scholar_search

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        request_params = {key: value for key, value in params.items() if value is not None}

        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers={"X-Api-Key": self._api_key, "User-Agent": _USER_AGENT},
            timeout=self._timeout,
        ) as client:
            response = await client.get(path, params=request_params)
            response.raise_for_status()

        raw = response.json()
        return raw if isinstance(raw, dict) else {}
