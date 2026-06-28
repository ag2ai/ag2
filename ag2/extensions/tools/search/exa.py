# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypeAlias

from exa_py import AsyncExa
from pydantic import Field

from ag2.annotations import Context, Variable
from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool

SearchType: TypeAlias = Literal["auto", "neural", "fast", "deep-lite", "deep", "deep-reasoning", "instant"]
Category: TypeAlias = Literal[
    "company",
    "research paper",
    "news",
    "pdf",
    "personal site",
    "financial report",
    "people",
]
Livecrawl: TypeAlias = Literal["never", "fallback", "always", "preferred"]


@dataclass(slots=True)
class ExaSearchResult:
    title: str
    url: str
    score: float | None = None
    published_date: str | None = None
    author: str | None = None
    text: str | None = None


@dataclass(slots=True)
class ExaSearchResponse:
    query: str
    results: list[ExaSearchResult] = field(default_factory=list)


@dataclass(slots=True)
class ExaContentResult:
    url: str
    title: str
    text: str
    author: str | None = None
    published_date: str | None = None


@dataclass(slots=True)
class ExaAnswerCitation:
    url: str
    title: str
    text: str


@dataclass(slots=True)
class ExaAnswerResult:
    answer: str
    citations: list[ExaAnswerCitation] = field(default_factory=list)


class ExaToolkit(Toolkit):
    """Toolkit that exposes the Exa neural search engine as four related tools
    sharing one HTTP client.

    The four tools mirror Exa's primary endpoints:
      - ``exa_search``: web search with optional text content
      - ``exa_find_similar``: find pages similar to a given URL
      - ``exa_get_contents``: fetch full text for specific URLs
      - ``exa_answer``: get an AI-generated answer with citations

    By default, passing the whole toolkit to an agent registers all four tools.
    To use a subset, or to customise per-tool parameters, call the factory
    methods directly and pass the returned tools to the agent::

        toolkit = ExaToolkit(api_key=...)

        # all four tools
        agent = Agent("a", config=config, tools=[toolkit])

        # only two, with custom parameters
        agent = Agent(
            "a",
            config=config,
            tools=[
                toolkit.search(num_results=5, search_type="neural"),
                toolkit.answer(),
            ],
        )

    The constructor reads ``EXA_API_KEY`` from the environment when ``api_key``
    is omitted (handled by the underlying ``exa_py.AsyncExa`` SDK).
    """

    __slots__ = ("_api_key",)

    def __init__(
        self,
        api_key: str | None = None,
        *,
        num_results: int | Variable | None = None,
        max_characters: int | Variable | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        self._api_key = api_key

        super().__init__(
            self.search(num_results=num_results, max_characters=max_characters),
            self.find_similar(num_results=num_results),
            self.get_contents(),
            self.answer(),
            name="exa_toolkit",
            middleware=middleware,
        )

    def search(
        self,
        *,
        num_results: int | Variable | None = None,
        max_characters: int | Variable | None = None,
        search_type: SearchType | Variable | None = None,
        category: Category | Variable | None = None,
        include_domains: Sequence[str] | Variable | None = None,
        exclude_domains: Sequence[str] | Variable | None = None,
        start_published_date: str | Variable | None = None,
        end_published_date: str | Variable | None = None,
        start_crawl_date: str | Variable | None = None,
        end_crawl_date: str | Variable | None = None,
        livecrawl: Livecrawl | Variable | None = None,
        user_location: str | Variable | None = None,
        moderation: bool | Variable | None = None,
        name: str = "exa_search",
        description: str = (
            "Search the web using Exa's neural search engine. "
            "Returns ranked results with titles, URLs, relevance scores, and optional text content."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        api_key = self._api_key

        @tool(name=name, description=description, middleware=middleware)
        async def exa_search(
            query: Annotated[str, Field(description="The search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Search the web using Exa and return ranked results."""
            params: dict[str, Any] = {
                "num_results": resolve_variable(num_results, ctx, param_name="num_results"),
                "type": resolve_variable(search_type, ctx, param_name="search_type"),
                "category": resolve_variable(category, ctx, param_name="category"),
                "include_domains": resolve_variable(include_domains, ctx, param_name="include_domains"),
                "exclude_domains": resolve_variable(exclude_domains, ctx, param_name="exclude_domains"),
                "start_published_date": resolve_variable(start_published_date, ctx, param_name="start_published_date"),
                "end_published_date": resolve_variable(end_published_date, ctx, param_name="end_published_date"),
                "start_crawl_date": resolve_variable(start_crawl_date, ctx, param_name="start_crawl_date"),
                "end_crawl_date": resolve_variable(end_crawl_date, ctx, param_name="end_crawl_date"),
            }
            kwargs = {k: v for k, v in params.items() if v is not None}

            contents: dict[str, Any] = {}
            resolved_max_chars = resolve_variable(max_characters, ctx, param_name="max_characters")
            if resolved_max_chars is not None:
                contents["text"] = {"maxCharacters": resolved_max_chars}
            resolved_livecrawl = resolve_variable(livecrawl, ctx, param_name="livecrawl")
            if resolved_livecrawl is not None:
                contents["livecrawl"] = resolved_livecrawl
            if contents:
                kwargs["contents"] = contents

            c = AsyncExa(api_key=api_key)
            c.headers["x-exa-integration"] = "ag2"
            try:
                raw = await c.search(query, **kwargs)
            finally:
                await c.client.aclose()

            return ToolResult(
                ExaSearchResponse(
                    query=query,
                    results=[
                        ExaSearchResult(
                            title=r.title or "",
                            url=r.url,
                            score=r.score,
                            published_date=r.published_date,
                            author=r.author,
                            text=r.text,
                        )
                        for r in raw.results
                    ],
                )
            )

        return exa_search

    def find_similar(
        self,
        *,
        num_results: int | Variable | None = None,
        include_domains: Sequence[str] | Variable | None = None,
        exclude_domains: Sequence[str] | Variable | None = None,
        exclude_source_domain: bool | Variable | None = None,
        category: Category | Variable | None = None,
        name: str = "exa_find_similar",
        description: str = (
            "Find web pages similar to a given URL. Useful for discovering "
            "related content, competitors, or alternative sources."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        api_key = self._api_key

        @tool(name=name, description=description, middleware=middleware)
        async def exa_find_similar(
            url: Annotated[str, Field(description="The URL to find similar pages for.")],
            ctx: Context,
        ) -> ToolResult:
            """Find pages similar to a given URL."""
            params: dict[str, Any] = {
                "num_results": resolve_variable(num_results, ctx, param_name="num_results"),
                "include_domains": resolve_variable(include_domains, ctx, param_name="include_domains"),
                "exclude_domains": resolve_variable(exclude_domains, ctx, param_name="exclude_domains"),
                "exclude_source_domain": resolve_variable(
                    exclude_source_domain, ctx, param_name="exclude_source_domain"
                ),
            }
            kwargs = {k: v for k, v in params.items() if v is not None}
            c = AsyncExa(api_key=api_key)
            c.headers["x-exa-integration"] = "ag2"
            try:
                raw = await c.find_similar(url, **kwargs)
            finally:
                await c.client.aclose()
            results = [
                ExaSearchResult(
                    title=r.title or "",
                    url=r.url,
                    score=r.score,
                    published_date=r.published_date,
                    author=r.author,
                    text=r.text,
                )
                for r in raw.results
            ]
            return ToolResult(results)

        return exa_find_similar

    def get_contents(
        self,
        *,
        name: str = "exa_get_contents",
        description: str = (
            "Get the full text content of specific URLs. Useful when you already "
            "know which pages you need and want to read them."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        api_key = self._api_key

        @tool(name=name, description=description, middleware=middleware)
        async def exa_get_contents(
            urls: Annotated[list[str], Field(description="URLs to fetch content for.")],
            ctx: Context,
        ) -> ToolResult:
            """Fetch the full text content of specific URLs."""
            c = AsyncExa(api_key=api_key)
            c.headers["x-exa-integration"] = "ag2"
            try:
                raw = await c.get_contents(urls, text=True)
            finally:
                await c.client.aclose()
            results = [
                ExaContentResult(
                    url=r.url,
                    title=r.title or "",
                    text=r.text or "",
                    author=r.author,
                    published_date=r.published_date,
                )
                for r in raw.results
            ]
            return ToolResult(results)

        return exa_get_contents

    def answer(
        self,
        *,
        name: str = "exa_answer",
        description: str = ("Generate an AI-powered answer to a question with citations from web sources."),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        api_key = self._api_key

        @tool(name=name, description=description, middleware=middleware)
        async def exa_answer(
            query: Annotated[str, Field(description="The question to answer.")],
            ctx: Context,
        ) -> ToolResult:
            """Generate an AI answer with citations."""
            c = AsyncExa(api_key=api_key)
            c.headers["x-exa-integration"] = "ag2"
            try:
                raw = await c.answer(query, text=True)
            finally:
                await c.client.aclose()
            return ToolResult(
                ExaAnswerResult(
                    answer=raw.answer,
                    citations=[
                        ExaAnswerCitation(url=c.url, title=c.title or "", text=c.text or "")
                        for c in (raw.citations or [])
                    ],
                )
            )

        return exa_answer
