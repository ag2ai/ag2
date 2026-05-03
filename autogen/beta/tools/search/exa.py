# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypeAlias

from exa_py import AsyncExa
from pydantic import Field

from autogen.beta.annotations import Context, Variable
from autogen.beta.events import ToolResult
from autogen.beta.middleware import ToolMiddleware
from autogen.beta.tools.builtin._resolve import resolve_variable
from autogen.beta.tools.final import Toolkit, tool
from autogen.beta.tools.final.function_tool import FunctionTool

SearchType: TypeAlias = Literal["neural", "keyword", "hybrid", "auto", "fast", "deep"]
Category: TypeAlias = Literal[
    "company",
    "research paper",
    "news",
    "pdf",
    "github",
    "tweet",
    "personal site",
    "linkedin profile",
    "financial report",
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
    autoprompt_string: str | None = None


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
      - ``exa_search``: neural/keyword/hybrid web search with optional text content
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

    __slots__ = ("_api_key", "_client")

    def __init__(
        self,
        api_key: str | None = None,
        *,
        num_results: int | Variable | None = None,
        max_characters: int | Variable | None = None,
        client: AsyncExa | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        self._api_key = api_key
        self._client = client

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
        use_autoprompt: bool | Variable | None = None,
        livecrawl: Livecrawl | Variable | None = None,
        name: str = "exa_search",
        description: str = (
            "Search the web using Exa's neural search engine. "
            "Returns ranked results with titles, URLs, relevance scores, and optional text content."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        api_key = self._api_key
        user_client = self._client

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
                "use_autoprompt": resolve_variable(use_autoprompt, ctx, param_name="use_autoprompt"),
                "livecrawl": resolve_variable(livecrawl, ctx, param_name="livecrawl"),
            }
            kwargs = {k: v for k, v in params.items() if v is not None}
            resolved_max_chars = resolve_variable(max_characters, ctx, param_name="max_characters")
            if resolved_max_chars is not None:
                kwargs["text"] = {"maxCharacters": resolved_max_chars}

            if user_client is not None:
                if resolved_max_chars is not None:
                    raw = await user_client.search_and_contents(query, **kwargs)
                else:
                    raw = await user_client.search(query, **kwargs)
            else:
                c = AsyncExa(api_key=api_key)
                try:
                    if resolved_max_chars is not None:
                        raw = await c.search_and_contents(query, **kwargs)
                    else:
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
                            score=getattr(r, "score", None),
                            published_date=getattr(r, "published_date", None),
                            author=getattr(r, "author", None),
                            text=getattr(r, "text", None),
                        )
                        for r in raw.results
                    ],
                    autoprompt_string=getattr(raw, "autoprompt_string", None) or getattr(raw, "autoprompt", None),
                )
            )

        return exa_search

    def find_similar(
        self,
        *,
        num_results: int | Variable | None = None,
        exclude_source_domain: bool | Variable | None = None,
        name: str = "exa_find_similar",
        description: str = (
            "Find web pages similar to a given URL. Useful for discovering "
            "related content, competitors, or alternative sources."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        api_key = self._api_key
        user_client = self._client

        @tool(name=name, description=description, middleware=middleware)
        async def exa_find_similar(
            url: Annotated[str, Field(description="The URL to find similar pages for.")],
            ctx: Context,
        ) -> ToolResult:
            """Find pages similar to a given URL."""
            params: dict[str, Any] = {
                "num_results": resolve_variable(num_results, ctx, param_name="num_results"),
                "exclude_source_domain": resolve_variable(
                    exclude_source_domain, ctx, param_name="exclude_source_domain"
                ),
            }
            kwargs = {k: v for k, v in params.items() if v is not None}
            if user_client is not None:
                raw = await user_client.find_similar(url, **kwargs)
            else:
                c = AsyncExa(api_key=api_key)
                try:
                    raw = await c.find_similar(url, **kwargs)
                finally:
                    await c.client.aclose()
            results = [
                ExaSearchResult(
                    title=r.title or "",
                    url=r.url,
                    score=getattr(r, "score", None),
                    published_date=getattr(r, "published_date", None),
                    author=getattr(r, "author", None),
                    text=getattr(r, "text", None),
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
        user_client = self._client

        @tool(name=name, description=description, middleware=middleware)
        async def exa_get_contents(
            urls: Annotated[list[str], Field(description="URLs to fetch content for.")],
            ctx: Context,
        ) -> ToolResult:
            """Fetch the full text content of specific URLs."""
            if user_client is not None:
                raw = await user_client.get_contents(urls, text=True)
            else:
                c = AsyncExa(api_key=api_key)
                try:
                    raw = await c.get_contents(urls, text=True)
                finally:
                    await c.client.aclose()
            results = [
                ExaContentResult(
                    url=r.url,
                    title=r.title or "",
                    text=r.text or "",
                    author=getattr(r, "author", None),
                    published_date=getattr(r, "published_date", None),
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
        user_client = self._client

        @tool(name=name, description=description, middleware=middleware)
        async def exa_answer(
            query: Annotated[str, Field(description="The question to answer.")],
            ctx: Context,
        ) -> ToolResult:
            """Generate an AI answer with citations."""
            if user_client is not None:
                raw = await user_client.answer(query, text=True)
            else:
                c = AsyncExa(api_key=api_key)
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
