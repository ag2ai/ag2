# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypeAlias

from perplexity import Perplexity
from perplexity.types import APIPublicSearchResult
from perplexity.types.search_create_response import Result as SearchAPIResult
from pydantic import Field

from autogen.beta.annotations import Context, Variable
from autogen.beta.events import ImageInput, ToolResult
from autogen.beta.middleware import ToolMiddleware
from autogen.beta.tools.builtin._resolve import resolve_variable
from autogen.beta.tools.final import Toolkit, tool
from autogen.beta.tools.final.function_tool import FunctionTool

SonarModel: TypeAlias = Literal[
    "sonar",
    "sonar-pro",
    "sonar-deep-research",
    "sonar-reasoning",
    "sonar-reasoning-pro",
]
SearchMode: TypeAlias = Literal["web", "academic", "sec"]
SearchContextSize: TypeAlias = Literal["low", "medium", "high"]
RecencyFilter: TypeAlias = Literal["hour", "day", "week", "month", "year"]


@dataclass(slots=True)
class PerplexitySearchResult:
    title: str
    url: str
    snippet: str | None = None
    date: str | None = None


@dataclass(slots=True)
class PerplexityImageMeta:
    image_url: str
    origin_url: str | None = None
    title: str | None = None
    width: int | None = None
    height: int | None = None


@dataclass(slots=True)
class PerplexitySearchResponse:
    query: str
    results: list[PerplexitySearchResult] = field(default_factory=list)
    content: str | None = None
    citations: list[str] = field(default_factory=list)
    images: list[PerplexityImageMeta] = field(default_factory=list)


class PerplexitySearchToolkit(Toolkit):
    """Toolkit that exposes the Perplexity search APIs as two related tools
    sharing one HTTP client.

    The two tools mirror Perplexity's primary search endpoints:
      - ``perplexity_search``: raw web search via the Search API
      - ``perplexity_answer``: LLM-generated answer with citations via Sonar Chat Completions

    By default, passing the whole toolkit to an agent registers both tools.
    To use a subset, or to customise per-tool parameters, call the factory
    methods directly and pass the returned tools to the agent::

        toolkit = PerplexitySearchToolkit(api_key=...)

        # both tools
        agent = Agent("a", config=config, tools=[toolkit])

        # only one, with custom parameters
        agent = Agent(
            "a",
            config=config,
            tools=[toolkit.search(max_results=5)],
        )

    The constructor reads ``PERPLEXITY_API_KEY`` from the environment when
    ``api_key`` is omitted (handled by the underlying ``perplexity.Perplexity`` SDK).
    """

    __slots__ = ("_client",)

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: Perplexity | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        self._client = client if client is not None else Perplexity(api_key=api_key)

        super().__init__(
            self.search(),
            self.answer(),
            name="perplexity_toolkit",
            middleware=middleware,
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> "PerplexitySearchToolkit":
        # The Perplexity SDK wraps httpx.Client, whose state holds a _thread.RLock
        # that fails pickle-based deepcopy. Because Agent.add_tool calls deepcopy
        # on each registered tool (function_tool.py:99), passing the toolkit
        # whole (tools=[toolkit]) would otherwise raise TypeError: cannot pickle '_thread.RLock' object.
        # The override copies the _tools dict but shares the SDK client by reference
        # (the client is thread-safe and stateless across requests)
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        new._client = self._client
        new.name = self.name
        new._middleware = self._middleware
        new._tools = dict(self._tools)
        return new

    def search(
        self,
        *,
        max_results: int | Variable | None = None,
        max_tokens_per_page: int | Variable | None = None,
        search_domain_filter: Sequence[str] | Variable | None = None,
        search_recency_filter: RecencyFilter | Variable | None = None,
        search_after_date_filter: str | Variable | None = None,
        search_before_date_filter: str | Variable | None = None,
        name: str = "perplexity_search",
        description: str = (
            "Search the web with the Perplexity Search API. Returns ranked results "
            "with title, URL, snippet, and date. Supports domain allow/deny filters "
            "(prefix a domain with '-' to exclude it) and recency / date-range filters."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        client = self._client

        @tool(name=name, description=description, middleware=middleware)
        def perplexity_search(
            query: Annotated[str, Field(description="The search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Run a Perplexity Search API call and return structured results."""
            params: dict[str, Any] = {
                "max_results": resolve_variable(max_results, ctx, param_name="max_results"),
                "max_tokens_per_page": resolve_variable(max_tokens_per_page, ctx, param_name="max_tokens_per_page"),
                "search_domain_filter": resolve_variable(search_domain_filter, ctx, param_name="search_domain_filter"),
                "search_recency_filter": resolve_variable(
                    search_recency_filter, ctx, param_name="search_recency_filter"
                ),
                "search_after_date_filter": resolve_variable(
                    search_after_date_filter, ctx, param_name="search_after_date_filter"
                ),
                "search_before_date_filter": resolve_variable(
                    search_before_date_filter, ctx, param_name="search_before_date_filter"
                ),
            }
            kwargs = {k: v for k, v in params.items() if v is not None}

            raw = client.search.create(query=query, **kwargs)

            raw_results: list[SearchAPIResult] = getattr(raw, "results", None) or []
            results = [
                PerplexitySearchResult(
                    title=r.title or "",
                    url=r.url or "",
                    snippet=r.snippet,
                    date=r.date,
                )
                for r in raw_results
            ]

            return ToolResult(PerplexitySearchResponse(query=query, results=results))

        return perplexity_search

    def answer(
        self,
        *,
        model: SonarModel | Variable | None = None,
        max_tokens: int | Variable | None = None,
        search_domain_filter: Sequence[str] | Variable | None = None,
        search_context_size: SearchContextSize | Variable | None = None,
        search_mode: SearchMode | Variable | None = None,
        search_recency_filter: RecencyFilter | Variable | None = None,
        return_images: bool | Variable | None = None,
        return_related_questions: bool | Variable | None = None,
        name: str = "perplexity_answer",
        description: str = (
            "Ask Perplexity AI for an LLM-generated answer with web citations. "
            "Useful for conversational questions, in-depth research, and analysis. "
            "Returns content text plus ranked search results and citations."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        client = self._client

        @tool(name=name, description=description, middleware=middleware)
        def perplexity_answer(
            query: Annotated[str, Field(description="The question to answer.")],
            ctx: Context,
        ) -> ToolResult:
            """Generate a Perplexity Sonar answer with citations and search results."""
            resolved_model = resolve_variable(model, ctx, param_name="model") or "sonar"
            resolved_max_tokens = resolve_variable(max_tokens, ctx, param_name="max_tokens") or 1000
            resolved_context_size = (
                resolve_variable(search_context_size, ctx, param_name="search_context_size") or "high"
            )

            params: dict[str, Any] = {
                "search_domain_filter": resolve_variable(search_domain_filter, ctx, param_name="search_domain_filter"),
                "search_mode": resolve_variable(search_mode, ctx, param_name="search_mode"),
                "search_recency_filter": resolve_variable(
                    search_recency_filter, ctx, param_name="search_recency_filter"
                ),
                "return_images": resolve_variable(return_images, ctx, param_name="return_images"),
                "return_related_questions": resolve_variable(
                    return_related_questions, ctx, param_name="return_related_questions"
                ),
            }
            kwargs = {k: v for k, v in params.items() if v is not None}

            raw = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": "Be precise and concise."},
                    {"role": "user", "content": query},
                ],
                max_tokens=resolved_max_tokens,
                web_search_options={"search_context_size": resolved_context_size},
                **kwargs,
            )

            content: str | None = None
            choices = getattr(raw, "choices", None) or []
            if choices:
                message = getattr(choices[0], "message", None)
                content = getattr(message, "content", None) if message is not None else None

            search_results: list[APIPublicSearchResult] = getattr(raw, "search_results", None) or []
            results = [
                PerplexitySearchResult(
                    title=r.title or "",
                    url=r.url or "",
                    snippet=r.snippet,
                    date=r.date,
                )
                for r in search_results
            ]

            citations = list(getattr(raw, "citations", None) or [])

            raw_images: list[dict[str, Any]] = list(getattr(raw, "images", None) or [])
            images = [
                PerplexityImageMeta(
                    image_url=url,
                    origin_url=img.get("origin_url"),
                    title=img.get("title"),
                    width=img.get("width"),
                    height=img.get("height"),
                )
                for img in raw_images
                if (url := img.get("image_url"))
            ]

            response = PerplexitySearchResponse(
                query=query,
                results=results,
                content=content,
                citations=citations,
                images=images,
            )

            image_parts = [ImageInput(url=img.image_url) for img in images]
            return ToolResult(response, *image_parts)

        return perplexity_answer
