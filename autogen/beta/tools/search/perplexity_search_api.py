# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypeAlias

from perplexity import Perplexity
from pydantic import Field

from autogen.beta.annotations import Context, Variable
from autogen.beta.events import ToolResult
from autogen.beta.middleware import BaseMiddleware, ToolMiddleware
from autogen.beta.tools.builtin._resolve import resolve_variable
from autogen.beta.tools.final.function_tool import FunctionToolSchema, tool
from autogen.beta.tools.tool import Tool

RecencyFilter: TypeAlias = Literal["hour", "day", "week", "month", "year"]


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str | None = None
    date: str | None = None


@dataclass(slots=True)
class SearchResponse:
    query: str
    results: list[SearchResult] = field(default_factory=list)


class PerplexitySearchAPITool(Tool):
    """Search the web using the Perplexity Search API.

    This tool calls Perplexity's dedicated Search endpoint (`client.search.create`),
    which returns a ranked list of web results with title, URL, snippet, and date.
    Unlike chat-completion-based search, this does not generate an LLM answer — it
    returns raw search results, which keeps token usage minimal and lets the
    calling agent decide how to consume them.

    See https://docs.perplexity.ai/docs/search/quickstart for endpoint docs.
    """

    __slots__ = ("_tool", "name")

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int | Variable | None = None,
        max_tokens_per_page: int | Variable | None = None,
        search_domain_filter: Sequence[str] | Variable | None = None,
        search_recency_filter: RecencyFilter | Variable | None = None,
        search_after_date_filter: str | Variable | None = None,
        search_before_date_filter: str | Variable | None = None,
        client: Perplexity | None = None,
        name: str = "perplexity_search_api",
        *,
        description: str = (
            "Search the web with the Perplexity Search API. Returns ranked results "
            "with title, URL, snippet, and date. Supports domain allow/deny filters "
            "(prefix a domain with '-' to exclude it) and recency / date-range filters."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        _client = client if client is not None else Perplexity(api_key=api_key)

        @tool(
            name=name,
            description=description,
            middleware=middleware,
        )
        def perplexity_search_api(
            query: Annotated[str, Field(description="The search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Run a Perplexity Search API call and return structured results."""
            params: dict[str, Any] = {
                "max_results": resolve_variable(max_results, ctx, param_name="max_results"),
                "max_tokens_per_page": resolve_variable(
                    max_tokens_per_page, ctx, param_name="max_tokens_per_page"
                ),
                "search_domain_filter": resolve_variable(
                    search_domain_filter, ctx, param_name="search_domain_filter"
                ),
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

            raw = _client.search.create(query=query, **kwargs)

            raw_results: list[Any] = list(getattr(raw, "results", None) or [])
            results = [
                SearchResult(
                    title=getattr(r, "title", "") or "",
                    url=getattr(r, "url", "") or "",
                    snippet=getattr(r, "snippet", None),
                    date=getattr(r, "date", None),
                )
                for r in raw_results
            ]

            return ToolResult(SearchResponse(query=query, results=results))

        self._tool = perplexity_search_api
        self.name = name

    async def schemas(self, context: Context) -> list[FunctionToolSchema]:
        return await self._tool.schemas(context)

    def register(
        self,
        stack: ExitStack,
        context: Context,
        *,
        middleware: Iterable[BaseMiddleware] = (),
    ) -> None:
        self._tool.register(stack, context, middleware=middleware)
