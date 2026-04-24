# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field
from tavily import TavilyClient

from autogen.beta.annotations import Context, Variable
from autogen.beta.middleware import BaseMiddleware, ToolMiddleware
from autogen.beta.tools.builtin._resolve import resolve_variable
from autogen.beta.tools.final.function_tool import FunctionToolSchema, tool
from autogen.beta.tools.tool import Tool

SearchDepth: TypeAlias = Literal["basic", "advanced", "fast", "ultra-fast"]
Topic: TypeAlias = Literal["general", "news", "finance"]
TimeRange: TypeAlias = Literal["day", "week", "month", "year"]
IncludeAnswer: TypeAlias = bool | Literal["basic", "advanced"]
IncludeRawContent: TypeAlias = bool | Literal["markdown", "text"]


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    content: str
    score: float | None = None


@dataclass(slots=True)
class SearchResponse:
    query: str
    results: list[SearchResult] = field(default_factory=list)
    answer: str | None = None
    images: list[str] = field(default_factory=list)


class TavilySearchTool(Tool):
    __slots__ = ("_tool", "name")

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int | Variable | None = None,
        search_depth: SearchDepth | Variable | None = None,
        topic: Topic | Variable | None = None,
        include_answer: IncludeAnswer | Variable | None = None,
        include_raw_content: IncludeRawContent | Variable | None = None,
        include_images: bool | Variable | None = None,
        time_range: TimeRange | Variable | None = None,
        start_date: str | Variable | None = None,
        end_date: str | Variable | None = None,
        days: int | Variable | None = None,
        include_domains: Sequence[str] | Variable | None = None,
        exclude_domains: Sequence[str] | Variable | None = None,
        country: str | Variable | None = None,
        auto_parameters: bool | Variable | None = None,
        include_favicon: bool | Variable | None = None,
        client: TavilyClient | None = None,
        name: str = "tavily_search",
        *,
        description: str = (
            "Search the web using Tavily. Returns titles, URLs, snippets, relevance scores, "
            "and optional LLM-generated answer, raw content, and images."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        _client = client if client is not None else TavilyClient(api_key=api_key)

        @tool(
            name=name,
            description=description,
            middleware=middleware,
        )
        def tavily_search(
            query: Annotated[str, Field(description="The search query string.")],
            ctx: Context,
        ) -> SearchResponse:
            """Search the web using Tavily and return structured results."""
            params: dict[str, Any] = {
                "max_results": resolve_variable(max_results, ctx, param_name="max_results"),
                "search_depth": resolve_variable(search_depth, ctx, param_name="search_depth"),
                "topic": resolve_variable(topic, ctx, param_name="topic"),
                "include_answer": resolve_variable(include_answer, ctx, param_name="include_answer"),
                "include_raw_content": resolve_variable(include_raw_content, ctx, param_name="include_raw_content"),
                "include_images": resolve_variable(include_images, ctx, param_name="include_images"),
                "time_range": resolve_variable(time_range, ctx, param_name="time_range"),
                "start_date": resolve_variable(start_date, ctx, param_name="start_date"),
                "end_date": resolve_variable(end_date, ctx, param_name="end_date"),
                "days": resolve_variable(days, ctx, param_name="days"),
                "include_domains": resolve_variable(include_domains, ctx, param_name="include_domains"),
                "exclude_domains": resolve_variable(exclude_domains, ctx, param_name="exclude_domains"),
                "country": resolve_variable(country, ctx, param_name="country"),
                "auto_parameters": resolve_variable(auto_parameters, ctx, param_name="auto_parameters"),
                "include_favicon": resolve_variable(include_favicon, ctx, param_name="include_favicon"),
            }
            kwargs = {k: v for k, v in params.items() if v is not None}

            raw = _client.search(query, **kwargs)
            results = [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                    score=r.get("score"),
                )
                for r in (raw.get("results") or [])
            ]
            return SearchResponse(
                query=raw.get("query", query),
                results=results,
                answer=raw.get("answer"),
                images=list(raw.get("images") or []),
            )

        self._tool = tavily_search
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
