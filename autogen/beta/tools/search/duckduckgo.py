# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Annotated

from ddgs.ddgs import DDGS
from pydantic import Field

from autogen.beta.annotations import Context, Variable
from autogen.beta.middleware import BaseMiddleware, ToolMiddleware
from autogen.beta.tools.builtin._resolve import resolve_variable
from autogen.beta.tools.final.function_tool import FunctionToolSchema, tool
from autogen.beta.tools.tool import Tool


@dataclass(slots=True)
class SearchResult:
    title: str
    href: str
    body: str


@dataclass(slots=True)
class SearchResponse:
    query: str
    results: list[SearchResult] = field(default_factory=list)


class DuckDuckSearchTool(Tool):
    __slots__ = (
        "_tool",
        "name",
    )

    def __init__(
        self,
        max_results: int | Variable | None = None,
        region: str | Variable | None = None,
        safesearch: str | Variable | None = None,
        client: DDGS | None = None,
        name: str = "duckduckgo_search",
        *,
        description: str = ("Search the web using DuckDuckGo. Returns titles, URLs, and snippets for each result."),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        _client = client if client is not None else DDGS()
        _max_results = 5 if max_results is None else max_results
        _region = "us-en" if region is None else region
        _safesearch = "moderate" if safesearch is None else safesearch

        @tool(
            name=name,
            description=description,
            middleware=middleware,
        )
        def duckduckgo_search(
            query: Annotated[str, Field(description="The search query string.")],
            ctx: Context,
        ) -> SearchResponse:
            """Search the web using DuckDuckGo and return structured results."""
            resolved_max = resolve_variable(_max_results, ctx, param_name="max_results")
            resolved_region = resolve_variable(_region, ctx, param_name="region")
            resolved_safesearch = resolve_variable(_safesearch, ctx, param_name="safesearch")

            raw = _client.text(
                query,
                region=resolved_region,
                safesearch=resolved_safesearch,
                max_results=resolved_max,
            )
            items = [SearchResult(title=r["title"], href=r["href"], body=r["body"]) for r in (raw or [])]
            return SearchResponse(query=query, results=items)

        self._tool = duckduckgo_search
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
