# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypeAlias

from perplexity import Perplexity
from perplexity.types import APIPublicSearchResult
from pydantic import Field

from autogen.beta.annotations import Context, Variable
from autogen.beta.events import ImageInput, ToolResult
from autogen.beta.middleware import BaseMiddleware, ToolMiddleware
from autogen.beta.tools.builtin._resolve import resolve_variable
from autogen.beta.tools.final.function_tool import FunctionToolSchema, tool
from autogen.beta.tools.tool import Tool

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
class SearchResult:
    title: str
    url: str
    date: str | None = None
    snippet: str | None = None


@dataclass(slots=True)
class ImageMeta:
    image_url: str
    origin_url: str | None = None
    title: str | None = None
    width: int | None = None
    height: int | None = None


@dataclass(slots=True)
class SearchResponse:
    query: str
    content: str | None = None
    search_results: list[SearchResult] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    images: list[ImageMeta] = field(default_factory=list)


class PerplexitySearchTool(Tool):
    __slots__ = ("_tool", "name")

    def __init__(
        self,
        api_key: str | None = None,
        model: SonarModel | Variable | None = None,
        max_tokens: int | Variable | None = None,
        search_domain_filter: Sequence[str] | Variable | None = None,
        search_context_size: SearchContextSize | Variable | None = None,
        search_mode: SearchMode | Variable | None = None,
        search_recency_filter: RecencyFilter | Variable | None = None,
        return_images: bool | Variable | None = None,
        return_related_questions: bool | Variable | None = None,
        client: Perplexity | None = None,
        name: str = "perplexity_search",
        *,
        description: str = (
            "Perplexity AI search tool for web search, news search, and conversational search "
            "for finding answers to everyday questions, conducting in-depth research and analysis."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        _client = client if client is not None else Perplexity(api_key=api_key)

        @tool(
            name=name,
            description=description,
            middleware=middleware,
        )
        def perplexity_search(
            query: Annotated[str, Field(description="The search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Search the web using Perplexity AI and return content with sources."""
            resolved_model = resolve_variable(model, ctx, param_name="model") or "sonar"
            resolved_max_tokens = resolve_variable(max_tokens, ctx, param_name="max_tokens") or 1000
            resolved_context_size = (
                resolve_variable(search_context_size, ctx, param_name="search_context_size") or "high"
            )

            kwargs: dict[str, Any] = {
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
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            raw = _client.chat.completions.create(
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
                SearchResult(
                    title=r.title or "",
                    url=r.url or "",
                    date=r.date,
                    snippet=r.snippet,
                )
                for r in search_results
            ]

            citations = list(getattr(raw, "citations", None) or [])

            raw_images: list[dict[str, Any]] = getattr(raw, "images", None) or []
            images = [
                ImageMeta(
                    image_url=url,
                    origin_url=img.get("origin_url"),
                    title=img.get("title"),
                    width=img.get("width"),
                    height=img.get("height"),
                )
                for img in raw_images
                if (url := img.get("image_url"))
            ]

            response = SearchResponse(
                query=query,
                content=content,
                search_results=results,
                citations=citations,
                images=images,
            )

            image_parts = [ImageInput(url=img.image_url) for img in images]
            return ToolResult(response, *image_parts)

        self._tool = perplexity_search
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
