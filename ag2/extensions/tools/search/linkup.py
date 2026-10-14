# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Linkup search extension for AG2.

Linkup is a web search API for AI applications. This extension wraps its
``/search`` and ``/fetch`` endpoints as agent tools.

Maintainer: shauryajain21
Docs: https://docs.ag2.ai/docs/user-guide/extensions/tools/search/linkup/
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypeAlias

from linkup import LinkupClient, LinkupSearchResults, LinkupSearchTextResult, LinkupSourcedAnswer
from pydantic import Field

from ag2.annotations import Context, Variable
from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool

Depth: TypeAlias = Literal["flash", "fast", "standard", "deep"]
OutputType: TypeAlias = Literal["searchResults", "sourcedAnswer"]


@dataclass(slots=True)
class LinkupSearchResult:
    name: str
    url: str
    content: str


@dataclass(slots=True)
class LinkupSearchResponse:
    query: str
    results: list[LinkupSearchResult] = field(default_factory=list)


@dataclass(slots=True)
class LinkupAnswerSource:
    name: str
    url: str
    snippet: str


@dataclass(slots=True)
class LinkupAnswerResult:
    answer: str
    sources: list[LinkupAnswerSource] = field(default_factory=list)


@dataclass(slots=True)
class LinkupFetchResult:
    url: str
    markdown: str


class LinkupToolkit(Toolkit):
    """Toolkit that exposes the Linkup web search API as two related tools.

    The two tools mirror Linkup's primary endpoints:
      - ``linkup_search``: real-time web search returning citable sources,
        or a sourced answer when ``output_type="sourcedAnswer"``
      - ``linkup_fetch``: fetch a single web page as clean markdown

    By default, passing the whole toolkit to an agent registers both tools.
    To use a subset, or to customise per-tool parameters, call the factory
    methods directly and pass the returned tools to the agent::

        toolkit = LinkupToolkit(api_key=...)

        # both tools
        agent = Agent("a", config=config, tools=[toolkit])

        # only search, with custom parameters
        agent = Agent(
            "a",
            config=config,
            tools=[toolkit.search(depth="deep", output_type="sourcedAnswer")],
        )

    The tools read ``LINKUP_API_KEY`` from the environment when ``api_key``
    is omitted (handled by the underlying ``linkup.LinkupClient`` SDK).
    """

    __slots__ = ("_api_key",)

    def __init__(
        self,
        api_key: str | None = None,
        *,
        depth: Depth | Variable = "standard",
        max_results: int | Variable | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        self._api_key = api_key

        super().__init__(
            self.search(depth=depth, max_results=max_results),
            self.fetch(),
            name="linkup_toolkit",
            middleware=middleware,
        )

    def search(
        self,
        *,
        depth: Depth | Variable = "standard",
        output_type: OutputType | Variable = "searchResults",
        max_results: int | Variable | None = None,
        include_domains: Sequence[str] | Variable | None = None,
        exclude_domains: Sequence[str] | Variable | None = None,
        from_date: str | Variable | None = None,
        to_date: str | Variable | None = None,
        name: str = "linkup_search",
        description: str = (
            "Search the web in real time with Linkup and return relevant, citable sources "
            "(title, URL, content). Use for current events, facts that may have changed, "
            "and anything that needs a verifiable source."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        api_key = self._api_key

        @tool(name=name, description=description, middleware=middleware)
        async def linkup_search(
            query: Annotated[str, Field(description="The search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Search the web using Linkup and return sources or a sourced answer."""
            resolved_output_type = resolve_variable(output_type, ctx, param_name="output_type")
            resolved_include_domains = resolve_variable(include_domains, ctx, param_name="include_domains")
            resolved_exclude_domains = resolve_variable(exclude_domains, ctx, param_name="exclude_domains")
            params: dict[str, Any] = {
                "max_results": resolve_variable(max_results, ctx, param_name="max_results"),
                "include_domains": list(resolved_include_domains) if resolved_include_domains is not None else None,
                "exclude_domains": list(resolved_exclude_domains) if resolved_exclude_domains is not None else None,
                "from_date": resolve_variable(from_date, ctx, param_name="from_date"),
                "to_date": resolve_variable(to_date, ctx, param_name="to_date"),
            }
            kwargs = {k: v for k, v in params.items() if v is not None}

            c = LinkupClient(api_key=api_key)
            raw = await c.async_search(
                query,
                depth=resolve_variable(depth, ctx, param_name="depth"),
                output_type=resolved_output_type,
                **kwargs,
            )

            if isinstance(raw, LinkupSourcedAnswer):
                return ToolResult(
                    LinkupAnswerResult(
                        answer=raw.answer,
                        sources=[LinkupAnswerSource(name=s.name, url=s.url, snippet=s.snippet) for s in raw.sources],
                    )
                )

            results = raw.results if isinstance(raw, LinkupSearchResults) else []
            return ToolResult(
                LinkupSearchResponse(
                    query=query,
                    results=[
                        LinkupSearchResult(name=r.name, url=r.url, content=r.content)
                        for r in results
                        if isinstance(r, LinkupSearchTextResult)
                    ],
                )
            )

        return linkup_search

    def fetch(
        self,
        *,
        render_js: bool | Variable | None = None,
        name: str = "linkup_fetch",
        description: str = (
            "Fetch a single web page with Linkup and return its content as clean markdown. "
            "Useful when you already know which page you need and want to read it."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        api_key = self._api_key

        @tool(name=name, description=description, middleware=middleware)
        async def linkup_fetch(
            url: Annotated[str, Field(description="The URL of the web page to fetch.")],
            ctx: Context,
        ) -> ToolResult:
            """Fetch a web page and return its markdown content."""
            c = LinkupClient(api_key=api_key)
            raw = await c.async_fetch(url, render_js=resolve_variable(render_js, ctx, param_name="render_js"))
            return ToolResult(LinkupFetchResult(url=url, markdown=raw.markdown))

        return linkup_fetch
