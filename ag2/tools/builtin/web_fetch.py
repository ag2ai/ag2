# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import AsyncExitStack, ExitStack
from dataclasses import dataclass, field
from typing import Final, Literal, TypeAlias, get_args

from ag2.annotations import Context, Variable
from ag2.events import BuiltinToolCallEvent, ToolCallEvent
from ag2.middleware import BaseMiddleware
from ag2.tools.schemas import ToolSchema
from ag2.tools.tool import Tool

from ._resolve import resolve_variable

WEB_FETCH_TOOL_NAME: Final = "web_fetch"

WebFetchVersions: TypeAlias = Literal[
    "web_fetch_20250910",
    "web_fetch_20260209",
    "web_fetch_20260309",
    "web_fetch_20260318",
]

#: Every declared version, oldest first, for callers that enumerate them.
WEB_FETCH_VERSIONS: tuple[WebFetchVersions, ...] = get_args(WebFetchVersions)


@dataclass(slots=True)
class OnlyTools:
    """The filter under which only the named tools' results supply fetchable URLs."""

    tools: list[str]


@dataclass(slots=True)
class ExceptTools:
    """The filter under which every result but the named tools' supplies fetchable URLs."""

    tools: list[str]


#: What a tool's results may contribute: everything, nothing, or a named subset.
ToolResultSources: TypeAlias = Literal["all", "none"] | OnlyTools | ExceptTools

#: What a user message may contribute. A named subset is meaningless here — there is no tool to name.
UserInputSources: TypeAlias = Literal["all", "none"]


@dataclass(slots=True)
class UrlSources:
    """Which sources may supply the URLs web fetch is allowed to fetch.

    This is a policy control, not a tuning knob: an agent that fetches URLs out of user text is
    reachable by anyone who can type into the conversation. ``allowed_domains`` cannot stand in
    for it, because that filters by host rather than by who proposed the URL. A field left
    ``None`` is not sent, and the API's own default applies to that source.
    """

    user_input: UserInputSources | None = None
    client_tool_results: ToolResultSources | None = None
    server_tool_results: ToolResultSources | None = None


@dataclass(slots=True)
class WebFetchToolSchema(ToolSchema):
    """What an application asks web fetch to do. Anthropic honours it; Gemini ignores all of it.

    Three fields of the API's tool parameter are deliberately absent, each because it needs a
    concept ag2 does not have. ``defer_loading`` only pays off with tool search, which loads a
    deferred tool via ``tool_reference``; with no tool-search surface, deferring would simply hide
    the tool. ``allowed_callers`` names which caller may invoke the tool, a relationship between
    two tool schemas that ag2 does not model. ``cache_control`` places a cache breakpoint on the tool block, and
    ``AnthropicConfig.prompt_caching`` already decides breakpoints; exposing both would leave two
    mechanisms competing over the same four.
    """

    type: str = field(default=WEB_FETCH_TOOL_NAME, init=False)
    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    citations: bool | None = None
    max_content_tokens: int | None = None
    strict: bool | None = None
    use_cache: bool | None = None
    response_inclusion: Literal["full", "excluded"] | None = None
    url_sources: UrlSources | None = None
    web_fetch_version: WebFetchVersions = "web_fetch_20250910"


class WebFetchTool(Tool):
    __slots__ = (
        "_params",
        "name",
    )

    def __init__(
        self,
        *,
        max_uses: int | Variable | None = None,
        allowed_domains: list[str] | Variable | None = None,
        blocked_domains: list[str] | Variable | None = None,
        citations: bool | Variable | None = None,
        max_content_tokens: int | Variable | None = None,
        strict: bool | Variable | None = None,
        use_cache: bool | Variable | None = None,
        response_inclusion: Literal["full", "excluded"] | Variable | None = None,
        url_sources: UrlSources | Variable | None = None,
        version: WebFetchVersions | Variable | None = None,
    ) -> None:
        self._params: dict[str, object] = {}
        if max_uses is not None:
            self._params["max_uses"] = max_uses
        if allowed_domains is not None:
            self._params["allowed_domains"] = allowed_domains
        if blocked_domains is not None:
            self._params["blocked_domains"] = blocked_domains
        if citations is not None:
            self._params["citations"] = citations
        if max_content_tokens is not None:
            self._params["max_content_tokens"] = max_content_tokens
        if strict is not None:
            self._params["strict"] = strict
        if use_cache is not None:
            self._params["use_cache"] = use_cache
        if response_inclusion is not None:
            self._params["response_inclusion"] = response_inclusion
        if url_sources is not None:
            self._params["url_sources"] = url_sources
        if version is not None:
            self._params["web_fetch_version"] = version

        self.name = WEB_FETCH_TOOL_NAME

    async def schemas(self, context: "Context") -> list[WebFetchToolSchema]:
        resolved = {k: resolve_variable(v, context, param_name=k) for k, v in self._params.items()}
        return [WebFetchToolSchema(**resolved)]

    def register(
        self,
        stack: "ExitStack | AsyncExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        async def execute(event: "ToolCallEvent", context: "Context") -> None:
            pass

        stack.enter_context(
            context.stream.where(BuiltinToolCallEvent.name == WEB_FETCH_TOOL_NAME).sub_scope(execute),
        )
