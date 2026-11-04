# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Membase long-term memory extension for AG2.

Membase keeps a person's long-term memory in named Memories and serves it to any
AI they use. This extension gives an agent three tools over one person's Membase
account: search it, save a note to it when the user asks, and read the user's
profile. Every search passage names the Memory it came from, so the agent can
cite it. There is deliberately no delete tool.

Maintainer: unibaseio
Docs: https://docs.ag2.ai/docs/user-guide/extensions/membase/
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any

from membase import Membase
from pydantic import Field

from ag2.annotations import Context, Variable
from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool


@dataclass(slots=True)
class MembasePassage:
    """One search hit: the text and the Memory it came from."""

    memory_name: str
    content: str
    id: str | None = None
    score: float | None = None


@dataclass(slots=True)
class MembaseSearchResponse:
    """Passages for ``query``, most relevant first.

    ``unavailable`` names the Memories that could not answer this search, so a
    missing passage from them is not evidence that nothing is there.
    """

    query: str
    passages: list[MembasePassage] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MembaseSaveResult:
    """Where a saved note landed. ``status`` is ``succeeded`` once the Memory has learned it."""

    id: str | None
    status: str
    memory_names: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MembaseProfile:
    """The user's standing statements (name, role, lasting preferences), newest first."""

    facts: list[str] = field(default_factory=list)


def _client(api_key: str | None, base_url: str | None) -> Membase:
    """A Membase client. Without ``api_key`` the SDK reads ``MEMBASE_API_KEY``."""
    return Membase(api_key=api_key, base_url=base_url)


def _search_response(query: str, raw: Mapping[str, Any]) -> MembaseSearchResponse:
    passages = [
        MembasePassage(
            memory_name=str(hit.get("memory_name") or "Membase"),
            content=str(hit.get("content") or "").strip(),
            id=hit.get("id"),
            score=hit.get("score"),
        )
        for hit in raw.get("items") or []
        if str(hit.get("content") or "").strip()
    ]
    unavailable = [
        str(p.get("memory_name") or "a Memory") for p in raw.get("partial") or [] if p.get("reason") == "error"
    ]
    return MembaseSearchResponse(query=query, passages=passages, unavailable=unavailable)


def _save_result(raw: Mapping[str, Any] | None) -> MembaseSaveResult:
    raw = raw or {}
    names = sorted({str(x["memory_name"]) for x in raw.get("learning") or [] if x.get("memory_name")})
    return MembaseSaveResult(id=raw.get("id"), status=str(raw.get("status") or "pending"), memory_names=names)


def _profile(raw: Mapping[str, Any]) -> MembaseProfile:
    facts = [str(f.get("content") or "").strip() for f in raw.get("items") or []]
    return MembaseProfile(facts=[f for f in facts if f])


class MembaseToolkit(Toolkit):
    """Toolkit that exposes one person's Membase memory as three agent tools.

    The tools mirror Membase's memory API:
      - ``membase_search``: search every Memory the key reaches; each passage names its Memory
      - ``membase_save``: save one note in the user's own words, only when the user asks
      - ``membase_profile``: read the user's profile facts

    By default, passing the whole toolkit to an agent registers all three tools.
    To use a subset (for example, no saving with a Read key), call the factory
    methods and pass the returned tools to the agent::

        toolkit = MembaseToolkit()

        # all three tools
        agent = Agent("a", config=config, tools=[toolkit])

        # read-only
        agent = Agent("a", config=config, tools=[toolkit.search(limit=5), toolkit.profile()])

    A Membase key belongs to one person's account. When ``api_key`` is omitted,
    the SDK reads ``MEMBASE_API_KEY`` at call time. In a multi-user app, pass
    ``api_key=Variable("membase_api_key")`` and put each user's own key in the
    conversation's variables.
    """

    __slots__ = ("_api_key", "_base_url")

    def __init__(
        self,
        api_key: str | Variable | None = None,
        *,
        base_url: str | None = None,
        limit: int | Variable | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url

        super().__init__(
            self.search(limit=limit),
            self.save(),
            self.profile(),
            name="membase_toolkit",
            middleware=middleware,
        )

    def search(
        self,
        *,
        limit: int | Variable | None = None,
        name: str = "membase_search",
        description: str = (
            "Search the user's long-term memory in Membase for what they told their assistants before and what "
            "their Memories have read. Each passage names the Memory it came from; cite it. Use it when the answer "
            "may depend on the user's past, preferences or documents. The first search after a quiet spell can "
            "take up to a minute."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        api_key = self._api_key
        base_url = self._base_url

        @tool(name=name, description=description, middleware=middleware)
        def membase_search(
            query: Annotated[str, Field(description="What to look for, as a natural-language question or topic.")],
            ctx: Context,
        ) -> ToolResult:
            """Search the user's Membase memory."""
            key = resolve_variable(api_key, ctx, param_name="api_key")
            with _client(key, base_url) as client:
                raw = client.search(query, limit=resolve_variable(limit, ctx, param_name="limit"))
            return ToolResult(_search_response(query, raw))

        return membase_search

    def save(
        self,
        *,
        name: str = "membase_save",
        description: str = (
            "Save one fact or note to the user's long-term memory in Membase. Call it only when the user asked you "
            "to remember something or plainly meant it to be kept, and save their own words, not your summary of "
            "the conversation. Never save secrets, credentials or what other people said."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        api_key = self._api_key
        base_url = self._base_url

        @tool(name=name, description=description, middleware=middleware)
        def membase_save(
            content: Annotated[str, Field(description="What to save, in the user's own words. One fact per call.")],
            ctx: Context,
        ) -> ToolResult:
            """Save a note to the user's Membase memory."""
            key = resolve_variable(api_key, ctx, param_name="api_key")
            with _client(key, base_url) as client:
                raw = client.memories.add(content)
            return ToolResult(_save_result(raw))

        return membase_save

    def profile(
        self,
        *,
        name: str = "membase_profile",
        description: str = (
            "Read who the user is from their Membase profile: standing statements (name, role, lasting "
            "preferences), newest first. Call it once near the start of a conversation."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        api_key = self._api_key
        base_url = self._base_url

        @tool(name=name, description=description, middleware=middleware)
        def membase_profile(ctx: Context) -> ToolResult:
            """Read the user's Membase profile."""
            key = resolve_variable(api_key, ctx, param_name="api_key")
            with _client(key, base_url) as client:
                raw = client.profile_facts.list()
            return ToolResult(_profile(raw))

        return membase_profile
