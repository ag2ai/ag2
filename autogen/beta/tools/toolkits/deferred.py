# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""DeferredToolkit — expose a large tool catalog through two meta-tools.

When an agent has hundreds of tools, loading every schema upfront wastes
tokens.  ``DeferredToolkit`` hides the catalog behind two meta-tools:

* **search_tools(query)** — keyword-search the catalog and return matching
  tool names with descriptions and parameter schemas.
* **use_tool(name, arguments)** — invoke any catalog entry by name, passing
  arguments as a JSON object string.

The agent spends no tokens on schemas it never needs; it discovers and invokes
tools on demand.

Example::

    from autogen.beta import Agent
    from autogen.beta.tools import DeferredToolkit, FilesystemToolkit, MemoryToolkit
    from autogen.beta.config import OpenAIConfig

    # Wrap a large set of tools in a DeferredToolkit
    deferred = DeferredToolkit(
        *FilesystemToolkit().tools,  # 5 tools
        *MemoryToolkit().tools,  # 4 tools
    )

    # The agent only sees 2 tools: search_tools + use_tool
    agent = Agent("assistant", config=OpenAIConfig("gpt-4o-mini"), tools=[deferred])

    reply = await agent.ask("Find all Python files under /tmp and remember the count.")
    # Agent will: search_tools("filesystem") → find_files schema
    #             use_tool("find_files", '{"pattern": "**/*.py", "path": "/tmp"}')
    #             search_tools("memory") → remember schema
    #             use_tool("remember", '{"content": "42 Python files", "key": "py-count"}')
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Annotated, Any

from pydantic import Field

from autogen.beta.middleware import ToolMiddleware
from autogen.beta.tools.final import Toolkit, tool
from autogen.beta.tools.final.function_tool import FunctionTool

__all__ = ("DeferredToolkit",)


class _CatalogEntry:
    """Metadata + callable for one deferred tool."""

    __slots__ = ("name", "description", "parameters_schema", "_call")

    def __init__(self, ft: FunctionTool) -> None:
        self.name: str = ft.name
        self.description: str = ft.schema.function.description
        self.parameters_schema: dict[str, Any] = ft.schema.function.parameters or {}
        self._call: Callable[..., Any] = ft.model.call

    async def invoke(self, **kwargs: Any) -> str:
        import asyncio

        result = self._call(**kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return str(result)

    def summary(self) -> str:
        """One-line description for search results."""
        return f"[{self.name}] {self.description}"

    def full_description(self) -> str:
        """Name + description + parameter schema (for agent reference)."""
        schema_str = json.dumps(self.parameters_schema, indent=2)
        return f"Tool: {self.name}\nDescription: {self.description}\nParameters:\n{schema_str}"


class DeferredToolkit(Toolkit):
    """Toolkit that exposes a large tool catalog via search + dispatch meta-tools.

    The agent only sees two tools:

    * ``search_tools(query, max_results?)`` — keyword-search tool names and
      descriptions; returns names, descriptions, and parameter schemas.
    * ``use_tool(name, arguments?)`` — invoke a catalog tool by name;
      *arguments* is a JSON object string (``'{}'`` if no params needed).

    This keeps the per-call token cost proportional to what the agent
    actually uses, not the total catalog size.

    Args:
        *tools:
            :class:`~autogen.beta.tools.final.FunctionTool` instances (or
            callables) to add to the hidden catalog.
        middleware:
            Tool-level middleware applied to the two meta-tools.

    Example::

        deferred = DeferredToolkit(
            *FilesystemToolkit().tools,
            *MemoryToolkit().tools,
        )
        agent = Agent("assistant", config=config, tools=[deferred])
    """

    __slots__ = ("_catalog",)

    def __init__(
        self,
        *tools: FunctionTool | Callable[..., Any],
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        self._catalog: dict[str, _CatalogEntry] = {}
        for t in tools:
            ft = t if isinstance(t, FunctionTool) else FunctionTool.ensure_tool(t)
            assert isinstance(ft, FunctionTool)
            self._catalog[ft.name] = _CatalogEntry(ft)

        super().__init__(
            self._search_tools_fn(middleware=middleware),
            self._use_tool_fn(middleware=middleware),
            name="deferred_toolkit",
            middleware=(),
        )

    # ------------------------------------------------------------------
    # Meta-tools
    # ------------------------------------------------------------------

    def _search_tools_fn(self, *, middleware: Iterable[ToolMiddleware] = ()) -> FunctionTool:
        catalog = self._catalog

        @tool(
            name="search_tools",
            description=(
                "Search available tools by keyword. Returns matching tool names, "
                "descriptions, and parameter schemas. Call this before use_tool "
                "if you are not sure which tool to use or what arguments it expects."
            ),
            middleware=middleware,
        )
        def _search_tools(
            query: Annotated[str, Field(description="Search query — matched against tool names and descriptions.")],
            max_results: Annotated[
                int,
                Field(description="Maximum number of results to return.", ge=1, le=50),
            ] = 5,
        ) -> str:
            q = query.lower()
            hits = [entry for entry in catalog.values() if q in entry.name.lower() or q in entry.description.lower()]
            if not hits:
                all_names = ", ".join(sorted(catalog))
                return f"No tools matched '{query}'. Available tools: {all_names}"
            hits = hits[:max_results]
            lines = [f"Found {len(hits)} tool(s) matching '{query}':\n"]
            for entry in hits:
                lines.append(entry.full_description())
                lines.append("")
            return "\n".join(lines)

        return _search_tools

    def _use_tool_fn(self, *, middleware: Iterable[ToolMiddleware] = ()) -> FunctionTool:
        catalog = self._catalog

        @tool(
            name="use_tool",
            description=(
                "Invoke a tool from the catalog by name. "
                "Pass arguments as a JSON object string, e.g. "
                '\'{"path": "src/", "pattern": "**/*.py"}\'. '
                "Use search_tools() first if you are unsure of the tool name or its parameters."
            ),
            middleware=middleware,
        )
        async def _use_tool(
            name: Annotated[str, Field(description="Name of the tool to invoke.")],
            arguments: Annotated[
                str,
                Field(
                    description="JSON object of arguments to pass to the tool. Use '{}' for tools with no required params."
                ),
            ] = "{}",
        ) -> str:
            entry = catalog.get(name)
            if entry is None:
                known = ", ".join(sorted(catalog))
                return f"Tool '{name}' not found in catalog. Available tools: {known}"
            try:
                kwargs = json.loads(arguments)
            except json.JSONDecodeError as exc:
                return f"Invalid JSON in arguments: {exc}"
            if not isinstance(kwargs, dict):
                return "arguments must be a JSON object (dict), not a list or primitive."
            try:
                return await entry.invoke(**kwargs)
            except TypeError as exc:
                return f"Wrong arguments for '{name}': {exc}"
            except Exception as exc:
                return f"Tool '{name}' raised an error: {exc}"

        return _use_tool

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def catalog_names(self) -> tuple[str, ...]:
        """Sorted names of all tools in the catalog."""
        return tuple(sorted(self._catalog))

    def add_to_catalog(self, *tools: FunctionTool | Callable[..., Any]) -> None:
        """Add more tools to the catalog after construction."""
        for t in tools:
            ft = t if isinstance(t, FunctionTool) else FunctionTool.ensure_tool(t)
            assert isinstance(ft, FunctionTool)
            self._catalog[ft.name] = _CatalogEntry(ft)
