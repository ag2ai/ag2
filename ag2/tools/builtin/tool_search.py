# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from contextlib import AsyncExitStack, ExitStack
from dataclasses import dataclass, field
from typing import Any, Literal

from ag2.annotations import Context
from ag2.events import BuiltinToolCallEvent, ToolCallEvent
from ag2.exceptions import ToolConflictError
from ag2.middleware import BaseMiddleware
from ag2.tools.final.function_tool import FunctionTool, FunctionToolSchema
from ag2.tools.schemas import ToolSchema
from ag2.tools.tool import Tool

TOOL_SEARCH_TOOL_NAME = "tool_search"


@dataclass(slots=True)
class ToolSearchToolSchema(ToolSchema):
    type: str = field(default=TOOL_SEARCH_TOOL_NAME, init=False)
    mode: Literal["regex", "bm25"] = "regex"


class ToolSearchTool(Tool):
    """Server-side tool search over a set of deferred tools.

    Wrap the tools you want kept out of the upfront context::

        ToolSearchTool(get_weather, get_stock_price)

    The wrapped tools are sent to the provider as searchable references
    (names/descriptions only) instead of full definitions. The model discovers
    the few it needs on demand via this tool, and the provider expands matches
    server-side. The fixed context stays small and the prompt cache is
    preserved. Unwrapped tools on the agent are loaded eagerly as usual.

    ``mode`` selects Anthropic's variant ("regex" or "bm25"). OpenAI exposes a
    single tool-search tool, so ``mode`` is ignored on OpenAI.
    """

    __slots__ = ("_mode", "_tools", "name")

    def __init__(
        self,
        *tools: "Tool | Callable[..., Any]",
        mode: Literal["regex", "bm25"] = "regex",
    ) -> None:
        if not tools:
            raise ValueError(
                "ToolSearchTool requires at least one tool to defer; pass the tools to "
                "search over, e.g. ToolSearchTool(tool_a, tool_b)."
            )
        self._mode: Literal["regex", "bm25"] = mode
        self.name = TOOL_SEARCH_TOOL_NAME

        self._tools: dict[str, Tool] = {}
        for t in tools:
            ft = FunctionTool.ensure_tool(t)
            if ft.name in self._tools:
                raise ToolConflictError(ft.name)
            self._tools[ft.name] = ft

    @property
    def tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools.values())

    async def schemas(self, context: "Context") -> list[ToolSchema]:
        schemas: list[ToolSchema] = [ToolSearchToolSchema(mode=self._mode)]
        for t in self._tools.values():
            for s in await t.schemas(context):
                schemas.append(
                    FunctionToolSchema(function=s.function, defer_loading=True)
                    if isinstance(s, FunctionToolSchema)
                    else s
                )
        return schemas

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
            context.stream.where(BuiltinToolCallEvent.name == TOOL_SEARCH_TOOL_NAME).sub_scope(execute),
        )

        for t in self._tools.values():
            t.register(stack, context, middleware=middleware)
