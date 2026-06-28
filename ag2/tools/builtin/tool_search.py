# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import AsyncExitStack, ExitStack
from dataclasses import dataclass, field
from typing import Literal

from ag2.annotations import Context
from ag2.events import BuiltinToolCallEvent, ToolCallEvent
from ag2.middleware import BaseMiddleware
from ag2.tools.final.function_tool import FunctionToolSchema
from ag2.tools.schemas import ToolSchema
from ag2.tools.tool import Tool

TOOL_SEARCH_TOOL_NAME = "tool_search"


@dataclass(slots=True)
class ToolSearchToolSchema(ToolSchema):
    type: str = field(default=TOOL_SEARCH_TOOL_NAME, init=False)
    mode: Literal["regex", "bm25"] = "regex"


class ToolSearchTool(Tool):
    """Server-side tool search.

    Lets the model discover deferred tools on demand instead of receiving
    every tool definition up front. The provider runs the search and expands
    matching tools server-side; ag2 only emits the tool definition.

    `mode` selects Anthropic's variant ("regex" or "bm25"). OpenAI exposes a
    single tool-search tool, so `mode` is ignored on OpenAI.
    """

    __slots__ = ("_mode", "name")

    def __init__(self, *, mode: Literal["regex", "bm25"] = "regex") -> None:
        self._mode: Literal["regex", "bm25"] = mode
        self.name = TOOL_SEARCH_TOOL_NAME

    async def schemas(self, context: "Context") -> list[ToolSearchToolSchema]:
        return [ToolSearchToolSchema(mode=self._mode)]

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


def assert_tool_search_config(schemas: "Iterable[ToolSchema]") -> None:
    """Raise if deferred tools are present without a ToolSearchTool to load them.

    A deferred tool is only sent to the model as a searchable reference; with no
    tool-search tool the model can never discover it, so the configuration is a
    silent no-op. Fail fast with an actionable message instead.
    """
    schemas = list(schemas)
    has_search = any(isinstance(s, ToolSearchToolSchema) for s in schemas)
    if has_search:
        return

    deferred = [s.function.name for s in schemas if isinstance(s, FunctionToolSchema) and s.defer_loading]
    if deferred:
        raise ValueError(
            "Tools marked defer_loading=True require a ToolSearchTool in the agent's "
            f"tools so the model can discover them: {', '.join(deferred)}"
        )
