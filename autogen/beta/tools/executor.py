# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from traceback import format_exc

from autogen.beta.events import ToolCall, ToolCalls, ToolError, ToolResult, ToolResults
from autogen.beta.stream import Context

from .tool import Tool


class ToolsExecutor:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self.tools = {t.name: t for t in tools}

    def add_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    async def execute_tools(self, event: ToolCalls, ctx: Context) -> None:
        results = []

        for call in event.calls:
            async with ctx.stream.get((ToolError.id == call.id) | (ToolResult.id == call.id)) as result:
                await ctx.send(call)
                results.append(await result)

        await ctx.send(ToolResults(results=results))

    async def execute_tool(self, event: ToolCall, ctx: Context) -> None:
        result = await self._execute(event, ctx)
        await ctx.send(result)

    async def _execute(self, event: ToolCall, ctx: Context) -> ToolError | ToolResult:
        if tool := self.tools.get(event.name):
            try:
                result = await tool.execute(event.arguments, ctx)

            except Exception:
                return ToolError(id=event.id, name=event.name, content=format_exc(limit=3))

            else:
                return ToolResult(id=event.id, name=event.name, content=result.decode())

        else:
            return ToolError(id=event.id, name=event.name, content="Tool not found")
