from collections.abc import Iterable

from autogen.beta.events import ToolCall, ToolResult
from autogen.beta.stream import Context

from .tool import Tool


class ToolsExecutor:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self.tools = {t.name: t for t in tools}

    def add_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    async def execute_tool(self, event: ToolCall, ctx: Context) -> None:
        if tool := self.tools.get(event.name):
            result = await tool.execute(event.arguments, ctx)
            await ctx.send(ToolResult(name=event.name, result=result.decode()))
