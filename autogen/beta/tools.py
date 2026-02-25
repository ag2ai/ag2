from collections.abc import Awaitable, Callable
from functools import wraps

from .events import ToolCall, ToolResult
from .stream import Context, Subscriber


def tool(func: Callable[[str, Context], Awaitable[str]]) -> Subscriber:
    @wraps(func)
    async def execute_tool(event: ToolCall, ctx: Context) -> None:
        result = await func(event.arguments, ctx)
        event = ToolResult(name=event.name, result=result)
        await ctx.stream.send(event)

    return execute_tool
