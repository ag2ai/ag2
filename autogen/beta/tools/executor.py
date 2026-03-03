# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from contextlib import ExitStack
from typing import Any

from autogen.beta.annotations import Context
from autogen.beta.events import (
    ClientToolCall,
    ModelResponse,
    ToolCall,
    ToolCalls,
    ToolError,
    ToolNotFoundEvent,
    ToolResult,
    ToolResults,
)
from autogen.beta.exceptions import ToolNotFoundError

from .tool import Tool


class ToolExecutor:
    def register(self, stack: "ExitStack", ctx: "Context", *, tools: Iterable[Tool] = ()) -> None:
        stack.enter_context(ctx.stream.where(ToolCalls).sub_scope(self.execute_tools))

        known_tools: set[str] = set()
        for tool in tools:
            known_tools.add(tool.name)
            tool.register(stack, ctx)

        # fallback subscriber to raise NotFound event
        stack.enter_context(
            ctx.stream.where(ToolCall).sub_scope(_tool_not_found(known_tools)),
        )

    async def execute_tools(self, event: ToolCalls, ctx: Context) -> None:
        results = []
        client_calls = []

        for call in event.calls:
            async with ctx.stream.get(
                (ToolError.parent_id == call.id) | (ToolResult.parent_id == call.id) | ClientToolCall
            ) as result:
                await ctx.send(call)

                match await result:
                    case ClientToolCall() as ev:
                        client_calls.append(ev)
                    case ev:
                        results.append(ev)

        if client_calls:
            await ctx.send(
                ModelResponse(
                    tool_calls=ToolCalls(calls=client_calls),
                    response_force=True,
                )
            )

        else:
            await ctx.send(ToolResults(results=results))


def _tool_not_found(known_tools: set[str]) -> Callable[..., Any]:
    async def _tool_not_found(event: "ToolCall", ctx: "Context") -> None:
        if event.name not in known_tools:
            err = ToolNotFoundError(event.name)
            event = ToolNotFoundEvent(
                parent_id=event.id,
                name=event.name,
                content=repr(err),
                error=err,
            )
            await ctx.send(event)

    return _tool_not_found
