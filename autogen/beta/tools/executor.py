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
from autogen.beta.middleware import BaseMiddleware

from .tool import Tool


class ToolExecutor:
    def register(
        self,
        stack: "ExitStack",
        context: "Context",
        *,
        tools: Iterable["Tool"] = (),
        known_tools: Iterable[str] = (),
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        stack.enter_context(context.stream.where(ToolCalls).sub_scope(self.execute_tools))

        for tool in tools:
            tool.register(stack, context, middleware=middleware)

        # fallback subscriber to raise NotFound event
        stack.enter_context(
            context.stream.where(ToolCall).sub_scope(_tool_not_found(known_tools)),
        )

    async def execute_tools(self, event: ToolCalls, context: Context) -> None:
        results = []
        client_calls = []

        for call in event.calls:
            async with context.stream.get(
                (ToolError.parent_id == call.id) | (ToolResult.parent_id == call.id) | ClientToolCall
            ) as result:
                await context.send(call)

                match await result:
                    case ClientToolCall() as ev:
                        client_calls.append(ev)
                    case ev:
                        results.append(ev)

        if client_calls:
            await context.send(
                ModelResponse(
                    tool_calls=ToolCalls(calls=client_calls),
                    response_force=True,
                )
            )

        else:
            await context.send(ToolResults(results=results))


def _tool_not_found(known_tools: Iterable[str]) -> Callable[..., Any]:
    async def _tool_not_found(event: "ToolCall", context: "Context") -> None:
        if event.name not in known_tools:
            err = ToolNotFoundError(event.name)
            event = ToolNotFoundEvent(
                parent_id=event.id,
                name=event.name,
                content=repr(err),
                error=err,
            )
            await context.send(event)

    return _tool_not_found
