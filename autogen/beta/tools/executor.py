# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from typing import Any

from autogen.beta.annotations import Context
from autogen.beta.events import (
    ClientToolCallEvent,
    ModelMessage,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    ToolErrorEvent,
    ToolNotFoundEvent,
    ToolResultEvent,
    ToolResultsEvent,
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
        stack.enter_context(context.stream.where(ToolCallsEvent).sub_scope(self.execute_tools))

        for tool in tools:
            tool.register(stack, context, middleware=middleware)

        # fallback subscriber to raise NotFound event
        stack.enter_context(
            context.stream.where(ToolCallEvent).sub_scope(_tool_not_found(known_tools)),
        )

    async def execute_tools(self, event: ToolCallsEvent, context: Context) -> None:
        results: list[ToolErrorEvent | ToolResultEvent] = []
        client_calls: list[ClientToolCallEvent] = []

        # Execute called tools in parallel
        for event in await asyncio.gather(*(_execute_call(context, call) for call in event.calls)):
            match event:
                case ClientToolCallEvent() as ev:
                    client_calls.append(ev)

                case ToolErrorEvent() as ev:
                    results.append(ev)

                case ToolResultEvent(result=result) as ev:
                    if result.final:
                        await context.send(
                            ModelResponse(
                                message=ModelMessage(ev.content),
                                response_force=True,
                            )
                        )
                        return
                    else:
                        results.append(ev)

                case ev:
                    results.append(ev)

        if client_calls:
            await context.send(
                ModelResponse(
                    tool_calls=ToolCallsEvent(client_calls),
                    response_force=True,
                )
            )

        else:
            await context.send(ToolResultsEvent(results))


async def _execute_call(
    context: Context, call: ToolCallEvent
) -> ToolErrorEvent | ToolResultEvent | ClientToolCallEvent:
    async with context.stream.get(
        (ToolErrorEvent.parent_id == call.id) | (ToolResultEvent.parent_id == call.id) | ClientToolCallEvent
    ) as result:
        await context.send(call)
        return await result


def _tool_not_found(known_tools: Iterable[str]) -> Callable[..., Any]:
    async def _tool_not_found(event: "ToolCallEvent", context: "Context") -> None:
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
