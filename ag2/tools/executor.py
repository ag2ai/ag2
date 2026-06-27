# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Callable, Iterable
from contextlib import AsyncExitStack, ExitStack
from typing import Any

from fast_depends.library.serializer import SerializerProto

from ag2.annotations import Context
from ag2.events import (
    ClientToolCallEvent,
    DataInput,
    ModelMessage,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolErrorEvent,
    ToolNotFoundEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from ag2.exceptions import ToolNotFoundError
from ag2.middleware import BaseMiddleware

from .tool import Tool


class ToolExecutor:
    def __init__(self, serializer: SerializerProto) -> None:
        self.__serializer = serializer

    def register(
        self,
        stack: "ExitStack | AsyncExitStack",
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
                        if len(result.parts) != 1:
                            raise ValueError("ToolResult with final=True must have exactly one part")
                        part = result.parts[0]
                        if isinstance(part, TextInput):
                            content = part.content
                        elif isinstance(part, DataInput):
                            content = self.__serializer.encode(part.data).decode()
                        else:
                            raise ValueError(f"Unsupported part type: {type(part)}")

                        await context.send(
                            ModelResponse(
                                message=ModelMessage(
                                    content,
                                    metadata=result.metadata,
                                ),
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
        (ToolErrorEvent.parent_id == call.id)
        | (ToolResultEvent.parent_id == call.id)
        | (ClientToolCallEvent.id == call.id)
    ) as result:
        await context.send(call)
        return await result


def _tool_not_found(known_tools: Iterable[str]) -> Callable[..., Any]:
    async def _tool_not_found(event: "ToolCallEvent", context: "Context") -> None:
        if event.name not in known_tools:
            err = ToolNotFoundError(event.name)
            # Build via from_call so the event always carries a populated
            # ``result`` (the formatted error). Constructing it by hand is what
            # left ``result`` as ``None`` and crashed the provider mappers.
            await context.send(ToolNotFoundEvent.from_call(event, err))

    return _tool_not_found
