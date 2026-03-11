# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol, TypeAlias

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ClientToolCall, ModelResponse, ToolCall, ToolError, ToolResult


class MiddlewareFactory(Protocol):
    def __call__(self, event: "BaseEvent", context: "Context") -> "BaseMiddleware": ...


class Middleware(MiddlewareFactory):
    """Public class to simplify middleware registration."""

    def __init__(
        self,
        middleware_cls: type["BaseMiddleware"],
        **kwargs: Any,
    ) -> None:
        self._cls = middleware_cls
        self._options = kwargs

    def __call__(
        self,
        event: "BaseEvent",
        context: "Context",
    ) -> "BaseMiddleware":
        return self._cls(event, context, **self._options)


ToolResultType: TypeAlias = "ToolResult | ToolError | ClientToolCall"
AgentTurn: TypeAlias = Callable[["BaseEvent", "Context"], Awaitable["ModelResponse"]]
ToolExecution: TypeAlias = Callable[["ToolCall", "Context"], Awaitable[ToolResultType]]
LLMCall: TypeAlias = Callable[["Sequence[BaseEvent]", "Context"], Awaitable["ModelResponse"]]


class BaseMiddleware:
    def __init__(
        self,
        event: "BaseEvent",
        context: "Context",
    ) -> None:
        self.initial_event = event
        self.context = context

    async def on_turn(
        self,
        call_next: AgentTurn,
        event: "BaseEvent",
        context: "Context",
    ) -> "ModelResponse":
        return await call_next(event, context)

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: "ToolCall",
        context: "Context",
    ) -> ToolResultType:
        return await call_next(event, context)

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: "Sequence[BaseEvent]",
        context: "Context",
    ) -> "ModelResponse":
        return await call_next(events, context)
