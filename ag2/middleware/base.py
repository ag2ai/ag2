# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias

from ag2.annotations import Context
from ag2.events import (
    BaseEvent,
    ClientToolCallEvent,
    Condition,
    HumanInputRequest,
    HumanMessage,
    ModelResponse,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
)
from ag2.events.conditions import TypeCondition
from ag2.middleware.describe import MiddlewareDescription, _describe


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

    @property
    def cls(self) -> type["BaseMiddleware"]:
        """The middleware class this factory instantiates per turn."""

        return self._cls

    @property
    def options(self) -> Mapping[str, Any]:
        """The options this factory was constructed with.

        Unlike :meth:`describe`, this exposes the values. A description is meant
        to be logged or committed as a fixture, so it reports option names only;
        reading an option off a factory you already hold is a deliberate act.
        """

        return MappingProxyType(self._options)

    def describe(self) -> "MiddlewareDescription":
        """Report the wrapped class and option names, but not option values.

        Always incomplete: option values are caller-supplied and may hold
        credentials, and the wrapped class is only instantiated per turn.
        """

        return MiddlewareDescription(
            kind=self._cls.__qualname__,
            config={"options": tuple(sorted(self._options))},
            complete=False,
        )

    def __call__(
        self,
        event: "BaseEvent",
        context: "Context",
    ) -> "BaseMiddleware":
        return self._cls(event, context, **self._options)


ToolResultType: TypeAlias = "ToolResultEvent | ToolErrorEvent | ClientToolCallEvent"
AgentTurn: TypeAlias = Callable[["BaseEvent", "Context"], Awaitable["ModelResponse"]]
LLMCall: TypeAlias = Callable[["Sequence[BaseEvent]", "Context"], Awaitable["ModelResponse"]]
HumanInputHook: TypeAlias = Callable[["HumanInputRequest", "Context"], Awaitable["HumanMessage"]]

ToolExecution: TypeAlias = Callable[["ToolCallEvent", "Context"], Awaitable[ToolResultType]]
# call_next + ToolExecution type. BaseMiddleware.on_tool_execution() hook signature.
ToolMiddleware: TypeAlias = Callable[[ToolExecution, "ToolCallEvent", "Context"], Awaitable[ToolResultType]]


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
        event: "ToolCallEvent",
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

    async def on_human_input(
        self,
        call_next: HumanInputHook,
        event: "HumanInputRequest",
        context: "Context",
    ) -> "HumanMessage":
        return await call_next(event, context)


class _ConditionalWrapper(BaseMiddleware):
    def __init__(
        self,
        event: "BaseEvent",
        context: "Context",
        inner: BaseMiddleware,
        condition: Condition,
    ) -> None:
        super().__init__(event, context)
        self._inner = inner
        self._condition = condition

    async def on_turn(
        self,
        call_next: AgentTurn,
        event: "BaseEvent",
        context: "Context",
    ) -> "ModelResponse":
        if self._condition(event):
            return await self._inner.on_turn(call_next, event, context)
        return await call_next(event, context)

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: "ToolCallEvent",
        context: "Context",
    ) -> ToolResultType:
        if self._condition(event):
            return await self._inner.on_tool_execution(call_next, event, context)
        return await call_next(event, context)

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: "Sequence[BaseEvent]",
        context: "Context",
    ) -> "ModelResponse":
        if self._condition(self.initial_event):
            return await self._inner.on_llm_call(call_next, events, context)
        return await call_next(events, context)

    async def on_human_input(
        self,
        call_next: HumanInputHook,
        event: "HumanInputRequest",
        context: "Context",
    ) -> "HumanMessage":
        if self._condition(event):
            return await self._inner.on_human_input(call_next, event, context)
        return await call_next(event, context)


class ConditionalMiddleware:
    """Middleware wrapper that evaluates a condition per-hook."""

    def __init__(
        self,
        middleware: "MiddlewareFactory",
        condition: "Condition | type",
    ) -> None:
        self._middleware = middleware
        self._condition = condition if isinstance(condition, Condition) else TypeCondition(condition)

    def describe(self) -> "MiddlewareDescription":
        return MiddlewareDescription(
            kind=type(self).__qualname__,
            config={"condition": type(self._condition).__qualname__},
            inner=(_describe(self._middleware),),
        )

    def __call__(
        self,
        event: "BaseEvent",
        context: "Context",
    ) -> "BaseMiddleware":
        inner = self._middleware(event, context)
        return _ConditionalWrapper(event, context, inner=inner, condition=self._condition)
