# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, HumanInputRequest, HumanMessage, ModelResponse, ToolCallEvent
from autogen.beta.events.conditions import Condition
from autogen.beta.middleware.base import (
    AgentTurn,
    BaseMiddleware,
    HumanInputHook,
    LLMCall,
    MiddlewareFactory,
    ToolExecution,
    ToolResultType,
)


class ConditionalMiddleware(MiddlewareFactory):
    """Activates an inner middleware only when a Condition matches the relevant event.

    Each hook checks the condition against its event independently:

    - ``on_turn``: condition evaluated against the turn-triggering event.
    - ``on_tool_execution``: condition evaluated against the :class:`ToolCallEvent`.
    - ``on_llm_call``: condition evaluated against the initial turn event.
    - ``on_human_input``: condition evaluated against the :class:`HumanInputRequest`.

    If the condition is not satisfied the hook short-circuits — ``call_next`` is
    invoked directly and the inner middleware layer is transparent for that hook.

    Args:
        condition: A :class:`~autogen.beta.events.conditions.Condition` instance.
            Build one from event-field DSL expressions
            (e.g. ``ToolCallEvent.name == "execute_code"``) or compose with
            ``&``, ``|``, ``~``.
        middleware: The :class:`MiddlewareFactory` to activate when the condition
            matches.

    Example::

        from autogen.beta.events import ToolCallEvent
        from autogen.beta.middleware import ConditionalMiddleware, Middleware
        from autogen.beta.middleware.builtin.tools import approval_required

        agent = Agent(
            "assistant",
            ...,
            middleware=[
                ConditionalMiddleware(
                    condition=ToolCallEvent.name == "execute_code",
                    middleware=Middleware(approval_required),
                )
            ],
        )
    """

    def __init__(self, condition: Condition, middleware: MiddlewareFactory) -> None:
        self._condition = condition
        self._middleware = middleware

    def __call__(self, event: BaseEvent, context: Context) -> BaseMiddleware:
        return _ConditionalMiddlewareInstance(
            event,
            context,
            condition=self._condition,
            inner=self._middleware(event, context),
        )


class _ConditionalMiddlewareInstance(BaseMiddleware):
    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        *,
        condition: Condition,
        inner: BaseMiddleware,
    ) -> None:
        super().__init__(event, context)
        self._condition = condition
        self._inner = inner

    async def on_turn(
        self,
        call_next: AgentTurn,
        event: BaseEvent,
        context: Context,
    ) -> ModelResponse:
        if not self._condition(event):
            return await call_next(event, context)
        return await self._inner.on_turn(call_next, event, context)

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        if not self._condition(self.initial_event):
            return await call_next(events, context)
        return await self._inner.on_llm_call(call_next, events, context)

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        if not self._condition(event):
            return await call_next(event, context)
        return await self._inner.on_tool_execution(call_next, event, context)

    async def on_human_input(
        self,
        call_next: HumanInputHook,
        event: HumanInputRequest,
        context: Context,
    ) -> HumanMessage:
        if not self._condition(event):
            return await call_next(event, context)
        return await self._inner.on_human_input(call_next, event, context)
