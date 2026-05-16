# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import random
from collections.abc import Sequence

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, HumanInputRequest, HumanMessage, ModelResponse, ToolCallEvent
from autogen.beta.middleware.base import (
    AgentTurn,
    BaseMiddleware,
    HumanInputHook,
    LLMCall,
    MiddlewareFactory,
    ToolExecution,
    ToolResultType,
)


class SamplingMiddleware(MiddlewareFactory):
    """Activates an inner middleware on a random fraction of agent turns.

    One sampling decision is made per turn — all hooks within that turn
    either all run through the inner middleware or all bypass it.  This
    keeps traces coherent: if a turn is sampled you see the full picture
    (on_turn, on_llm_call, on_tool_execution …), never a partial one.

    Args:
        rate: Probability of activating the inner middleware, in ``[0.0, 1.0]``.
            ``1.0`` means always active (equivalent to using the middleware
            directly); ``0.0`` means never active.
        middleware: The :class:`MiddlewareFactory` to activate when sampled.

    Example::

        from autogen.beta.middleware import Middleware, SamplingMiddleware
        from autogen.beta.middleware.builtin import LoggingMiddleware

        # Trace roughly 10% of all agent turns
        agent = Agent(
            "assistant",
            ...,
            middleware=[
                SamplingMiddleware(
                    rate=0.1,
                    middleware=LoggingMiddleware(),
                )
            ],
        )
    """

    def __init__(self, rate: float, middleware: MiddlewareFactory) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"SamplingMiddleware rate must be in [0.0, 1.0], got {rate!r}")
        self._rate = rate
        self._middleware = middleware

    def __call__(self, event: BaseEvent, context: Context) -> BaseMiddleware:
        sampled = random.random() < self._rate
        return _SamplingMiddlewareInstance(
            event,
            context,
            sampled=sampled,
            inner=self._middleware(event, context),
        )


class _SamplingMiddlewareInstance(BaseMiddleware):
    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        *,
        sampled: bool,
        inner: BaseMiddleware,
    ) -> None:
        super().__init__(event, context)
        self._sampled = sampled
        self._inner = inner

    async def on_turn(
        self,
        call_next: AgentTurn,
        event: BaseEvent,
        context: Context,
    ) -> ModelResponse:
        if not self._sampled:
            return await call_next(event, context)
        return await self._inner.on_turn(call_next, event, context)

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        if not self._sampled:
            return await call_next(events, context)
        return await self._inner.on_llm_call(call_next, events, context)

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        if not self._sampled:
            return await call_next(event, context)
        return await self._inner.on_tool_execution(call_next, event, context)

    async def on_human_input(
        self,
        call_next: HumanInputHook,
        event: HumanInputRequest,
        context: Context,
    ) -> HumanMessage:
        if not self._sampled:
            return await call_next(event, context)
        return await self._inner.on_human_input(call_next, event, context)
