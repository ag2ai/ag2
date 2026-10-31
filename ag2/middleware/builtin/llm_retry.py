# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from ag2.annotations import Context
from ag2.events import BaseEvent, ModelRequest, ModelResponse
from ag2.events.conditions import TypeCondition
from ag2.middleware.base import AgentTurn, BaseMiddleware, LLMCall, MiddlewareFactory
from ag2.middleware.builtin.call_limiter import _CallGroup
from ag2.middleware.describe import MiddlewareDescription


class RetryMiddleware(MiddlewareFactory):
    def __init__(
        self,
        max_retries: int = 3,
        retry_on: tuple[type[Exception], ...] = (Exception,),
    ):
        self._max_retries = max_retries
        self._retry_on = retry_on

    def describe(self) -> MiddlewareDescription:
        return MiddlewareDescription(
            kind=type(self).__qualname__,
            config={
                "max_retries": self._max_retries,
                "retry_on": tuple(exc.__qualname__ for exc in self._retry_on),
            },
        )

    def __call__(self, event: "BaseEvent", context: "Context") -> "BaseMiddleware":
        return _RetryMiddleware(
            event,
            context,
            max_retries=self._max_retries,
            retry_on=self._retry_on,
        )


class _RetryMiddleware(BaseMiddleware):
    """Retry LLM calls on transient failures."""

    def __init__(
        self,
        event: "BaseEvent",
        context: "Context",
        *,
        max_retries: int = 3,
        retry_on: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        super().__init__(event, context)
        self._max_retries = max_retries
        self._retry_on = retry_on

    async def on_turn(self, call_next: AgentTurn, event: BaseEvent, context: Context) -> ModelResponse:
        group = context.dependencies.get(_CallGroup)
        if group is not None and group.collecting and group.stream_id == context.stream.id:
            raise ValueError("RetryMiddleware must wrap every CallLimiter so each provider attempt is counted")
        return await call_next(event, context)

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        for _ in range(self._max_retries):
            # Middleware can append control input (such as budget notes) inside
            # this scope. Only output makes a failed attempt unsafe to repeat.
            async with context.stream.get(~TypeCondition(ModelRequest)) as published:
                try:
                    return await call_next(events, context)
                except self._retry_on:
                    # An attempt that published nothing left no trace to contradict,
                    # so it is safely repeatable. One that published is not: the
                    # retry's output would be concatenated onto it by every live
                    # consumer, while the reply carries only the retry's.
                    if published.done():
                        raise
        # Final attempt — let the original exception propagate.
        return await call_next(events, context)
