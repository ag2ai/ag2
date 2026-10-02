# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from ag2.annotations import Context
from ag2.events import BaseEvent, ModelResponse
from ag2.middleware.base import BaseMiddleware, LLMCall, MiddlewareFactory
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

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        for _ in range(self._max_retries):
            # Watch for *any* event: an ``on_llm_call`` scope bottoms out at the
            # provider client, and the agent loop publishes its own events —
            # ``ModelRequest``, ``ToolCallsEvent``, ``UsageEvent`` — outside the
            # middleware chain, so whatever lands here is the client's own output:
            # streamed chunks and reasoning, the message, server-side tool calls.
            async with context.stream.get(BaseEvent) as published:
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
