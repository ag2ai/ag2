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


class _RetryAttempt:
    """Records whether one attempt has already put an event on the stream.

    No event type is named here on purpose. An ``on_llm_call`` scope bottoms out
    at the provider client, and the agent loop publishes its own events —
    ``ModelRequest``, ``ToolCallsEvent``, ``UsageEvent`` — outside the middleware
    chain. So everything reaching this hook is the client's own output: streamed
    chunks and reasoning, the message, server-side tool calls and results, provider
    replay carriers. All of it is equally unrecallable once sent, and a list of
    types would only be a copy of the clients' send sites, going stale the first
    time one of them learns a new event.

    Registered as an *interrupter* rather than a plain subscriber, because
    interrupters run first in ``Stream.send`` and may drop an event: whatever an
    earlier interrupter filtered out never reaches a subscriber, so it must not
    count as published here either. Returning the event unchanged is what hands it
    on — returning ``None`` would swallow it for every consumer.
    """

    __slots__ = ("published",)

    def __init__(self) -> None:
        self.published = False

    def on_published(self, event: BaseEvent) -> BaseEvent:
        self.published = True
        return event


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
            attempt = _RetryAttempt()
            with context.stream.sub_scope(
                attempt.on_published,
                interrupt=True,
                sync_to_thread=False,
            ):
                try:
                    return await call_next(events, context)
                except self._retry_on:
                    # An attempt that published nothing left no trace to contradict,
                    # so it is safely repeatable. One that published is not: the
                    # retry's output would be concatenated onto it by every live
                    # consumer, while the reply carries only the retry's.
                    if attempt.published:
                        raise
        # Final attempt — let the original exception propagate.
        return await call_next(events, context)
