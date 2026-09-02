# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any, cast

from ag2.annotations import Context
from ag2.context import Stream
from ag2.events import BaseEvent, ModelResponse
from ag2.middleware.base import BaseMiddleware, LLMCall, MiddlewareFactory
from ag2.middleware.describe import MiddlewareDescription


class _AttemptStream:
    """Stream proxy that records whether an attempt published an event."""

    __slots__ = ("_inner", "has_emitted")

    def __init__(self, inner: Stream) -> None:
        self._inner = inner
        self.has_emitted = False

    async def send(self, event: BaseEvent, context: Context) -> None:
        self.has_emitted = True
        await self._inner.send(event, context)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


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
            original_stream = context.stream
            attempt_stream = _AttemptStream(original_stream)
            context.stream = cast(Stream, attempt_stream)
            try:
                return await call_next(events, context)
            except self._retry_on:
                if attempt_stream.has_emitted:
                    raise
            finally:
                if context.stream is attempt_stream:
                    context.stream = original_stream
        # Final attempt — let the original exception propagate.
        return await call_next(events, context)
