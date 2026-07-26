# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from ag2.annotations import Context
from ag2.events import BaseEvent, ModelResponse
from ag2.middleware.base import BaseMiddleware, LLMCall, MiddlewareFactory


class RetryMiddleware(MiddlewareFactory):
    def __init__(
        self,
        max_retries: int = 3,
        retry_on: tuple[type[Exception], ...] = (Exception,),
    ):
        self._max_retries = max_retries
        self._retry_on = retry_on

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
            try:
                return await call_next(events, context)
            except self._retry_on:
                pass
        # Final attempt — let the original exception propagate.
        return await call_next(events, context)
