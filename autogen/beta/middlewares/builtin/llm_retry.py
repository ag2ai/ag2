# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ModelResponse
from autogen.beta.middlewares.base import BaseMiddleware, LLMCall, MiddlewareFactory


class RetryMiddleware(MiddlewareFactory):
    def __init__(
        self,
        max_retries: int = 3,
        retry_on: tuple[type[Exception], ...] = (Exception,),
    ):
        self._max_retries = max_retries
        self._retry_on = retry_on

    def __call__(self, event: "BaseEvent", ctx: "Context") -> "BaseMiddleware":
        return _RetryMiddleware(
            event,
            ctx,
            max_retries=self._max_retries,
            retry_on=self._retry_on,
        )


class _RetryMiddleware(BaseMiddleware):
    """Retry LLM calls on transient failures."""

    def __init__(
        self,
        event: "BaseEvent",
        ctx: "Context",
        *,
        max_retries: int = 3,
        retry_on: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        super().__init__(event, ctx)
        self._max_retries = max_retries
        self._retry_on = retry_on

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        ctx: Context,
    ) -> ModelResponse:
        last_error: Exception | None = None
        for _ in range(self._max_retries + 1):
            try:
                return await call_next(events, ctx)
            except self._retry_on as e:
                last_error = e
        raise last_error
