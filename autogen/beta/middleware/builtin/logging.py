# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from collections.abc import Sequence

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ModelResponse, ToolCall
from autogen.beta.middleware.base import (
    AgentTurn,
    BaseMiddleware,
    LLMCall,
    MiddlewareFactory,
    ToolExecution,
    ToolResultType,
)


class LoggingMiddleware(MiddlewareFactory):
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("autogen")

    def __call__(self, event: "BaseEvent", context: "Context") -> BaseMiddleware:
        return _LoggingMiddleware(event, context, self._logger)


class _LoggingMiddleware(BaseMiddleware):
    """Log LLM calls, tool executions, and turns with timing."""

    def __init__(self, event: "BaseEvent", context: Context, logger: logging.Logger) -> None:
        super().__init__(event, context)
        self._logger = logger

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        self._logger.info("LLM call: %s", events[-1])
        t0 = time.perf_counter()
        response = await call_next(events, context)
        elapsed = time.perf_counter() - t0
        self._logger.info("LLM response: %s in %.2fs", response, elapsed)
        return response

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: ToolCall,
        context: Context,
    ) -> "ToolResultType":
        self._logger.info("Tool execute: %s(%s)", event.name, event.arguments)
        result = await call_next(event, context)
        self._logger.info("Tool result: %s", result)
        return result

    async def on_turn(
        self,
        call_next: AgentTurn,
        event: BaseEvent,
        context: Context,
    ) -> ModelResponse:
        self._logger.info("Agent turn started")
        response = await call_next(event, context)
        self._logger.info("Agent turn finished")
        return response
