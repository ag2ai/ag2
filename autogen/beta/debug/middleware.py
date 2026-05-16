# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Sequence

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ToolCallEvent
from autogen.beta.middleware.base import (
    AgentTurn,
    BaseMiddleware,
    LLMCall,
    ModelResponse,
    ToolExecution,
    ToolResultType,
)

from .session import DEBUG_SESSION_VAR, DebugSession


class DebugMiddleware(BaseMiddleware):
    """Middleware that forwards agent events to the debug server.

    When ``AG2_DEBUG_SERVER_URL`` is not set and no :class:`DebugSession`
    exists in *context.variables*, the middleware is a silent pass-through.
    """

    def __init__(
        self,
        event: BaseEvent,
        context: Context,
    ) -> None:
        super().__init__(event, context)

        session: DebugSession | None = context.variables.get(DEBUG_SESSION_VAR)
        self._auto_session = False
        self._enabled = True

        if session is None:
            if not os.environ.get("AG2_DEBUG_SERVER_URL"):
                self._enabled = False
            else:
                session = DebugSession()
                context.variables[DEBUG_SESSION_VAR] = session
                self._auto_session = True

        self._session = session

    async def on_turn(
        self,
        call_next: AgentTurn,
        event: BaseEvent,
        context: Context,
    ) -> ModelResponse:
        if not self._enabled:
            return await call_next(event, context)

        await self._session.record_event(event, context)
        try:
            return await call_next(event, context)
        finally:
            if self._auto_session:
                await self._session.close()

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        if not self._enabled:
            return await call_next(events, context)

        result = await call_next(events, context)
        await self._session.record_event(result, context)
        return result

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        if not self._enabled:
            return await call_next(event, context)

        await self._session.record_event(event, context)
        result = await call_next(event, context)
        await self._session.record_event(result, context)
        return result
