# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ToolCallEvent
from autogen.beta.middleware.base import AgentTurn, BaseMiddleware, ModelResponse, ToolExecution, ToolResultType

from .session import DebugSession


class DebugMiddleware(BaseMiddleware):
    """Middleware that forwards agent events to the debug server."""

    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        *,
        session: DebugSession,
    ) -> None:
        super().__init__(event, context)
        self._session = session

    async def on_turn(
        self,
        call_next: AgentTurn,
        event: BaseEvent,
        context: Context,
    ) -> ModelResponse:
        return await call_next(event, context)

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        return await call_next(event, context)
