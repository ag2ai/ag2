# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ToolCallEvent
from autogen.beta.middleware.base import AgentTurn, BaseMiddleware, ModelResponse, ToolExecution, ToolResultType

from .session import DebugSession


class DebugMiddleware(BaseMiddleware):
    """
    Middleware that pauses agent execution at each turn and tool call.

    Communicates directly with the in-process :class:`DebugSession` via
    ``asyncio.Event`` — no HTTP round-trip on the hot path.
    """

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
        event = await self._session.pause("TURN_START", event)
        return await call_next(event, context)

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        event = await self._session.pause("TOOL_CALL", event)
        return await call_next(event, context)
