# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ToolCallEvent
from autogen.beta.middleware.base import AgentTurn, BaseMiddleware, ToolExecution, ToolResultType

from .client import DebugClient


class DebugMiddleware(BaseMiddleware):
    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        *,
        client: DebugClient,
        session_id: str,
    ) -> None:
        super().__init__(event, context)
        self._client = client
        self._session_id = session_id

    async def on_turn(
        self,
        call_next: AgentTurn,
        event: BaseEvent,
        context: Context,
    ) -> "BaseEvent":
        await self._client.hit_breakpoint(self._session_id, "TURN_START", event)
        return await call_next(event, context)

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        await self._client.hit_breakpoint(self._session_id, "TOOL_CALL", event)
        return await call_next(event, context)
