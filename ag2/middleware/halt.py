# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Stop model calls when a policy or observer emits a halt event."""

from collections.abc import Sequence

from ag2.annotations import Context
from ag2.events import BaseEvent, HaltEvent, ModelMessage, ModelResponse

from .base import AgentTurn, BaseMiddleware, LLMCall


class HaltCheckMiddleware(BaseMiddleware):
    """Short-circuit model calls after a ``HaltEvent`` during this turn.

    Register after ``AssemblerMiddleware`` so a FATAL alert emitted by
    ``AlertPolicy`` stops the same model call. Subscriptions are scoped to
    the turn and removed even when the turn fails or is cancelled.
    """

    def __init__(self, event: BaseEvent, context: Context) -> None:
        super().__init__(event, context)
        self._halted = False
        self._halt_reason = ""

    async def _on_halt(self, event: HaltEvent) -> None:
        self._halted = True
        self._halt_reason = event.reason

    async def on_turn(self, call_next: AgentTurn, event: BaseEvent, context: Context) -> ModelResponse:
        subscription = context.stream.where(HaltEvent).subscribe(self._on_halt)
        try:
            return await call_next(event, context)
        finally:
            context.stream.unsubscribe(subscription)

    async def on_llm_call(self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context) -> ModelResponse:
        if self._halted:
            return ModelResponse(message=ModelMessage(content=f"HALTED: {self._halt_reason}"))
        return await call_next(events, context)
