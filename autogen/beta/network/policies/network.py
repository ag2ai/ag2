# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""NetworkPolicy — includes network events in the LLM context."""

from __future__ import annotations

from autogen.beta.context import Context
from autogen.beta.events import BaseEvent
from autogen.beta.events.base import Field

from autogen.beta.compact import CompactionSummary
from autogen.beta.events.alert import ObserverAlert
from autogen.beta.policies.conversation import CONVERSATION_TYPES

from ..events import DelegationResult, SchedulerTriggerFired, TopicMessage


class FormattedEvent(BaseEvent):
    """Wraps a policy-formatted string as a synthetic event.

    Used by NetworkPolicy to pass formatted events through to the LLM
    client while preserving the original event for reference.
    """

    content: str
    role: str = Field(default="system")
    original: BaseEvent | None = Field(default=None)


_NETWORK_TYPES = CONVERSATION_TYPES + (
    DelegationResult,
    ObserverAlert,
    SchedulerTriggerFired,
    TopicMessage,
    FormattedEvent,
)


class NetworkPolicy:
    """Includes network events in the LLM context.

    The actor sees delegation results, signals, scheduler events,
    and topic messages alongside conversation -- enabling
    network-aware reasoning.
    """

    name = "network"

    async def apply(
        self,
        prompts: list[str],
        events: list[BaseEvent],
        context: Context,
    ) -> tuple[list[str], list[BaseEvent]]:
        filtered = [e for e in events if isinstance(e, _NETWORK_TYPES)]

        # Format network events
        formatted: list[BaseEvent] = []
        for event in filtered:
            fmt = self._format(event)
            if fmt is not None:
                formatted.append(FormattedEvent(content=fmt, original=event))
            else:
                formatted.append(event)
        return prompts, formatted

    def _format(self, event: BaseEvent) -> str | None:
        if isinstance(event, ObserverAlert):
            level = event.severity.upper() if isinstance(event.severity, str) else str(event.severity)
            return f"[ALERT/{level}] ({event.source}): {event.message}"
        if isinstance(event, DelegationResult):
            return f"[DELEGATION RESULT] {event.source} \u2192 {event.target}: {event.result}"
        if isinstance(event, SchedulerTriggerFired):
            return f"[SCHEDULED] Trigger '{event.watch_id}' fired for {event.target}"
        if isinstance(event, TopicMessage):
            return f"[TOPIC/{event.topic}] {event.sender}: {event.message}"
        if isinstance(event, CompactionSummary):
            return f"[CONTEXT SUMMARY] ({event.event_count} earlier events)\n{event.summary}"
        return None
