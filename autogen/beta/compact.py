# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""CompactStrategy — reduces stream history to respect system constraints.

Compaction protects runtime stability. It is the constraint-respecting
operation: triggered when measurable limits (event count, token count)
are approached. Returns a reduced event list that replaces the current
stream history.

Compaction removes. Aggregation creates. They are separate concerns.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from autogen.beta.annotations import Context
from autogen.beta.config import ModelConfig
from autogen.beta.context import ConversationContext
from autogen.beta.events import BaseEvent, ModelRequest
from autogen.beta.stream import MemoryStream

from .knowledge import EventLogWriter, KnowledgeStore


class CompactionSummary(BaseEvent):
    """Synthetic event replacing a sequence of compacted events.

    Created by SummarizeCompact (and similar strategies) to preserve
    context when old events are dropped. Assembly policies format it
    for LLM consumption.
    """

    summary: str
    event_count: int  # How many events were summarized


@runtime_checkable
class CompactStrategy(Protocol):
    """Reduces stream history to respect system constraints.

    Returns a reduced event list that replaces the current stream history.
    Must preserve causal ordering of retained events.
    """

    async def compact(
        self,
        events: list[BaseEvent],
        context: Context,
        store: KnowledgeStore | None,
    ) -> list[BaseEvent]:
        """Return compacted event list.

        Args:
            events: Current stream history.
            context: Execution context.
            store: Agent's knowledge store (for persisting dropped content).
                   None if not configured.
        """
        ...


@dataclass(slots=True)
class CompactTrigger:
    """Deterministic conditions for triggering compaction.

    Compaction fires when ANY condition is exceeded.
    """

    max_events: int = 0  # Compact when event count exceeds this. 0 = disabled.
    max_tokens: int = 0  # Compact when estimated token count exceeds this. 0 = disabled.
    chars_per_token: int = 4  # For token estimation.


class TailWindowCompact:
    """Keep the last N events. Drop the rest.

    Zero LLM cost. Simplest strategy. Suitable when old context
    has diminishing value and recent events are most relevant.
    """

    def __init__(self, target: int) -> None:
        self._target = target

    async def compact(
        self,
        events: list[BaseEvent],
        context: Context,
        store: KnowledgeStore | None,
    ) -> list[BaseEvent]:
        if len(events) <= self._target:
            return events

        dropped = events[: -self._target]
        retained = events[-self._target :]

        if store:
            writer = EventLogWriter(store)
            await writer.persist_dropped(context.stream.id, dropped)

        return retained


class SummarizeCompact:
    """Summarize old events into a CompactionSummary event, keep recent.

    Uses an LLM call to create a summary of dropped events. The summary
    becomes a CompactionSummary event at the head of the history.

    Costs one LLM call per compaction.
    """

    def __init__(self, target: int, config: ModelConfig) -> None:
        self._target = target
        self._config = config
        self.last_usage: dict = {}

    async def compact(
        self,
        events: list[BaseEvent],
        context: Context,
        store: KnowledgeStore | None,
    ) -> list[BaseEvent]:
        if len(events) <= self._target:
            return events

        old = events[: -self._target]
        recent = events[-self._target :]

        # Persist full old events before dropping
        if store:
            writer = EventLogWriter(store)
            await writer.persist_dropped(context.stream.id, old)

        # Summarize via LLM
        summary_text = await self._summarize(old)
        summary_event = CompactionSummary(
            summary=summary_text,
            event_count=len(old),
        )
        return [summary_event] + recent

    async def _summarize(self, events: list[BaseEvent]) -> str:
        client = self._config.create()
        prompt_event = ModelRequest.ensure_request([
            "Summarize the following conversation history concisely, "
            "preserving key decisions, findings, and context:\n\n" + "\n".join(str(e) for e in events)
        ])
        response = await client(
            [prompt_event],
            ConversationContext(MemoryStream()),
            tools=[],
            response_schema=None,
        )
        self.last_usage = response.usage if hasattr(response, "usage") and response.usage else {}
        return response.content or ""
