# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
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

from fast_depends.pydantic import PydanticSerializer

from ag2.annotations import Context
from ag2.config import ModelConfig
from ag2.context import ConversationContext
from ag2.events import (
    BaseEvent,
    ModelRequest,
    ModelResponse,
    ToolResultEvent,
    ToolResultsEvent,
    UsageEvent,
    is_conversational,
    render_for_prompt,
)
from ag2.stream import MemoryStream

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


def _retained_is_self_contained(events: list[BaseEvent], cut: int) -> bool:
    """True when no tool result in events[cut:] references a dropped tool call."""
    have: set[str] = set()
    need: set[str] = set()
    for event in events[cut:]:
        if isinstance(event, ModelResponse):
            have.update(call.id for call in event.tool_calls.calls)
        elif isinstance(event, ToolResultsEvent):
            need.update(r.parent_id for r in event.results if r.parent_id)
        elif isinstance(event, ToolResultEvent) and event.parent_id:
            need.add(event.parent_id)
    return need <= have


def _snap_to_turn_boundary(events: list[BaseEvent], cut: int) -> int:
    """Advance `cut` so a tool-cycle split by the boundary compacts whole."""
    while cut < len(events) and not _retained_is_self_contained(events, cut):
        cut += 1
    return cut


def _cut_for_target(events: list[BaseEvent], target: int) -> int:
    """Index such that ``events[cut:]`` holds the last ``target`` conversational
    events. Interleaved telemetry rides along free (kept for ``UsageReport``);
    returns 0 when there are fewer than ``target`` conversational events.
    """
    seen = 0
    for i in range(len(events) - 1, -1, -1):
        if is_conversational(events[i]):
            seen += 1
            if seen == target:
                return i
    return 0


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
        cut = _cut_for_target(events, self._target)
        if cut == 0:
            return events

        cut = _snap_to_turn_boundary(events, cut)
        dropped = events[:cut]
        retained = events[cut:]

        if not retained:
            return events

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
        self._serializer = PydanticSerializer(
            pydantic_config={"arbitrary_types_allowed": True},
            use_fastdepends_errors=False,
        )
        self.last_usage: dict = {}

    async def compact(
        self,
        events: list[BaseEvent],
        context: Context,
        store: KnowledgeStore | None,
    ) -> list[BaseEvent]:
        cut = _cut_for_target(events, self._target)
        if cut == 0:
            return events

        cut = _snap_to_turn_boundary(events, cut)
        old = events[:cut]
        recent = events[cut:]

        if not recent:
            return events

        # Persist full old events before dropping
        if store:
            writer = EventLogWriter(store)
            await writer.persist_dropped(context.stream.id, old)

        # Summarize via LLM
        summary_text = await self._summarize(old, context)
        summary_event = CompactionSummary(
            summary=summary_text,
            event_count=len(old),
        )
        return [summary_event] + recent

    async def _summarize(self, events: list[BaseEvent], context: Context) -> str:
        client = self._config.create()
        prompt_event = ModelRequest.ensure_request([
            "Summarize the following conversation history concisely, "
            "preserving key decisions, findings, and context:\n\n"
            + "\n".join(render_for_prompt(e) for e in events if is_conversational(e))
        ])
        response = await client(
            [prompt_event],
            ConversationContext(MemoryStream()),
            tools=[],
            response_schema=None,
            serializer=self._serializer,
        )
        self.last_usage = response.usage if hasattr(response, "usage") and response.usage else {}
        # The summarization call runs on a throwaway stream; surface its usage
        # onto the real agent stream so it isn't lost to monitoring.
        if response.usage:
            await context.send(
                UsageEvent(
                    response.usage,
                    kind="compaction",
                    model=response.model,
                    provider=response.provider,
                )
            )
        return response.content or ""
