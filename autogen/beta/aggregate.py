# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""AggregateStrategy — organizes knowledge for sustained performance.

Aggregation extracts structured knowledge from raw events and writes it
to the knowledge store. This is the knowledge-organizing operation:
triggered at deterministic milestones to maintain actor effectiveness.

Unlike compaction (which removes), aggregation creates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from autogen.beta.events import BaseEvent

from .knowledge import CONVERSATIONS_PREFIX, WORKING_MEMORY_PATH

if TYPE_CHECKING:
    from autogen.beta.annotations import Context
    from autogen.beta.config import ModelConfig

    from .knowledge import KnowledgeStore


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AggregateStrategy(Protocol):
    """Organizes knowledge for sustained performance.

    Extracts structured knowledge from raw events and writes it to the
    knowledge store.
    """

    async def aggregate(
        self,
        events: list[BaseEvent],
        context: Context,
        store: KnowledgeStore,
    ) -> None:
        """Extract and store knowledge.

        Args:
            events: Current stream history.
            context: Execution context.
            store: Actor's knowledge store to write into.
        """
        ...


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AggregateTrigger:
    """Deterministic conditions for triggering aggregation.

    Multiple conditions can be set. Each fires independently.
    """

    every_n_turns: int = 0  # Aggregate every N LLM turns. 0 = disabled.
    every_n_events: int = 0  # Aggregate every N new events since last aggregation. 0 = disabled.
    on_end: bool = True  # Aggregate when conversation ends.


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


class ConversationSummaryAggregate:
    """Summarize conversation and write to /memory/conversations/.

    Creates a per-conversation summary in the knowledge store.
    Costs one LLM call per aggregation.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.last_usage: dict = {}

    async def aggregate(
        self,
        events: list[BaseEvent],
        context: Context,
        store: KnowledgeStore,
    ) -> None:
        if not events:
            return
        summary = await self._summarize(events)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        stream_id = str(context.stream.id)
        await store.write(f"{CONVERSATIONS_PREFIX}{ts}_{stream_id}.md", summary)

    async def _summarize(self, events: list[BaseEvent]) -> str:
        from autogen.beta.context import ConversationContext as Ctx
        from autogen.beta.events import ModelRequest
        from autogen.beta.stream import MemoryStream

        client = self._config.create()
        prompt_event = ModelRequest.ensure_request([
            "Summarize this conversation. Include key decisions, "
            "findings, outcomes, and any unfinished work:\n\n" + "\n".join(str(e) for e in events)
        ])
        response = await client(
            [prompt_event],
            Ctx(MemoryStream()),
            tools=[],
            response_schema=None,
        )
        self.last_usage = response.usage if hasattr(response, "usage") and response.usage else {}
        return response.content or ""


class WorkingMemoryAggregate:
    """Update /memory/working.md with latest context.

    Reads existing working memory, merges with new events, writes
    updated working memory. The actor starts each new conversation
    with this as context (via WorkingMemoryPolicy).

    Costs one LLM call per aggregation.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.last_usage: dict = {}

    async def aggregate(
        self,
        events: list[BaseEvent],
        context: Context,
        store: KnowledgeStore,
    ) -> None:
        if not events:
            return
        existing = await store.read(WORKING_MEMORY_PATH) or ""
        updated = await self._merge(existing, events)
        await store.write(WORKING_MEMORY_PATH, updated)

    async def _merge(self, existing: str, events: list[BaseEvent]) -> str:
        from autogen.beta.context import ConversationContext as Ctx
        from autogen.beta.events import ModelRequest
        from autogen.beta.stream import MemoryStream

        client = self._config.create()
        prompt = (
            "You maintain an actor's working memory. Update it based on "
            "the new conversation below. Preserve important existing context. "
            "Remove outdated information. Keep it concise and actionable.\n\n"
            f"## Current Working Memory\n{existing or '(empty)'}\n\n"
            "## New Conversation\n" + "\n".join(str(e) for e in events)
        )
        response = await client(
            [ModelRequest.ensure_request([prompt])],
            Ctx(MemoryStream()),
            tools=[],
            response_schema=None,
        )
        self.last_usage = response.usage if hasattr(response, "usage") and response.usage else {}
        return response.content or existing
