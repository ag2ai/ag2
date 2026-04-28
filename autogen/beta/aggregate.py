# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""AggregateStrategy — organizes knowledge for sustained performance.

Aggregation extracts structured knowledge from raw events and writes it
to the knowledge store. This is the knowledge-organizing operation:
triggered at deterministic milestones to maintain agent effectiveness.

Unlike compaction (which removes), aggregation creates.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from fast_depends.pydantic import PydanticSerializer

from autogen.beta.annotations import Context
from autogen.beta.config import ModelConfig
from autogen.beta.context import ConversationContext
from autogen.beta.events import BaseEvent, ModelRequest
from autogen.beta.stream import MemoryStream

from .knowledge import CONVERSATIONS_PREFIX, WORKING_MEMORY_PATH, KnowledgeStore


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
            store: Agent's knowledge store to write into.
        """
        ...


@dataclass(slots=True)
class AggregateTrigger:
    """Deterministic conditions for triggering aggregation.

    Multiple conditions can be set. Each fires independently.
    """

    every_n_turns: int = 0  # Aggregate every N LLM turns. 0 = disabled.
    every_n_events: int = 0  # Aggregate every N new events since last aggregation. 0 = disabled.
    on_end: bool = False  # Aggregate when conversation ends. Opt-in: each strategy is one LLM call.


class ConversationSummaryAggregate:
    """Summarize conversation and write to /memory/conversations/.

    Creates a per-conversation summary in the knowledge store.
    Costs one LLM call per aggregation.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._serializer = PydanticSerializer(
            pydantic_config={"arbitrary_types_allowed": True},
            use_fastdepends_errors=False,
        )
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
        client = self._config.create()
        prompt_event = ModelRequest.ensure_request([
            "Summarize this conversation. Include key decisions, "
            "findings, outcomes, and any unfinished work:\n\n" + "\n".join(str(e) for e in events)
        ])
        response = await client(
            [prompt_event],
            ConversationContext(MemoryStream()),
            tools=[],
            response_schema=None,
            serializer=self._serializer,
        )
        self.last_usage = response.usage if hasattr(response, "usage") and response.usage else {}
        return response.content or ""


class WorkingMemoryAggregate:
    """Update /memory/working.md with latest context.

    Reads existing working memory, merges with new events, writes
    updated working memory. The agent starts each new conversation
    with this as context (via WorkingMemoryPolicy).

    Costs one LLM call per aggregation.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._serializer = PydanticSerializer(
            pydantic_config={"arbitrary_types_allowed": True},
            use_fastdepends_errors=False,
        )
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
        client = self._config.create()
        prompt = (
            "You maintain an agent's working memory. Update it based on "
            "the new conversation below. Preserve important existing context. "
            "Remove outdated information. Keep it concise and actionable.\n\n"
            f"## Current Working Memory\n{existing or '(empty)'}\n\n"
            "## New Conversation\n" + "\n".join(str(e) for e in events)
        )
        response = await client(
            [ModelRequest.ensure_request([prompt])],
            ConversationContext(MemoryStream()),
            tools=[],
            response_schema=None,
            serializer=self._serializer,
        )
        self.last_usage = response.usage if hasattr(response, "usage") and response.usage else {}
        return response.content or existing
