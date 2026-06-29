# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
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

from .annotations import Context
from .config import ModelConfig
from .events import BaseEvent, ModelRequest, UsageEvent, render_for_prompt
from .knowledge import CONVERSATIONS_PREFIX, WORKING_MEMORY_PATH, KnowledgeStore
from .stream import MemoryStream


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
        summary = await self._summarize(events, context)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        stream_id = str(context.stream.id)
        await store.write(f"{CONVERSATIONS_PREFIX}{ts}_{stream_id}.md", summary)

    async def _summarize(self, events: list[BaseEvent], context: Context) -> str:
        client = self._config.create()
        prompt_event = ModelRequest.ensure_request([
            "Summarize this conversation. Include key decisions, "
            "findings, outcomes, and any unfinished work:\n\n" + "\n".join(render_for_prompt(e) for e in events)
        ])
        response = await client(
            [prompt_event],
            Context(MemoryStream()),
            tools=[],
            response_schema=None,
            serializer=self._serializer,
        )
        self.last_usage = response.usage if hasattr(response, "usage") and response.usage else {}
        await _emit_aggregation_usage(response, context)
        return response.content or ""


DEFAULT_WORKING_MEMORY_PROMPT = (
    "You maintain an agent's working memory. Update it based on "
    "the new conversation below. Preserve important existing context. "
    "Remove outdated information. Keep it concise and actionable.\n\n"
    "## Current Working Memory\n{existing}\n\n"
    "## New Conversation\n{events}"
)
"""Default prompt template for ``WorkingMemoryAggregate``.

Placeholders ``{existing}`` and ``{events}`` are substituted with the
current ``/memory/working.md`` contents (or ``(empty)``) and a textual
rendering of the new events. Pass a custom template via the ``prompt``
keyword argument; either placeholder may be omitted if not needed.
"""


class WorkingMemoryAggregate:
    """Update /memory/working.md with latest context.

    Reads existing working memory, merges with new events, writes
    updated working memory. The agent starts each new conversation
    with this as context (via ``WorkingMemoryPolicy``).

    The built-in prompt is journal-style: preserve facts that are still
    relevant, drop outdated content. For other memory shapes — procedural
    memory (what tactics worked), reflection (what to do differently
    next time), or task-state memory — override ``prompt`` with a
    template that uses the ``{existing}`` and ``{events}`` placeholders.

    When ``prompt`` is not enough — for example, you need a different
    storage path, multi-call extraction, or schema-validated output —
    write a small class that satisfies :class:`AggregateStrategy`:

    .. code-block:: python

        class ResearchLessonsAggregate:
            def __init__(self, config: ModelConfig) -> None:
                self._config = config
                self.last_usage: dict = {}

            async def aggregate(self, events, context, store) -> None:
                # ...your own logic, your own paths, your own prompts.

    Costs one LLM call per aggregation.

    Args:
        config: Model config used for the merge LLM call.
        prompt: Optional override template. Use ``{existing}`` and
            ``{events}`` placeholders. Defaults to
            :data:`DEFAULT_WORKING_MEMORY_PROMPT`.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        prompt: str = DEFAULT_WORKING_MEMORY_PROMPT,
    ) -> None:
        self._config = config
        self._prompt_template = prompt
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
        updated = await self._merge(existing, events, context)
        await store.write(WORKING_MEMORY_PATH, updated)

    async def _merge(self, existing: str, events: list[BaseEvent], context: Context) -> str:
        client = self._config.create()
        prompt = self._prompt_template.format(
            existing=existing or "(empty)",
            events="\n".join(render_for_prompt(e) for e in events),
        )
        response = await client(
            [ModelRequest.ensure_request([prompt])],
            Context(MemoryStream()),
            tools=[],
            response_schema=None,
            serializer=self._serializer,
        )
        self.last_usage = response.usage if hasattr(response, "usage") and response.usage else {}
        await _emit_aggregation_usage(response, context)
        return response.content or existing


async def _emit_aggregation_usage(response: BaseEvent, context: Context) -> None:
    """Surface an aggregation LLM call's usage onto the real agent stream.

    The aggregation call runs on a throwaway stream, so without this its tokens
    would never reach the event log the usage report reads from.
    """
    usage = getattr(response, "usage", None)
    if usage:
        await context.send(
            UsageEvent(
                usage,
                kind="aggregation",
                model=getattr(response, "model", None),
                provider=getattr(response, "provider", None),
            )
        )
