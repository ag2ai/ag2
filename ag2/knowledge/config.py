# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from ..aggregate import AggregateStrategy, AggregateTrigger
from ..compact import CompactStrategy, CompactTrigger
from .base import KnowledgeStore
from .bootstrap import StoreBootstrap


@dataclass(slots=True)
class KnowledgeConfig:
    """Groups knowledge-related Agent parameters.

    The ``store`` is registered into ``context.dependencies[KnowledgeStore]``
    so policies (e.g. ``WorkingMemoryPolicy``, ``EpisodicMemoryPolicy``) can
    read from it without an extra parameter. Everything else is a side
    effect of attaching a store; each is opt-out via its flag below.

    Attributes:
        store: The backing knowledge store.
        expose_tool: If True (default), the agent gets an auto-injected
            ``knowledge`` action-group tool that lets the LLM call
            ``read`` / ``write`` / ``list`` / ``delete`` on the store. Set
            to False when policies are the only consumer of the store and
            the LLM should not see it.
        write_event_log: If True (default), the agent persists its stream
            history to ``/log/{stream_id}.jsonl`` at the end of each
            ``ask`` call. Set to False to keep the store free of stream
            logs (useful when the store is purely user-facing memory).
        compact, compact_trigger: Optional compaction strategy and its
            firing rules.
        aggregate, aggregate_trigger: Optional aggregation strategy and
            its firing rules. Strategy failures emit ``AggregationFailed``
            on the stream — subscribe to that event for observability.
        bootstrap: Optional custom bootstrap. None falls back to
            ``DefaultBootstrap(mention_tool=expose_tool)``, so the
            generated SKILL.md text tells the LLM about the ``knowledge``
            tool only when the tool is actually exposed.
    """

    store: KnowledgeStore
    expose_tool: bool = True
    write_event_log: bool = True
    compact: CompactStrategy | None = None
    compact_trigger: CompactTrigger | None = None
    aggregate: AggregateStrategy | None = None
    aggregate_trigger: AggregateTrigger | None = None
    bootstrap: StoreBootstrap | None = None
