# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Conventional paths shared between aggregation strategies, assembly policies,
and the event-log writer. Producer/consumer contracts: change one, change the
matching consumer.
"""

WORKING_MEMORY_PATH = "/memory/working.md"
"""Agent's persistent state. Produced by WorkingMemoryAggregate, read by WorkingMemoryPolicy."""

CONVERSATIONS_PREFIX = "/memory/conversations/"
"""Past-conversation summaries. Produced by ConversationSummaryAggregate, read by EpisodicMemoryPolicy."""

FACTS_PREFIX = "/memory/facts/"
"""Individual fact records with a lifecycle, one JSON file per fact.

Produced by FactStore (written by FactExtractionAggregate), read back by the
same store and indexed for retrieval. The file name is the fact id; each file
is a serialized Fact, and RetrievalMemoryPolicy never injects records whose
``status`` is ``superseded`` or ``forgotten``.
"""

INDEX_PREFIX = "/memory/.index/"
"""Retrieval index sidecars, written and read by StoreBackedIndex (ADR 0036).

The indexer's default exclude list keeps this prefix out of the index — the
index must never index itself.
"""

LOG_PREFIX = "/log/"
"""Stream event logs and dropped-events snapshots (from compaction)."""
