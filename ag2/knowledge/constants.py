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

LOG_PREFIX = "/log/"
"""Stream event logs and dropped-events snapshots (from compaction)."""
