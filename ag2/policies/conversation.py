# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""ConversationPolicy — only conversation and tool events reach the LLM."""

from ag2.compact import CompactionSummary
from ag2.context import ConversationContext as Context
from ag2.events import (
    BaseEvent,
    ModelRequest,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    ToolErrorEvent,
    ToolResultEvent,
    ToolResultsEvent,
)

# Event types that are always part of conversation context
CONVERSATION_TYPES = (
    ModelRequest,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
    ToolErrorEvent,
    CompactionSummary,
)


class ConversationPolicy:
    """Only conversation and tool events reach the LLM.

    Opt-in policy that re-creates plain-loop semantics: only conversation
    and tool events reach the LLM, everything else is filtered out.
    CompactionSummary is included so compacted summaries remain visible.
    """

    name = "conversation"

    async def apply(
        self,
        prompts: list[str],
        events: list[BaseEvent],
        context: Context,
    ) -> tuple[list[str], list[BaseEvent]]:
        filtered = [e for e in events if isinstance(e, CONVERSATION_TYPES)]
        return prompts, filtered
