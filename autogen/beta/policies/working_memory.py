# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""WorkingMemoryPolicy — injects persistent working memory."""

from autogen.beta.context import ConversationContext as Context
from autogen.beta.events import BaseEvent
from autogen.beta.knowledge import WORKING_MEMORY_PATH, KnowledgeStore


class WorkingMemoryPolicy:
    """Injects /memory/working.md from the knowledge store.

    Working memory is the agent's persistent state -- updated by
    AggregateStrategy between conversations. This policy injects
    it as system prompt context so the agent has continuity.
    """

    name = "working_memory"

    async def apply(
        self,
        prompts: list[str],
        events: list[BaseEvent],
        context: Context,
    ) -> tuple[list[str], list[BaseEvent]]:
        store = context.dependencies.get(KnowledgeStore)
        if not store:
            return prompts, events

        content = await store.read(WORKING_MEMORY_PATH)
        if content:
            prompts = prompts + [f"## Working Memory\n\n{content}"]

        return prompts, events
