# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""EpisodicMemoryPolicy — injects past conversation summaries."""

from autogen.beta.context import ConversationContext as Context
from autogen.beta.events import BaseEvent
from autogen.beta.knowledge import CONVERSATIONS_PREFIX, KnowledgeStore


class EpisodicMemoryPolicy:
    """Injects past conversation summaries from the knowledge store.

    Reads /memory/conversations/ and injects the most recent summaries
    into the system prompt. This gives the agent context about past episodes.
    """

    name = "episodic_memory"

    def __init__(self, max_episodes: int = 5, transparent: bool = True) -> None:
        self._max = max_episodes
        self._transparent = transparent

    async def apply(
        self,
        prompts: list[str],
        events: list[BaseEvent],
        context: Context,
    ) -> tuple[list[str], list[BaseEvent]]:
        store = context.dependencies.get(KnowledgeStore)
        if not store:
            return prompts, events

        entries = await store.list(CONVERSATIONS_PREFIX)
        if not entries:
            return prompts, events

        # Read most recent summaries
        recent = entries[-self._max :]
        summaries: list[str] = []
        for entry in recent:
            content = await store.read(f"{CONVERSATIONS_PREFIX}{entry}")
            if content:
                summaries.append(content)

        if summaries:
            block = "## Past Conversations\n\n" + "\n\n---\n\n".join(summaries)
            prompts = prompts + [block]
            if self._transparent:
                prompts = prompts + [f"[{self.name}] Injected {len(summaries)} past conversation summaries."]

        return prompts, events
