# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, runtime_checkable

from .base import KnowledgeStore
from .constants import LOG_PREFIX


@runtime_checkable
class StoreBootstrap(Protocol):
    """Initializes a knowledge store with a starting structure.

    Called once when an agent first runs with a store. Subsequent
    runs skip bootstrapping (detected via a sentinel file).
    """

    async def bootstrap(self, store: KnowledgeStore, actor_name: str) -> None:
        """Create initial store structure."""
        ...


class DefaultBootstrap:
    """Creates the standard knowledge store layout with SKILL.md files.

    Args:
        mention_tool: When True (default), the root ``SKILL.md`` instructs
            the LLM to use the ``knowledge`` tool. Set to False when the
            store is configured with ``expose_tool=False`` so the prompt
            does not reference a tool the LLM cannot call.
    """

    def __init__(self, mention_tool: bool = True) -> None:
        self._mention_tool = mention_tool

    async def bootstrap(self, store: KnowledgeStore, actor_name: str) -> None:
        if self._mention_tool:
            root_intro = "This is your persistent knowledge store. Use the `knowledge` tool to manage it."
        else:
            root_intro = (
                "This is the agent's persistent knowledge store. "
                "It is read and written by framework policies and "
                "aggregation strategies — there is no tool exposed to "
                "the model for direct access."
            )

        await store.write(
            "/SKILL.md",
            f"# {actor_name} Knowledge Store\n\n"
            f"{root_intro}\n\n"
            "## Directories\n"
            "- `/log/` -- Conversation history (auto-managed)\n"
            "- `/artifacts/` -- External files and data\n"
            "- `/memory/` -- Working memory and summaries (auto-managed)\n",
        )

        await store.write(
            f"{LOG_PREFIX}SKILL.md",
            "Conversation logs. Each file is a JSONL record of one conversation's events. "
            "Auto-populated by the framework after each conversation.",
        )

        await store.write(
            "/artifacts/SKILL.md",
            "External data: uploaded files, downloaded content, reference materials. "
            "Write here to store data you want to reference later.",
        )

        await store.write(
            "/memory/SKILL.md",
            "Working memory and conversation summaries. "
            "`working.md` contains your current persistent state. "
            "`conversations/` contains per-conversation summaries. "
            "Both are auto-updated by aggregation strategies.",
        )
