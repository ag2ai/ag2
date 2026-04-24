# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, runtime_checkable

from .base import KnowledgeStore


@runtime_checkable
class StoreBootstrap(Protocol):
    """Initializes a knowledge store with a starting structure.

    Called once when an actor first runs with a store. Subsequent
    runs skip bootstrapping (detected via a sentinel file).
    """

    async def bootstrap(self, store: KnowledgeStore, actor_name: str) -> None:
        """Create initial store structure."""
        ...


class DefaultBootstrap:
    """Creates the standard knowledge store layout with SKILL.md files."""

    async def bootstrap(self, store: KnowledgeStore, actor_name: str) -> None:
        await store.write(
            "/SKILL.md",
            f"# {actor_name} Knowledge Store\n\n"
            "This is your persistent knowledge store. Use the `knowledge` tool to manage it.\n\n"
            "## Directories\n"
            "- `/log/` -- Conversation history (auto-managed)\n"
            "- `/artifacts/` -- External files and data\n"
            "- `/memory/` -- Working memory and summaries (auto-managed)\n",
        )

        await store.write(
            "/log/SKILL.md",
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
