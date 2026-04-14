"""06 · Journal companion — knowledge store with working memory

Persistent actor memory using the framework's three knowledge primitives:

- ``KnowledgeStore`` — virtual filesystem for actor state.
- ``WorkingMemoryAggregate`` — an LLM-driven summary rollup that runs at
  the end of every conversation and writes ``/memory/working.md``.
- ``WorkingMemoryPolicy`` — an assembly policy that reads
  ``/memory/working.md`` and injects it as context at the start of every
  subsequent conversation.

The actor therefore "remembers" what you told it even after a full restart,
because the state lives in the knowledge store — not in conversation
history.

Run::

    .venv-beta/bin/python playground/06_journal_companion.py
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from _config import default_config, section

from autogen.beta import Actor, KnowledgeConfig
from autogen.beta.aggregate import AggregateTrigger, WorkingMemoryAggregate
from autogen.beta.knowledge import DiskKnowledgeStore
from autogen.beta.policies import ConversationPolicy, WorkingMemoryPolicy


async def main() -> None:
    config = default_config()

    # Use a fresh tempdir so the example is reproducible.
    workdir = Path(tempfile.mkdtemp(prefix="journal-companion-"))

    try:
        store = DiskKnowledgeStore(str(workdir))

        def build_agent() -> Actor:
            return Actor(
                "journal",
                prompt=(
                    "You are a supportive daily journal companion. Keep a "
                    "running understanding of what the user is working on. "
                    "Be brief and reference their past entries when relevant."
                ),
                config=config,
                knowledge=KnowledgeConfig(
                    store=store,
                    aggregate=WorkingMemoryAggregate(config=config),
                    # on_end=True: roll up working memory when each conversation finishes
                    aggregate_trigger=AggregateTrigger(on_end=True),
                ),
                assembly=[
                    WorkingMemoryPolicy(),  # inject /memory/working.md on every LLM call
                    ConversationPolicy(),  # then filter to conversation events
                ],
            )

        section("Session 1 — tell the journal what you're doing")

        agent1 = build_agent()
        r = await agent1.ask(
            "Today I started learning to build a home espresso setup. Still "
            "choosing between a Silvia Pro and a Linea Mini."
        )
        print(r.body)
        r = await r.ask(
            "Also started reading The Pragmatic Programmer. On chapter 2 "
            "about orthogonality. That's the whole update."
        )
        print(r.body)
        # When the `with await agent1.ask(...)` exits the `_execute`,
        # WorkingMemoryAggregate writes /memory/working.md.

        working = await store.read("/memory/working.md")
        print()
        print("## /memory/working.md after session 1")
        print(working)

        section("Session 2 — new Actor instance, same store: memory persists")

        agent2 = build_agent()
        r2 = await agent2.ask(
            "Quick check-in: what was I working on? Answer in one line."
        )
        print(r2.body)
        # The answer should mention espresso and/or Pragmatic Programmer
        # even though agent2 is a brand-new object with no prior state,
        # because WorkingMemoryPolicy injected /memory/working.md as a prompt.
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
