"""Journal companion — knowledge store with working memory.

Mirrors website/docs/beta/code_examples/06_journal_companion.mdx. Persistent
agent memory using three primitives:

- KnowledgeStore — virtual filesystem for agent state.
- WorkingMemoryAggregate — LLM-driven summary rollup that runs at the end of
  every conversation and writes /memory/working.md.
- WorkingMemoryPolicy — assembly policy that reads /memory/working.md and
  injects it as context at the start of every subsequent conversation.

The agent therefore "remembers" what you told it even after a full restart,
because the state lives in the knowledge store — not in conversation history.

Run::

    python journal_companion.py
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

from autogen.beta import Agent, KnowledgeConfig
from autogen.beta.aggregate import AggregateTrigger, WorkingMemoryAggregate
from autogen.beta.config import GeminiConfig
from autogen.beta.knowledge import DiskKnowledgeStore
from autogen.beta.policies import ConversationPolicy, WorkingMemoryPolicy


def section(title: str) -> None:
    print(f"\n── {title} ───")


async def main() -> None:
    config = GeminiConfig(model="gemini-3-flash-preview", temperature=0)
    workdir = Path(tempfile.mkdtemp(prefix="journal-companion-"))

    try:
        store = DiskKnowledgeStore(str(workdir))

        def build_agent() -> Agent:
            return Agent(
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
                    aggregate_trigger=AggregateTrigger(on_end=True),
                ),
                assembly=[
                    WorkingMemoryPolicy(),
                    ConversationPolicy(),
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
            "Also started reading The Pragmatic Programmer. On chapter 2 about orthogonality. That's the whole update."
        )
        print(r.body)

        working = await store.read("/memory/working.md")
        print()
        print("## /memory/working.md after session 1")
        print(working)

        section("Session 2 — new Agent instance, same store: memory persists")

        agent2 = build_agent()
        r2 = await agent2.ask("Quick check-in: what was I working on? Answer in one line.")
        print(r2.body)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
