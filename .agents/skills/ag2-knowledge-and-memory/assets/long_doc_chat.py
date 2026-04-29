"""Long-doc chat — composing assembly policies.

Mirrors website/docs/beta/code_examples/07_long_doc_chat.mdx. Three policies
compose in order:

1. ConversationPolicy — drops every event that isn't conversation/tool traffic.
2. SlidingWindowPolicy(max_events=6) — hard-caps events forwarded to the LLM.
3. TokenBudgetPolicy(max_tokens=2000) — char-based secondary cap.

Also pairs the assembly chain with TailWindowCompact so the agent's stream
history itself is kept small.

Run::

    python long_doc_chat.py
"""

import asyncio

from autogen.beta import Agent, KnowledgeConfig
from autogen.beta.compact import CompactTrigger, TailWindowCompact
from autogen.beta.config import GeminiConfig
from autogen.beta.events.lifecycle import CompactionCompleted
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.policies import (
    ConversationPolicy,
    SlidingWindowPolicy,
    TokenBudgetPolicy,
)
from autogen.beta.stream import MemoryStream


def section(title: str) -> None:
    print(f"\n── {title} ───")


QUESTIONS = [
    "Remember the word 'oak'.",
    "Remember the word 'river'.",
    "Remember the word 'lantern'.",
    "Remember the word 'sable'.",
    "Remember the word 'quartz'.",
    "Name the three most recent words I asked you to remember.",
]


async def main() -> None:
    config = GeminiConfig(model="gemini-3-flash-preview", temperature=0)

    store = MemoryKnowledgeStore()
    compactions: list[CompactionCompleted] = []
    stream = MemoryStream()
    stream.where(CompactionCompleted).subscribe(lambda e: compactions.append(e))

    agent = Agent(
        "lexicon",
        prompt=("Be very terse — one short sentence per reply. Answer directly without calling any tools."),
        config=config,
        assembly=[
            ConversationPolicy(),
            SlidingWindowPolicy(max_events=6, transparent=True),
            TokenBudgetPolicy(max_tokens=2000),
        ],
        knowledge=KnowledgeConfig(
            store=store,
            compact=TailWindowCompact(target=4),
            compact_trigger=CompactTrigger(max_events=8),
        ),
    )

    section("Long-doc chat — assembly policies trim what the LLM actually sees")

    reply = await agent.ask(QUESTIONS[0], stream=stream)
    print(f"Q1> {QUESTIONS[0]}")
    print(f"A1> {reply.body}")

    for i, q in enumerate(QUESTIONS[1:], start=2):
        reply = await reply.ask(q)
        print(f"Q{i}> {q}")
        print(f"A{i}> {reply.body}")

    print()
    print(f"Compactions fired during run: {len(compactions)}")
    for c in compactions:
        print(f"  - {c.strategy}: {c.events_before} → {c.events_after} events")


if __name__ == "__main__":
    asyncio.run(main())
