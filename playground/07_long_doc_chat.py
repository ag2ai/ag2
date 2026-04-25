"""07 · Long-doc chat — composing assembly policies

Shows the assembly chain in action. Three policies compose in order:

1. ``ConversationPolicy`` — drops every event that isn't conversation or
   tool traffic (no lifecycle noise reaches the LLM).
2. ``SlidingWindowPolicy(max_events=6)`` — hard-caps the number of events
   forwarded to the LLM, so history can't grow unbounded.
3. ``TokenBudgetPolicy(max_tokens=2000)`` — character-based secondary cap,
   belt-and-braces against one huge event blowing the budget.

Also pairs the assembly chain with ``TailWindowCompact`` so the actor's
stream history itself (not just the view into it) is kept small.

Run::

    .venv-beta/bin/python playground/07_long_doc_chat.py
"""

import asyncio

from _config import default_config, section

from autogen.beta import Actor, KnowledgeConfig
from autogen.beta.compact import CompactTrigger, TailWindowCompact
from autogen.beta.events.lifecycle import CompactionCompleted
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.policies import (
    ConversationPolicy,
    SlidingWindowPolicy,
    TokenBudgetPolicy,
)
from autogen.beta.stream import MemoryStream

QUESTIONS = [
    "Remember the word 'oak'.",
    "Remember the word 'river'.",
    "Remember the word 'lantern'.",
    "Remember the word 'sable'.",
    "Remember the word 'quartz'.",
    "Name the three most recent words I asked you to remember.",
]


async def main() -> None:
    config = default_config()

    store = MemoryKnowledgeStore()
    compactions: list[CompactionCompleted] = []
    stream = MemoryStream()
    stream.where(CompactionCompleted).subscribe(lambda e: compactions.append(e))

    agent = Actor(
        "lexicon",
        prompt="Be very terse — one short sentence per reply.",
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
