"""03 · Travel planner — multi-turn conversation

Chained ``reply.ask()`` builds on the same conversation history. The planner
remembers constraints from earlier turns without the caller re-supplying
context each time — the stream stays alive across the whole dialogue.

Run::

    .venv-beta/bin/python playground/03_travel_planner.py
"""

from __future__ import annotations

import asyncio

from autogen.beta import Actor

from _config import default_config, section


TURNS = [
    "I want to plan a 5-day trip to Japan in late April. Just cherry-blossom season.",
    "Budget is around $2500 per person, two travellers. Optimise for sightseeing, not luxury.",
    "We prefer trains to flights once we're in Japan. Draft a day-by-day itinerary.",
    "Looks great. For day 3, swap the shopping stop for something outdoorsy in or near Kyoto.",
    "Summarize the final itinerary in a single bullet list, one line per day.",
]


async def main() -> None:
    config = default_config()

    agent = Actor(
        "travel-planner",
        prompt=(
            "You are a detail-oriented travel planner. When the user adds "
            "constraints, update the plan rather than starting over. Be "
            "concrete and concise."
        ),
        config=config,
    )

    section("Turn 1 — kick off")
    reply = await agent.ask(TURNS[0])
    print(reply.body)

    for i, question in enumerate(TURNS[1:], start=2):
        section(f"Turn {i} — {question}")
        reply = await reply.ask(question)
        print(reply.body)


if __name__ == "__main__":
    asyncio.run(main())
