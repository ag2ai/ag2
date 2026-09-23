"""Scoring: the answer belongs on an ordered rubric.

Severity, relevance, quality, frustration, suitability. An ``IntEnum`` numbered from
0 asks Jev for a score; the string under each member describes that level. Jev
answers with the expected level, so ``content()`` snaps it to the nearest member and
the metadata keeps the full distribution. Needs ``TYPESAFE_API_KEY``.
"""

import asyncio
from enum import IntEnum

from ag2 import Agent
from ag2.config import TypeSafeConfig


class Frustration(IntEnum):
    CALM = 0
    """Calm, just stating facts."""
    FRUSTRATED = 1
    """Frustrated but civil."""
    ANGRY = 2
    """Very angry, strong language."""


MESSAGES = [
    "Could you let me know when the invoice PDF will be available?",
    "Third time I'm writing about this. The export is still broken.",
    "This is a joke. Your product has cost me a week and nobody answers.",
]


async def main() -> None:
    grader = Agent(
        "frustration",
        prompt="How frustrated does the customer appear?",
        config=TypeSafeConfig(),
        response_schema=Frustration,
    )

    for message in MESSAGES:
        reply = await grader.ask(message)
        level = await reply.content()
        print(f"{level.name:<11} expected={reply.response.metadata['score']:.2f}  {message}")


if __name__ == "__main__":
    asyncio.run(main())
