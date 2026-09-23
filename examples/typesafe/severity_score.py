"""Grade incident reports on a rubric with Jev's score primitive.

An ``IntEnum`` numbered from 0 becomes a rubric, and the string under each member
describes that level. ``reply.content()`` is the member nearest Jev's expected
score; the expected value and the full distribution stay on ``reply.response.metadata``.
Needs ``TYPESAFE_API_KEY``.
"""

import asyncio
from enum import IntEnum

from ag2 import Agent
from ag2.config import TypeSafeConfig


class Severity(IntEnum):
    LOW = 0
    """Cosmetic; nothing is blocked."""
    MEDIUM = 1
    """Degraded but usable, or a workaround exists."""
    HIGH = 2
    """An outage, data loss, or a security exposure."""


REPORTS = [
    "The settings page logo is slightly blurry on retina screens.",
    "Exports take ten minutes instead of ten seconds, but they do finish.",
    "Customers can see each other's invoices after the last deploy.",
]


async def main() -> None:
    grader = Agent(
        "grader",
        prompt="How severe is this incident report?",
        config=TypeSafeConfig(),
        response_schema=Severity,
    )

    for report in REPORTS:
        reply = await grader.ask(report)
        severity = await reply.content()
        meta = reply.response.metadata
        print(f"{severity.name:<7} expected={meta['score']:.2f} {meta['probabilities']}  {report}")


if __name__ == "__main__":
    asyncio.run(main())
