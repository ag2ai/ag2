"""Scoring: an ``IntEnum`` is a rubric the Decisions API scores against.

Members are levels ``0..n-1`` from lowest to highest; their names become level
labels and the docstrings under them describe each level. The raw,
probability-weighted score is kept on the reply's metadata. Needs ``OPENAI_API_KEY``.
"""

import asyncio
from enum import IntEnum

from ag2 import Agent, ResponseSchema
from ag2.config import OpenAIDecisionsConfig


class Severity(IntEnum):
    COSMETIC = 0
    """Appearance only; no lost functionality."""
    WORKAROUND_AVAILABLE = 1
    """A task fails, but another way works."""
    FULLY_BLOCKED = 2
    """A task fails with no workaround."""


ISSUES = [
    "The settings icon is two pixels off-centre.",
    "Export fails in Safari but works in Chrome.",
    "Nobody can log in since the last deploy.",
]


async def main() -> None:
    grader = Agent(
        "grader",
        config=OpenAIDecisionsConfig(),
        response_schema=ResponseSchema(Severity, description="How severe is this issue?"),
    )

    for issue in ISSUES:
        reply = await grader.ask(issue)
        severity = await reply.content()
        print(f"{severity.name:<22} score={reply.response.metadata['score']:.2f}  {issue}")


if __name__ == "__main__":
    asyncio.run(main())
