"""Classification: one known category should win.

Intent, topic, department, risk type, entity type. An ``Enum`` of strings asks Jev
for a choice; the class docstring is the question and the string under each member
tells Jev what that label means. Needs ``TYPESAFE_API_KEY``.
"""

import asyncio
from enum import Enum

from ag2 import Agent
from ag2.config import TypeSafeConfig


class Intent(Enum):
    """What does the customer want?"""

    REFUND = "refund"
    """Money back for a charge or a subscription."""
    CANCEL = "cancel"
    """Stop a subscription or close the account."""
    HOW_TO = "how_to"
    """Help using a feature that works as designed."""
    BUG = "bug"
    """Something that used to work no longer does."""


MESSAGES = [
    "How do I export my invoices as CSV?",
    "Charged twice this month. I want the second one back.",
    "Exports worked yesterday, now every one fails with a 500.",
    "Please close my account, we've moved to another tool.",
]


async def main() -> None:
    classifier = Agent("intent", config=TypeSafeConfig(), response_schema=Intent)

    for message in MESSAGES:
        reply = await classifier.ask(message)
        intent = await reply.content()
        print(f"{intent.value:<8} {reply.response.metadata['confidence']:.2f}  {message}")


if __name__ == "__main__":
    asyncio.run(main())
