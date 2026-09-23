"""Route support tickets to a team with TypeSafe's Jev decision model.

Jev answers the question the ``response_schema`` poses instead of generating text.
An ``Enum`` becomes a choice between its members: the class docstring is the
question and the string under each member describes that option.
Needs ``TYPESAFE_API_KEY``.
"""

import asyncio
from enum import Enum

from ag2 import Agent
from ag2.config import TypeSafeConfig


class Department(Enum):
    """Which team should handle this ticket?"""

    BILLING = "billing"
    """Payments, invoicing, refunds and subscription changes."""
    TECHNICAL = "technical"
    """Bugs, outages, API and integration problems."""
    ACCOUNT = "account"
    """Login, password, permissions and profile changes."""


TICKETS = [
    "I was charged twice for Pro last week and still can't export my reports.",
    "Our webhook integration has been returning 502s since Monday.",
    "I can't log in after changing my email address.",
]


async def main() -> None:
    router = Agent(
        "router",
        prompt="You triage customer support tickets.",
        config=TypeSafeConfig(),
        response_schema=Department,
    )

    for ticket in TICKETS:
        reply = await router.ask(ticket)
        department = await reply.content()
        confidence = reply.response.metadata["confidence"]
        print(f"{department.value:<10} {confidence:.2f}  {ticket}")


if __name__ == "__main__":
    asyncio.run(main())
