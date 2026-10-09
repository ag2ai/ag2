"""Routing: a choice question selects the next code path.

The department the Decisions API picks chooses which function handles the
complaint, and its confidence decides whether that pick is trusted at all.
Needs ``OPENAI_API_KEY``.
"""

import asyncio
from collections.abc import Callable
from enum import Enum

from ag2 import Agent
from ag2.config import OpenAIDecisionsConfig


class Department(Enum):
    """Which department should handle this complaint?"""

    BILLING = "billing"
    """Payments, invoices, and refunds."""
    TECHNICAL = "technical"
    """Problems using the product."""
    SHIPPING = "shipping"
    """Delivery and tracking."""
    OTHER = "other"
    """Requests outside these categories."""


def open_billing_ticket(message: str) -> str:
    return f"opened a billing ticket for {message!r}"


def page_on_call(message: str) -> str:
    return f"paged the on-call engineer with {message!r}"


def track_parcel(message: str) -> str:
    return f"asked the carrier about {message!r}"


def hold_for_a_human(message: str) -> str:
    return f"held {message!r} for manual triage"


HANDLERS: dict[Department, Callable[[str], str]] = {
    Department.BILLING: open_billing_ticket,
    Department.TECHNICAL: page_on_call,
    Department.SHIPPING: track_parcel,
}

MESSAGES = [
    "I was charged twice for my order.",
    "Export fails in Safari but works in Chrome.",
    "My parcel has said 'out for delivery' for four days.",
    "Hello?",
]


async def main() -> None:
    router = Agent("router", config=OpenAIDecisionsConfig(), response_schema=Department)

    for message in MESSAGES:
        decision = await router.ask(message)
        department = await decision.content()
        confidence = decision.response.metadata["confidence"]

        # A low-confidence pick is not a pick: send it to a person instead.
        handler = HANDLERS.get(department, hold_for_a_human) if confidence >= 0.6 else hold_for_a_human
        print(f"{department.value:<10} {confidence:.2f}  {handler(message)}")


if __name__ == "__main__":
    asyncio.run(main())
