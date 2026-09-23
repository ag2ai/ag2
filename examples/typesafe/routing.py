"""Routing: a category selects the next code path.

Tool use, escalation, model routing, support queues. The choice Jev makes picks
which function handles the message, and its confidence decides whether that pick
is trusted at all. Needs ``TYPESAFE_API_KEY``.
"""

import asyncio
from collections.abc import Callable
from enum import Enum

from ag2 import Agent
from ag2.config import TypeSafeConfig


class Queue(Enum):
    """Which queue should handle this message?"""

    BILLING = "billing"
    """Charges, invoices, refunds and subscription changes."""
    TECHNICAL = "technical"
    """Bugs, outages, API and integration problems."""
    SALES = "sales"
    """Pricing, plans, demos and renewals."""


def open_billing_ticket(message: str) -> str:
    return f"opened a billing ticket for {message!r}"


def page_on_call(message: str) -> str:
    return f"paged the on-call engineer with {message!r}"


def forward_to_sales(message: str) -> str:
    return f"forwarded {message!r} to the sales inbox"


def hold_for_a_human(message: str) -> str:
    return f"held {message!r} for manual triage"


HANDLERS: dict[Queue, Callable[[str], str]] = {
    Queue.BILLING: open_billing_ticket,
    Queue.TECHNICAL: page_on_call,
    Queue.SALES: forward_to_sales,
}

MESSAGES = [
    "Our webhook integration has been returning 502s since Monday.",
    "Can I get a quote for 40 more seats on the annual plan?",
    "The invoice says Pro but we downgraded to Team last month.",
    "Hello?",
]


async def main() -> None:
    router = Agent("router", config=TypeSafeConfig(), response_schema=Queue)

    for message in MESSAGES:
        decision = await router.ask(message)
        queue = await decision.content()
        confidence = decision.response.metadata["confidence"]

        # A low-confidence pick is not a pick: send it to a person instead.
        handler = HANDLERS[queue] if confidence >= 0.6 else hold_for_a_human
        print(f"{queue.value:<10} {confidence:.2f}  {handler(message)}")


if __name__ == "__main__":
    asyncio.run(main())
