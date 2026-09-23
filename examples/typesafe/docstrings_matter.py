"""The same choice with and without member docstrings.

Jev only knows what a label means from its description. The string under an
``Enum`` member is read back from the class source and sent as that option's
description; a bare member is sent as nothing but its name. The agents below share
the prompt and the model, so the descriptions are the only difference on the wire.
The labels are deliberately opaque to make that difference visible.
Needs ``TYPESAFE_API_KEY``.
"""

import asyncio
from enum import Enum

from ag2 import Agent
from ag2.config import TypeSafeConfig

PROMPT = "Which queue should this customer message go to?"

MESSAGES = [
    "I was charged twice this month and want one of them refunded.",
    "The API has returned 502 on every request since your maintenance window.",
    "Can you compare the Team and Enterprise plans before we sign?",
]


class Queue(Enum):
    ALPHA = "alpha"
    """Charges, invoices and refunds."""
    BRAVO = "bravo"
    """Bugs, outages and integrations."""
    CHARLIE = "charlie"
    """Pre-sales: pricing, demos and plan comparisons."""


class BareQueue(Enum):
    ALPHA = "alpha"
    BRAVO = "bravo"
    CHARLIE = "charlie"


# ``criteria`` sends the same descriptions for the bare enum without touching the type,
# which is the way in when the source is unavailable (a REPL, a notebook, a frozen app).
CRITERIA = {
    "alpha": "Charges, invoices and refunds.",
    "bravo": "Bugs, outages and integrations.",
    "charlie": "Pre-sales: pricing, demos and plan comparisons.",
}


def format_row(label: str, probabilities: dict[str, float]) -> str:
    cells = "  ".join(f"{name}={p:.2f}" for name, p in probabilities.items())
    return f"  {label:<12}{cells}"


async def main() -> None:
    agents = {
        "docstrings": Agent("documented", prompt=PROMPT, config=TypeSafeConfig(), response_schema=Queue),
        "bare": Agent("bare", prompt=PROMPT, config=TypeSafeConfig(), response_schema=BareQueue),
        "criteria": Agent(
            "described", prompt=PROMPT, config=TypeSafeConfig(criteria=CRITERIA), response_schema=BareQueue
        ),
    }

    for message in MESSAGES:
        print(message)
        for label, agent in agents.items():
            reply = await agent.ask(message)
            print(format_row(label, reply.response.metadata["probabilities"]))
        print()


if __name__ == "__main__":
    asyncio.run(main())
