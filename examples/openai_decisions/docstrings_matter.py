"""The same choice with and without member docstrings.

The model only knows what a label means from its description. The string under an
``Enum`` member is read back from the class source and sent as that choice's
``description``; a bare member is sent as nothing but its value. The agents below
share the prompt and the model, so the descriptions are the only difference on the
wire. The labels are deliberately opaque to make that difference visible.
Needs ``OPENAI_API_KEY``.
"""

import asyncio
from enum import Enum
from typing import Any

from ag2 import Agent
from ag2.config import OpenAIDecisionsConfig

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


# ``descriptions`` sends the same text for the bare enum without touching the type,
# which is the way in when the source is unavailable (a REPL, a notebook, a frozen app).
DESCRIPTIONS = {
    "alpha": "Charges, invoices and refunds.",
    "bravo": "Bugs, outages and integrations.",
    "charlie": "Pre-sales: pricing, demos and plan comparisons.",
}


def format_row(label: str, probabilities: list[dict[str, Any]]) -> str:
    cells = "  ".join(f"{p['value']}={p['probability']:.2f}" for p in probabilities)
    return f"  {label:<14}{cells}"


async def main() -> None:
    config = OpenAIDecisionsConfig()
    agents = {
        "docstrings": Agent("documented", prompt=PROMPT, config=config, response_schema=Queue),
        "bare": Agent("bare", prompt=PROMPT, config=config, response_schema=BareQueue),
        "descriptions": Agent(
            "described",
            prompt=PROMPT,
            config=OpenAIDecisionsConfig(
                descriptions=DESCRIPTIONS,
            ),
            response_schema=BareQueue,
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
