"""The three kinds of answer Jev gives, and what each one puts on the reply.

Jev has three question primitives, and ``response_schema`` picks one:

* ``bool`` (or a number schema bounded to ``0..1``) asks a yes/no question and is
  answered with a probability, a *noul*;
* an ``Enum`` of strings asks for a choice and is answered with a label, a
  confidence and a probability per label;
* an ``IntEnum`` numbered from 0 asks for a score on a rubric and is answered with
  the expected level, a confidence, the rubric legend and a probability per level.

``await reply.content()`` is the answer coerced to the schema; the answer itself is
kept whole on ``reply.response.metadata``. Needs ``TYPESAFE_API_KEY``.
"""

import asyncio
from enum import Enum, IntEnum

from ag2 import Agent, ResponseSchema
from ag2.config import TypeSafeConfig

MESSAGE = "Your last deploy broke our checkout. We're losing orders every minute this stays down."


class Team(Enum):
    """Which team should take this?"""

    BILLING = "billing"
    """Charges, invoices, refunds."""
    ENGINEERING = "engineering"
    """Bugs, outages, integrations."""
    SALES = "sales"
    """Pricing, plans, renewals."""


class Urgency(IntEnum):
    LOW = 0
    """Can wait for the next release."""
    MEDIUM = 1
    """Should be handled today."""
    HIGH = 2
    """Revenue or data is at risk right now."""


ESCALATE = "Should this be escalated to an on-call engineer?"

# The same yes/no question, answered with the probability itself instead of a bool.
ESCALATION_PROBABILITY = ResponseSchema.from_schema(
    {"type": "number", "minimum": 0, "maximum": 1},
    name="escalation",
    description=ESCALATE,
)


async def main() -> None:
    config = TypeSafeConfig()

    print("1. yes/no (noul)")
    yes_no = Agent("yes_no", config=config, response_schema=ResponseSchema(bool, description=ESCALATE))
    reply = await yes_no.ask(MESSAGE)
    # metadata: {'noul': 0.97}
    print(f"   content : {await reply.content()!r}")
    print(f"   metadata: {reply.response.metadata}")

    # A schema built with ``from_schema`` is not parsed by ``content()``, so read the
    # body, which is the probability as JSON text.
    probability = Agent("probability", config=config, response_schema=ESCALATION_PROBABILITY)
    reply = await probability.ask(MESSAGE)
    print(f"   as float: {float(reply.body or 'nan')!r}")

    print("2. choice")
    choice = Agent("choice", config=config, response_schema=Team)
    reply = await choice.ask(MESSAGE)
    # metadata: {'choice': 'engineering', 'confidence': 0.93, 'probabilities': {'billing': ..., ...}}
    print(f"   content : {await reply.content()!r}")
    print(f"   metadata: {reply.response.metadata}")

    print("3. score")
    score = Agent("score", prompt="How urgent is this message?", config=config, response_schema=Urgency)
    reply = await score.ask(MESSAGE)
    # metadata: {'score': 1.8, 'confidence': 0.85, 'legend': {0: ..., 1: ..., 2: ...}, 'probabilities': {0: ..., ...}}
    print(f"   content : {await reply.content()!r}")
    print(f"   metadata: {reply.response.metadata}")


if __name__ == "__main__":
    asyncio.run(main())
