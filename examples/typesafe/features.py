"""ML feature extraction: a downstream classical ML model needs semantic signals.

Purchase intent, product interest, competitive pressure, churn signals. Each signal
is a yes/no question, and the probability Jev returns is the feature value, so a row
of features is one parallel batch of asks per record. Needs ``TYPESAFE_API_KEY``.
"""

import asyncio

from ag2 import Agent, ResponseSchema
from ag2.config import TypeSafeConfig

SIGNALS = {
    "purchase_intent": "The customer is close to buying or upgrading.",
    "churn_risk": "The customer is considering leaving.",
    "competitor": "The customer refers to a competing product.",
}

NOTES = [
    "Asked for a quote for 50 seats on the Enterprise plan, wants to sign before Q4.",
    "Said Acme's tool does the same for half the price and they're evaluating it.",
    "Renewed for another year, no open issues.",
]


async def main() -> None:
    config = TypeSafeConfig()
    detectors = [
        Agent(name, config=config, response_schema=ResponseSchema(bool, description=question))
        for name, question in SIGNALS.items()
    ]

    print(f"{'note':<50}" + "".join(f"{name:>17}" for name in SIGNALS))
    for note in NOTES:
        replies = await asyncio.gather(*(detector.ask(note) for detector in detectors))
        features = [reply.response.metadata["noul"] for reply in replies]
        print(f"{note[:47] + '...' if len(note) > 50 else note:<50}" + "".join(f"{f:>17.2f}" for f in features))


if __name__ == "__main__":
    asyncio.run(main())
