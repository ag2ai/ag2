"""Verification: an artifact must be checked for specific failure modes.

Citation support, policy violations, tool-call errors, response quality. Each
failure mode is one yes/no question over the artifact and whatever it is judged
against, passed together as structured state. Needs ``TYPESAFE_API_KEY``.
"""

import asyncio

from ag2 import Agent, ResponseSchema
from ag2.config import TypeSafeConfig
from ag2.events import DataInput

SOURCE = (
    "Refunds are available within 14 days of purchase for annual plans. "
    "Monthly plans are non-refundable but can be cancelled at any time."
)

DRAFTS = [
    "Annual plans can be refunded within 14 days; monthly plans can't be refunded but you can cancel anytime.",
    "All plans are refundable within 30 days, no questions asked.",
]


async def main() -> None:
    config = TypeSafeConfig(boolean_threshold=0.8)
    checks = {
        "supported": Agent(
            "supported",
            config=config,
            response_schema=ResponseSchema(bool, description="Every claim in the draft is supported by the source."),
        ),
        "overpromises": Agent(
            "overpromises",
            config=config,
            response_schema=ResponseSchema(bool, description="The draft promises something the source does not offer."),
        ),
    }

    for draft in DRAFTS:
        print(draft)
        for name, check in checks.items():
            reply = await check.ask(DataInput({"source": SOURCE, "draft": draft}))
            print(f"  {name:<13} {str(await reply.content()):<6} p={reply.response.metadata['noul']:.2f}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
