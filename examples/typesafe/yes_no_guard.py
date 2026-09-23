"""Gate a generative agent behind a yes/no decision from Jev.

A ``bool`` schema becomes a yes/no question. ``criteria`` describes both outcomes
and ``boolean_threshold`` sets how sure Jev must be before the answer is ``True``.
The generative agent only sees the message once the gate has decided how to treat it.
Needs ``TYPESAFE_API_KEY`` and ``ANTHROPIC_API_KEY``.
"""

import asyncio

from ag2 import Agent, ResponseSchema
from ag2.config import AnthropicConfig, TypeSafeConfig

refund_check = Agent(
    "refund_check",
    config=TypeSafeConfig(
        boolean_threshold=0.8,
        criteria={
            "true": "The customer wants money back for something they already paid for.",
            "false": "Anything else, including questions about pricing or upgrades.",
        },
    ),
    response_schema=ResponseSchema(bool, description="Is the customer asking for a refund?"),
)

support = Agent(
    "support",
    prompt="You answer customer emails in two short sentences.",
    config=AnthropicConfig(model="claude-haiku-4-5"),
)


async def main() -> None:
    message = "You billed me for a year of Pro but I cancelled in March. I want that money back."

    verdict = await refund_check.ask(message)
    is_refund = await verdict.content()
    print(f"refund request: {is_refund} (p={verdict.response.metadata['noul']:.2f})")

    if is_refund:
        reply = await support.ask(
            f"Acknowledge the refund request and say finance will reply within two business days:\n\n{message}"
        )
    else:
        reply = await support.ask(message)
    print(reply.body)


if __name__ == "__main__":
    asyncio.run(main())
