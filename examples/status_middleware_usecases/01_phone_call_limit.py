# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Example: show the per-merchant call count before every model decision.

Book: experiment 2-8. Three calls, followed by another user request.
The scripted model does not make a fourth call; this is a delivery check,
not an evaluation of whether a real model obeys the limit.
"""

import asyncio
from collections import Counter

from ag2 import Agent, Context, DataInput, tool
from ag2.events import ModelRequest, ToolCallEvent
from ag2.middleware import statusbar_middleware
from ag2.testing import TestConfig, TrackingConfig


@statusbar_middleware
async def merchant_calls(context: Context) -> DataInput:
    # Select scalar call events; batch and response wrappers are not expanded.
    calls = await context.stream.history.where(ToolCallEvent.name == "phone_call").all()
    counts = Counter(call.serialized_arguments["merchant"] for call in calls)
    return DataInput({"calls": dict(counts), "per_merchant_limit": 3}, metadata={"case": "merchant-limit"})


@tool
def phone_call(merchant: str) -> str:
    """Simulate a negotiation call; no telephone service is contacted."""
    return f"Negotiation with {merchant} recorded. A lower price is available."


async def main() -> None:
    config = TrackingConfig(
        TestConfig(
            ToolCallEvent("phone_call", arguments='{"merchant": "Xfinity"}'),
            ToolCallEvent("phone_call", arguments='{"merchant": "Xfinity"}'),
            ToolCallEvent("phone_call", arguments='{"merchant": "Xfinity"}'),
            "Three attempts completed; waiting for your decision.",
            "The call allowance is exhausted. We can follow up in writing.",
        )
    )
    agent = Agent(
        "negotiator",
        config=config,
        prompt="Negotiate a better price. Make at most three calls to each merchant across this conversation.",
        tools=[phone_call],
        middleware=[merchant_calls],
    )

    reply = await agent.ask("Please negotiate a lower Xfinity subscription price.")
    reply = await reply.ask("Please contact Xfinity by phone once more.")

    counts = []
    for call in config.calls:
        status = call.messages[-1]
        assert isinstance(status, ModelRequest)
        assert len(status.parts) == 1
        counts.append(status.parts[0].data["calls"].get("Xfinity", 0))
    assert counts == [0, 1, 2, 3, 3]
    assert await reply.history.where(ModelRequest).count() == 2
    assert await reply.history.where(ToolCallEvent).count() == 3
    assert all(call.prompt == config.calls[0].prompt for call in config.calls)
    # Inspect every model input: each contains exactly one current observation.
    for call in config.calls:
        parts = [part for event in call.messages for part in getattr(event, "parts", ())]
        assert sum(part.metadata.get("case") == "merchant-limit" for part in parts) == 1
    retained_parts = [part for event in await reply.history.get_events() for part in getattr(event, "parts", ())]
    assert not any(part.metadata.get("case") == "merchant-limit" for part in retained_parts)
    print("Merchant call counts seen by the model:", counts)
    print(reply.body)


if __name__ == "__main__":
    asyncio.run(main())
