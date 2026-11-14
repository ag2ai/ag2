# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Example: accumulate conversation statistics and reset only turn statistics.

Extension of the book's tool counters for our user-turn requirement.
History selects the current input interval without a caller-supplied turn ID.
"""

import asyncio
from collections import Counter
from typing import Annotated, Any

from ag2 import Agent, Context, DataInput, Variable, tool
from ag2.events import ToolCallEvent
from ag2.middleware import statusbar_middleware
from ag2.testing import TestConfig, TrackingConfig


@statusbar_middleware
async def call_statistics(
    context: Context,
    statistics: Annotated[dict[str, Any], Variable(default_factory=dict)],
) -> DataInput:
    history = context.stream.history
    total = statistics.setdefault("conversation_counts", {})
    seen = set(statistics.get("seen_call_ids", []))
    # Keep explicit deduplication until a history cursor contract is designed.
    for event in await history.where(ToolCallEvent).all():
        if event.id not in seen:
            seen.add(event.id)
            total[event.name] = total.get(event.name, 0) + 1
    statistics["seen_call_ids"] = sorted(seen)

    current_calls = await history.since_last_user_input().where(ToolCallEvent).all()
    current = Counter(event.name for event in current_calls)
    return DataInput({
        "conversation_counts": dict(total),
        "turn_counts": dict(current),
    })


@tool
def search_documents(query: str) -> str:
    """Return a local example result without contacting a search service."""
    return f"Found notes matching: {query}"


async def main() -> None:
    config = TrackingConfig(
        TestConfig(
            ToolCallEvent("search_documents", arguments='{"query": "pricing"}'),
            ToolCallEvent("search_documents", arguments='{"query": "discounts"}'),
            "Pricing and discounts found.",
            ToolCallEvent("search_documents", arguments='{"query": "renewals"}'),
            "Renewal terms found.",
        )
    )
    agent = Agent(
        "researcher",
        config=config,
        tools=[search_documents],
        middleware=[call_statistics],
    )

    reply = await agent.ask("Find pricing and discount information.")
    reply = await reply.ask("Now find renewal terms.")

    snapshots = [call.messages[-1].parts[0].data for call in config.calls]
    totals = [snapshot["conversation_counts"].get("search_documents", 0) for snapshot in snapshots]
    turns = [snapshot["turn_counts"].get("search_documents", 0) for snapshot in snapshots]
    assert totals == [0, 1, 2, 2, 3]
    assert turns == [0, 1, 2, 0, 1]

    # A fresh conversation on the same Agent gets its own default accumulator.
    await agent.ask("Start another research task.")
    fresh = config.calls[5].messages[-1].parts[0].data
    assert fresh["conversation_counts"] == {}
    assert fresh["turn_counts"] == {}
    assert reply.context.variables["statistics"]["conversation_counts"] == {"search_documents": 3}
    print("Conversation totals:", totals)
    print("Per-input totals:", turns)


if __name__ == "__main__":
    asyncio.run(main())
