# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Example: retain every progress snapshot across tools and user inputs.

persist=True appends a new record on each logical model call, even when the
value repeats. Service records must not become user-input boundaries.
"""

import asyncio
from typing import Annotated

from ag2 import Agent, Context, DataInput, Variable, tool
from ag2.events import ModelRequest, ToolCallEvent
from ag2.middleware import statusbar_middleware
from ag2.testing import ModelCall, TestConfig, TrackingConfig


def progress_snapshots(calls: list[ModelCall]) -> list[tuple[int, ...]]:
    return [
        tuple(
            part.data["completed"]
            for event in call.messages
            for part in getattr(event, "parts", ())
            if isinstance(part, DataInput) and part.metadata.get("case") == "persistent-progress"
        )
        for call in calls
    ]


@statusbar_middleware(persist=True)
def progress(completed: Annotated[int, Variable(default=0)]) -> DataInput:
    return DataInput({"completed": completed}, metadata={"case": "persistent-progress"})


@tool
def complete_step(context: Context) -> str:
    """Complete one step in this conversation's task."""
    context.variables["completed"] += 1
    return f"Completed {context.variables['completed']} steps."


async def main() -> None:
    config = TrackingConfig(
        TestConfig(
            ToolCallEvent("complete_step"),
            ToolCallEvent("complete_step"),
            "Two steps completed.",
            "Progress went from zero to one to two completed steps.",
        )
    )
    agent = Agent(
        "worker",
        config=config,
        tools=[complete_step],
        middleware=[progress],
    )

    reply = await agent.ask("Complete two steps and keep the progress trail.")
    reply = await reply.ask("How did progress change during the task?")

    snapshots = progress_snapshots(config.calls)
    assert snapshots == [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 2)]
    assert await reply.history.where(ModelRequest).count() == 2
    assert await reply.history.where(ToolCallEvent).count() == 2
    assert await reply.history.since_last_user_input().where(ToolCallEvent).count() == 0
    assert reply.context.variables["completed"] == 2
    previous_calls = len(config.calls)

    # Reusing the Agent for another conversation does not reuse its progress log.
    await agent.ask("Start a separate task and complete two steps.")
    assert progress_snapshots(config.calls[previous_calls:]) == [(0,), (0, 1), (0, 1, 2)]
    assert progress_snapshots(config.calls[:previous_calls]) == snapshots
    assert reply.context.variables["completed"] == 2
    print("Persistent progress in successive model inputs:", snapshots)
    print(reply.body)


if __name__ == "__main__":
    asyncio.run(main())
