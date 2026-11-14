# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Example: retain a progress update only when application state changes.

An empty sequence skips the initial unknown state; None skips unchanged state.
Skipping a new insertion leaves earlier persistent snapshots in context.
"""

import asyncio
from typing import Annotated

from ag2 import Agent, Context, DataInput, Variable, tool
from ag2.events import Input, ModelRequest, ToolCallEvent
from ag2.middleware import statusbar_middleware
from ag2.testing import TestConfig, TrackingConfig


@statusbar_middleware(persist=True)
def changed_progress(
    phase: Annotated[str, Variable(default="")],
    emitted: Annotated[dict[str, str], Variable(default_factory=dict)],
) -> list[Input] | None:
    if not phase:
        return []
    if emitted.get("phase") == phase:
        return None
    emitted["phase"] = phase
    return [DataInput({"phase": phase}, metadata={"case": "conditional-progress"})]


@tool
def poll_job(context: Context) -> str:
    """Simulate repeated polling: queued, queued again, then running."""
    polls = context.variables.get("polls", 0) + 1
    context.variables["polls"] = polls
    context.variables["phase"] = "queued" if polls < 3 else "running"
    return context.variables["phase"]


async def main() -> None:
    config = TrackingConfig(
        TestConfig(
            ToolCallEvent("poll_job"),
            ToolCallEvent("poll_job"),
            ToolCallEvent("poll_job"),
            "The job is running.",
            "The latest known state is still running.",
        )
    )
    agent = Agent(
        "job_assistant",
        config=config,
        tools=[poll_job],
        middleware=[changed_progress],
    )

    reply = await agent.ask("Poll the job until it starts running.")
    reply = await reply.ask("Summarize the known status without polling again.")

    snapshots = [
        tuple(
            part.data["phase"]
            for event in call.messages
            for part in getattr(event, "parts", ())
            if isinstance(part, DataInput) and part.metadata.get("case") == "conditional-progress"
        )
        for call in config.calls
    ]
    assert snapshots == [(), ("queued",), ("queued",), ("queued", "running"), ("queued", "running")]
    assert await reply.history.where(ModelRequest).count() == 2
    assert await reply.history.where(ToolCallEvent).count() == 3
    assert reply.context.variables["emitted"] == {"phase": "running"}
    print("Progress snapshots, with unchanged updates skipped:", snapshots)
    print(reply.body)


if __name__ == "__main__":
    asyncio.run(main())
