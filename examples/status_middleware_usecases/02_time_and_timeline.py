# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Example: report current time and the timing of the latest user/tool events.

Book: timestamp tracking. A tool takes time; the next input arrives tomorrow.
Event metadata carries timestamps without rewriting old conversation text.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated

from ag2 import Agent, Context, DataInput, Inject, TextInput, ToolResult, tool
from ag2.events import ModelRequest, ToolCallEvent, ToolResultEvent
from ag2.middleware import statusbar_middleware
from ag2.testing import TestConfig, TrackingConfig


@dataclass
class Clock:
    current: datetime


@statusbar_middleware
async def timeline(context: Context, clock: Annotated[Clock, Inject()]) -> DataInput:
    history = context.stream.history
    latest_input = await history.where(ModelRequest.origin == "user").last()
    assert latest_input is not None
    sent_at = next(part.metadata["sent_at"] for part in latest_input.parts if "sent_at" in part.metadata)
    latest_result = await history.where(ToolResultEvent.name == "find_recent_files").last()
    return DataInput({
        "now": clock.current.isoformat(),
        "latest_user_input_at": sent_at,
        "seconds_since_user_input": (clock.current - datetime.fromisoformat(sent_at)).total_seconds(),
        "latest_tool_result_at": latest_result.result.metadata.get("observed_at") if latest_result else None,
    })


@tool
def find_recent_files(clock: Annotated[Clock, Inject()]) -> ToolResult:
    """Simulate a five-minute search using the same clock as the status function."""
    clock.current += timedelta(minutes=5)
    return ToolResult("Found report.txt", metadata={"observed_at": clock.current.isoformat()})


async def main() -> None:
    clock = Clock(datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc))
    config = TrackingConfig(TestConfig(ToolCallEvent("find_recent_files"), "Found the report.", "It is a new day."))
    agent = Agent(
        "file_assistant",
        config=config,
        prompt="Use the supplied event times to interpret relative dates and explain elapsed time.",
        dependencies={"clock": clock},
        tools=[find_recent_files],
        middleware=[timeline],
    )

    reply = await agent.ask(TextInput("Find yesterday's report.", metadata={"sent_at": clock.current.isoformat()}))
    clock.current += timedelta(days=1)
    reply = await reply.ask(TextInput("What does today mean now?", metadata={"sent_at": clock.current.isoformat()}))

    snapshots = [call.messages[-1].parts[0].data for call in config.calls]
    assert [snapshot["seconds_since_user_input"] for snapshot in snapshots] == [0, 300, 0]
    assert snapshots[0]["now"].startswith("2026-10-20")
    assert snapshots[-1]["now"].startswith("2026-10-21")
    assert snapshots[-1]["latest_tool_result_at"] == "2026-10-20T10:05:00+00:00"
    assert await reply.history.where(ModelRequest).count() == 2
    print("Timeline snapshots:", snapshots)


if __name__ == "__main__":
    asyncio.run(main())
