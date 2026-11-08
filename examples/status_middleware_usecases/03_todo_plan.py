# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Example: keep the current plan visible while tools update its progress.

Book: TODO list management. Build a plan, draft a report, and complete a step.
The plan belongs to conversation variables; status is only its current view.
"""

import asyncio
from datetime import UTC, datetime
from typing import Annotated, Literal

from ag2 import Agent, Context, DataInput, TextInput, Variable, tool
from ag2.events import Input, ToolCallEvent
from ag2.middleware import statusbar_middleware
from ag2.testing import TestConfig, TrackingConfig


@statusbar_middleware
def plan_status(todos: Annotated[list[dict[str, str]], Variable(default_factory=list)]) -> list[Input]:
    return [
        TextInput("Task progress; continue with the first unfinished item."),
        DataInput({"todos": [dict(item) for item in todos]}),
    ]


@tool
def rewrite_todo_list(items: list[str], context: Context) -> str:
    """Replace the plan with ordered items having stable IDs and timestamps."""
    now = datetime.now(UTC).isoformat()
    context.variables["todos"] = [
        {"id": f"step-{index}", "content": content, "status": "pending", "updated_at": now}
        for index, content in enumerate(items, start=1)
    ]
    return "Plan saved."


@tool
def update_todo_status(
    item_id: str,
    status: Literal["pending", "in_progress", "completed", "cancelled"],
    todos: Annotated[list[dict[str, str]], Variable()],
) -> str:
    """Change one existing plan item's status."""
    item = next(item for item in todos if item["id"] == item_id)
    item["status"] = status
    item["updated_at"] = datetime.now(UTC).isoformat()
    return f"{item_id}: {status}"


@tool
def draft_report(context: Context) -> str:
    """Produce an example report and retain it for a later review step."""
    context.variables["report"] = "Revenue increased by 8%; costs remained stable."
    return context.variables["report"]


async def main() -> None:
    config = TrackingConfig(
        TestConfig(
            ToolCallEvent("rewrite_todo_list", arguments='{"items": ["Draft the report", "Review the report"]}'),
            ToolCallEvent("update_todo_status", arguments='{"item_id": "step-1", "status": "in_progress"}'),
            ToolCallEvent("draft_report"),
            ToolCallEvent("update_todo_status", arguments='{"item_id": "step-1", "status": "completed"}'),
            "The draft is ready. The review is still pending.",
            "The remaining step is to review the report.",
        )
    )
    agent = Agent(
        "report_writer",
        config=config,
        prompt="Plan the work, execute it, and keep the checklist accurate. Do not claim pending work is complete.",
        tools=[rewrite_todo_list, update_todo_status, draft_report],
        middleware=[plan_status],
    )

    reply = await agent.ask("Draft and review a report. Pause after drafting so I can inspect it.")
    reply = await reply.ask("Which part of the plan remains?")

    snapshots = [call.messages[-1].parts[1].data["todos"] for call in config.calls]
    assert snapshots[0] == []
    assert snapshots[1][0]["status"] == "pending"
    assert snapshots[2][0]["status"] == "in_progress"
    assert snapshots[-1][0]["status"] == "completed"
    assert snapshots[-1][1]["status"] == "pending"
    assert reply.context.variables["report"].startswith("Revenue")
    print("Final plan:", snapshots[-1])
    print(reply.body)


if __name__ == "__main__":
    asyncio.run(main())
