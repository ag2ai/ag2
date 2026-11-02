# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Example: combine persistent progress and a dynamic working directory.

The model sees the progress trail plus only the current environment snapshot.
Both status functions run in registration order and share normal AG2 DI.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from ag2 import Agent, Context, DataInput, Inject, Variable, tool
from ag2.events import ModelRequest, ToolCallEvent
from ag2.middleware import statusbar_middleware
from ag2.testing import TestConfig, TrackingConfig


@dataclass
class Workspace:
    cwd: Path


@statusbar_middleware(persist=True)
def progress(phase: Annotated[str, Variable(default="draft")]) -> DataInput:
    return DataInput({"section": "progress", "value": phase}, metadata={"case": "mixed-persistence"})


@statusbar_middleware(persist=False)
def environment(workspace: Annotated[Workspace, Inject()]) -> DataInput:
    return DataInput({"section": "cwd", "value": str(workspace.cwd)}, metadata={"case": "mixed-persistence"})


@tool
def start_review(context: Context, workspace: Annotated[Workspace, Inject()]) -> str:
    """Move the task and its virtual workspace to the review stage."""
    context.variables["phase"] = "review"
    workspace.cwd /= "review"
    return "Review started."


async def main() -> None:
    config = TrackingConfig(
        TestConfig(ToolCallEvent("start_review"), "Review started.", "We are reviewing the project.")
    )
    workspace = Workspace(Path("/project"))
    agent = Agent(
        "reviewer",
        config=config,
        dependencies={"workspace": workspace},
        tools=[start_review],
        middleware=[progress, environment],
    )

    reply = await agent.ask("Start reviewing the draft.")
    reply = await reply.ask("Which stage and directory are current?")

    snapshots = [
        tuple(
            (part.data["section"], part.data["value"])
            for event in call.messages
            for part in getattr(event, "parts", ())
            if isinstance(part, DataInput) and part.metadata.get("case") == "mixed-persistence"
        )
        for call in config.calls
    ]
    assert snapshots == [
        (("progress", "draft"), ("cwd", "/project")),
        (("progress", "draft"), ("progress", "review"), ("cwd", "/project/review")),
        (("progress", "draft"), ("progress", "review"), ("progress", "review"), ("cwd", "/project/review")),
    ]
    assert await reply.history.where(ModelRequest).count() == 2
    assert await reply.history.since_last_user_input().where(ToolCallEvent).count() == 0
    print("Persistent trail and current environment:", snapshots)
    print(reply.body)


if __name__ == "__main__":
    asyncio.run(main())
