# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Example: deliver external state as text, structured data, and an image.

An extension of the API examples, rather than a separate scenario from the book.
A local async double stands in for an external dashboard service.
"""

import asyncio
import base64
from dataclasses import dataclass
from typing import Annotated

from ag2 import Agent, DataInput, Depends, ImageInput, Inject, TextInput, Variable, tool
from ag2.events import BinaryInput, Input, ModelRequest, ToolCallEvent
from ag2.middleware import statusbar_middleware
from ag2.testing import TestConfig, TrackingConfig


@dataclass
class DashboardSnapshot:
    phase: str
    png: bytes


@dataclass
class DashboardClient:
    phase: str
    png: bytes

    async def fetch(self, project_id: str) -> DashboardSnapshot:
        # Replace this double with the application's async HTTP/SDK call.
        return DashboardSnapshot(phase=f"{project_id}: {self.phase}", png=self.png)


async def fetch_dashboard(
    dashboard: Annotated[DashboardClient, Inject()],
    project_id: Annotated[str, Variable()],
) -> DashboardSnapshot:
    return await dashboard.fetch(project_id)


@statusbar_middleware
def dashboard_status(snapshot: Annotated[DashboardSnapshot, Depends(fetch_dashboard)]) -> list[Input]:
    return [
        TextInput("Current project dashboard; use this snapshot for the next decision."),
        DataInput({"phase": snapshot.phase}),
        ImageInput(data=snapshot.png, media_type="image/png"),
    ]


@tool
def refresh_project(dashboard: Annotated[DashboardClient, Inject()]) -> str:
    """Simulate a project update in the external service."""
    dashboard.phase = "review"
    return "The project is now in review."


async def main() -> None:
    # One-pixel PNG fixture; real applications return a dashboard screenshot.
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGMwnpkGAAI1ATMXd8ShAAAAAElFTkSuQmCC"
    )
    dashboard = DashboardClient(phase="draft", png=png)
    config = TrackingConfig(TestConfig(ToolCallEvent("refresh_project"), "The project has reached review."))
    agent = Agent(
        "project_assistant",
        config=config,
        dependencies={"dashboard": dashboard},
        variables={"project_id": "launch"},
        tools=[refresh_project],
        middleware=[dashboard_status],
    )

    reply = await agent.ask("Refresh the project and summarize its current state.")

    requests = [call.messages[-1] for call in config.calls]
    assert all(isinstance(request, ModelRequest) for request in requests)
    assert all(len(request.parts) == 3 for request in requests)
    assert requests[0].parts[1].data == {"phase": "launch: draft"}
    assert requests[1].parts[1].data == {"phase": "launch: review"}
    assert isinstance(requests[1].parts[2], BinaryInput)
    assert requests[1].parts[2].data == png
    assert await reply.history.where(ModelRequest).count() == 1
    print("Dashboard states:", [request.parts[1].data for request in requests])
    print(reply.body)


if __name__ == "__main__":
    asyncio.run(main())
