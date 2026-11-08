# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Example: keep the model aware of its conversation's working directory.

Book: system state awareness. Change a tool's working directory, then read a file.
The workspace is an injected session dependency, not process-wide os.chdir().
"""

import asyncio
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from ag2 import Agent, DataInput, Inject, tool
from ag2.events import ToolCallEvent
from ag2.middleware import statusbar_middleware
from ag2.testing import TestConfig, TrackingConfig


@dataclass
class Workspace:
    root: Path
    cwd: Path

    def list_files(self) -> list[str]:
        return sorted(path.name for path in self.cwd.iterdir())


@statusbar_middleware
async def environment(workspace: Annotated[Workspace, Inject()]) -> DataInput:
    return DataInput({
        "cwd": str(workspace.cwd),
        "operating_system": platform.system(),
        "shell": os.environ.get("SHELL"),
        "python": platform.python_version(),
        "entries": await asyncio.to_thread(workspace.list_files),
    })


@tool
def change_directory(path: str, workspace: Annotated[Workspace, Inject()]) -> str:
    """Change this conversation's working directory within its workspace."""
    destination = (workspace.cwd / path).resolve()
    destination.relative_to(workspace.root)
    if not destination.is_dir():
        raise NotADirectoryError(destination)
    workspace.cwd = destination
    return str(workspace.cwd)


@tool
def read_file(name: str, workspace: Annotated[Workspace, Inject()]) -> str:
    """Read relative to the same working directory the status function reports."""
    return (workspace.cwd / name).read_text()


async def main() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        (root / "reports").mkdir()
        (root / "reports" / "daily.txt").write_text("Daily report ready.")
        workspace = Workspace(root=root, cwd=root)
        config = TrackingConfig(
            TestConfig(
                ToolCallEvent("change_directory", arguments='{"path": "reports"}'),
                ToolCallEvent("read_file", arguments='{"name": "daily.txt"}'),
                "Daily report ready.",
            )
        )
        agent = Agent(
            "workspace_assistant",
            config=config,
            prompt="Use the current workspace state when choosing paths.",
            dependencies={"workspace": workspace},
            tools=[change_directory, read_file],
            middleware=[environment],
        )

        process_cwd = Path.cwd()
        reply = await agent.ask("Go to reports and read daily.txt.")

        snapshots = [call.messages[-1].parts[0].data for call in config.calls]
        assert snapshots[0]["cwd"] == str(root)
        assert snapshots[1]["cwd"] == str(root / "reports")
        assert snapshots[1]["entries"] == ["daily.txt"]
        assert Path.cwd() == process_cwd
        print("Directories seen by the model:", [snapshot["cwd"] for snapshot in snapshots])
        print(reply.body)


if __name__ == "__main__":
    asyncio.run(main())
