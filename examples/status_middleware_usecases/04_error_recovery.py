# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Example: enrich a tool failure and remove the warning after recovery.

Book: detailed error information. A missing file leads to listing the directory
and then reading the correct file. No status is produced before a failure.
"""

import asyncio
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from ag2 import Agent, Context, DataInput, Inject, tool
from ag2.events import ToolCallEvent, ToolErrorEvent, ToolResultEvent, ToolResultsEvent
from ag2.middleware import statusbar_middleware
from ag2.testing import TestConfig, TrackingConfig


@statusbar_middleware
async def recovery_status(context: Context) -> DataInput | None:
    history = context.stream.history
    current_input = history.since_last_user_input()
    latest_read = await current_input.where(ToolResultEvent.name == "read_file").last()
    if not isinstance(latest_read, ToolErrorEvent):
        return None
    failed_call = await history.where(ToolCallEvent.id == latest_read.parent_id).first()
    assert failed_call is not None
    failed_reads = await current_input.where(ToolErrorEvent.name == "read_file").count()
    return DataInput({
        "error_type": type(latest_read.error).__name__,
        "message": str(latest_read.error),
        "arguments": dict(failed_call.serialized_arguments),
        "traceback": "".join(traceback.format_exception(latest_read.error)),
        "failed_read_attempts": failed_reads,
        "suggestion": "Inspect the directory and choose an existing file before another read.",
    })


@tool
def read_file(name: str, directory: Annotated[Path, Inject()]) -> str:
    """Read a file in the example directory."""
    return (directory / name).read_text()


@tool
def list_files(directory: Annotated[Path, Inject()]) -> list[str]:
    """List names to help recover from a mistaken path."""
    return sorted(path.name for path in directory.iterdir())


async def main() -> None:
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory / "report.txt").write_text("Quarterly report: revenue increased.")
        config = TrackingConfig(
            TestConfig(
                ToolCallEvent("read_file", arguments='{"name": "missing.txt"}'),
                ToolCallEvent("list_files"),
                ToolCallEvent("read_file", arguments='{"name": "report.txt"}'),
                "I found and read report.txt.",
                raise_tool_errors=False,
            )
        )
        agent = Agent(
            "reader",
            config=config,
            prompt="Read the report. If a path fails, investigate the directory before retrying.",
            dependencies={"directory": directory},
            tools=[read_file, list_files],
            middleware=[recovery_status],
        )

        reply = await agent.ask("Read missing.txt; locate the report if that filename is wrong.")

        warning = config.calls[1].messages[-1].parts[0].data
        assert warning["error_type"] == "FileNotFoundError"
        assert warning["arguments"] == {"name": "missing.txt"}
        assert "read_file" in warning["traceback"]
        assert warning["failed_read_attempts"] == 1
        # The final request ends in the real tool results, without a stale status.
        final_event = config.calls[-1].messages[-1]
        assert isinstance(final_event, ToolResultsEvent)
        print("Recovery advice seen by the model:", warning["suggestion"])
        print(reply.body)


if __name__ == "__main__":
    asyncio.run(main())
