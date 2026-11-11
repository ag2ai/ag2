# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Which database a delegated worker writes to when the coordinator and the worker both have one."""

from typing import Annotated

import pytest

from ag2 import Agent, Inject, tool
from ag2.events import ToolCallEvent
from ag2.testing import TestConfig


class Database:
    def __init__(self) -> None:
        self.rows: list[str] = []


@tool
def save_report(db: Annotated[Database, Inject("database")]) -> str:
    """Save the report to the database."""
    db.rows.append("report")
    return "saved"


SAVE_REPORT = ToolCallEvent(name="save_report", arguments="{}")
DELEGATE_TO_WRITER = ToolCallEvent(name="task_writer", arguments='{"objective":"Write the report"}')
DELEGATE_TO_EDITOR = ToolCallEvent(name="task_editor", arguments='{"objective":"Get the report written"}')


@pytest.mark.asyncio
class TestDelegatedDependencies:
    async def test_database_passed_to_ask_reaches_the_worker(self) -> None:
        request_db = Database()
        writer_db = Database()
        writer = Agent(
            "writer",
            config=TestConfig(SAVE_REPORT, "Saved."),
            tools=[save_report],
            dependencies={"database": writer_db},
        )
        coordinator = Agent(
            "coordinator",
            config=TestConfig(DELEGATE_TO_WRITER, "Done."),
            tools=[writer.as_tool(description="Writes reports")],
        )

        await coordinator.ask("Write the report", dependencies={"database": request_db})

        assert request_db.rows == ["report"]
        assert writer_db.rows == []

    async def test_worker_keeps_its_own_database_over_the_coordinators(self) -> None:
        coordinator_db = Database()
        writer_db = Database()
        writer = Agent(
            "writer",
            config=TestConfig(SAVE_REPORT, "Saved."),
            tools=[save_report],
            dependencies={"database": writer_db},
        )
        coordinator = Agent(
            "coordinator",
            config=TestConfig(DELEGATE_TO_WRITER, "Done."),
            tools=[writer.as_tool(description="Writes reports")],
            dependencies={"database": coordinator_db},
        )

        await coordinator.ask("Write the report")

        assert writer_db.rows == ["report"]
        assert coordinator_db.rows == []

    async def test_worker_without_a_database_uses_the_coordinators(self) -> None:
        coordinator_db = Database()
        writer = Agent(
            "writer",
            config=TestConfig(SAVE_REPORT, "Saved."),
            tools=[save_report],
        )
        coordinator = Agent(
            "coordinator",
            config=TestConfig(DELEGATE_TO_WRITER, "Done."),
            tools=[writer.as_tool(description="Writes reports")],
            dependencies={"database": coordinator_db},
        )

        await coordinator.ask("Write the report")

        assert coordinator_db.rows == ["report"]

    async def test_nested_worker_keeps_its_own_database_over_the_root_agents(self) -> None:
        root_db = Database()
        writer_db = Database()
        writer = Agent(
            "writer",
            config=TestConfig(SAVE_REPORT, "Saved."),
            tools=[save_report],
            dependencies={"database": writer_db},
        )
        editor = Agent(
            "editor",
            config=TestConfig(DELEGATE_TO_WRITER, "Edited."),
            tools=[writer.as_tool(description="Writes reports")],
        )
        coordinator = Agent(
            "coordinator",
            config=TestConfig(DELEGATE_TO_EDITOR, "Done."),
            tools=[editor.as_tool(description="Edits reports")],
            dependencies={"database": root_db},
        )

        await coordinator.ask("Write the report")

        assert writer_db.rows == ["report"]
        assert root_db.rows == []

    async def test_database_passed_to_ask_reaches_a_nested_worker(self) -> None:
        request_db = Database()
        writer_db = Database()
        writer = Agent(
            "writer",
            config=TestConfig(SAVE_REPORT, "Saved."),
            tools=[save_report],
            dependencies={"database": writer_db},
        )
        editor = Agent(
            "editor",
            config=TestConfig(DELEGATE_TO_WRITER, "Edited."),
            tools=[writer.as_tool(description="Writes reports")],
        )
        coordinator = Agent(
            "coordinator",
            config=TestConfig(DELEGATE_TO_EDITOR, "Done."),
            tools=[editor.as_tool(description="Edits reports")],
        )

        await coordinator.ask("Write the report", dependencies={"database": request_db})

        assert request_db.rows == ["report"]
        assert writer_db.rows == []
