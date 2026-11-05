# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""What an ACP agent can ask of AG2, as the agent sees it: files, terminals and permission."""

import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import acp
import pytest
from acp import RequestError, schema
from acp.agent.connection import AgentSideConnection

from ag2 import Agent
from ag2.acp.testing import duplex_acp_config

SESSION_ID = "calling-session"

Act = Callable[[AgentSideConnection, str], Awaitable[Any]]


class CallingAgent:
    """An ACP agent whose one prompt turn is a script of requests to the client."""

    def __init__(self, conn: AgentSideConnection, act: Act, outcome: list[Any]) -> None:
        self._conn = conn
        self._act = act
        self._outcome = outcome

    async def initialize(self, **kwargs: Any) -> schema.InitializeResponse:
        return schema.InitializeResponse(protocol_version=acp.PROTOCOL_VERSION)

    async def new_session(self, **kwargs: Any) -> schema.NewSessionResponse:
        return schema.NewSessionResponse(session_id=SESSION_ID)

    async def prompt(self, **kwargs: Any) -> schema.PromptResponse:
        try:
            self._outcome.append(await self._act(self._conn, SESSION_ID))
        except RequestError as error:
            self._outcome.append(error)
        return schema.PromptResponse(stop_reason="end_turn")


async def agent_asks(act: Act, workspace: Path, **config: Any) -> Any:
    """Run one turn in which the agent does ``act``, and return what ``act`` returned.

    A request the client refuses comes back as the ``RequestError`` the agent got.
    """
    outcome: list[Any] = []
    cfg = duplex_acp_config(lambda conn: CallingAgent(conn, act, outcome), cwd=str(workspace), **config)
    try:
        await Agent("acp", config=cfg).ask("go")
    finally:
        await cfg.aclose()
    [result] = outcome
    return result


@pytest.mark.asyncio
class TestFiles:
    async def test_a_file_the_agent_writes_can_be_read_back(self, tmp_path: Path) -> None:
        async def act(conn: AgentSideConnection, sid: str) -> str:
            await conn.write_text_file(session_id=sid, path="notes.txt", content="hi\nthere\n")
            return (await conn.read_text_file(session_id=sid, path="notes.txt")).content

        assert await agent_asks(act, tmp_path) == "hi\nthere\n"
        assert (tmp_path / "notes.txt").read_text() == "hi\nthere\n"

    async def test_an_absolute_path_inside_the_workspace_is_fine(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("hi\n")

        async def act(conn: AgentSideConnection, sid: str) -> str:
            return (await conn.read_text_file(session_id=sid, path=str(tmp_path / "notes.txt"))).content

        assert await agent_asks(act, tmp_path) == "hi\n"

    async def test_a_read_can_ask_for_a_window_of_lines(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("l1\nl2\nl3\nl4\n")

        async def act(conn: AgentSideConnection, sid: str) -> str:
            return (await conn.read_text_file(session_id=sid, path="f.txt", line=2, limit=2)).content

        assert await agent_asks(act, tmp_path) == "l2\nl3\n"

    @pytest.mark.parametrize("path", ["../../etc/passwd", "/etc/passwd"])
    async def test_a_path_outside_the_workspace_is_refused(self, tmp_path: Path, path: str) -> None:
        async def act(conn: AgentSideConnection, sid: str) -> str:
            return (await conn.read_text_file(session_id=sid, path=path)).content

        assert isinstance(await agent_asks(act, tmp_path), RequestError)

    async def test_a_write_outside_the_workspace_is_refused(self, tmp_path: Path) -> None:
        workspace = tmp_path / "work"
        workspace.mkdir()

        async def act(conn: AgentSideConnection, sid: str) -> None:
            await conn.write_text_file(session_id=sid, path="../escaped.txt", content="x")

        assert isinstance(await agent_asks(act, workspace), RequestError)
        assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.asyncio
class TestPermission:
    OPTIONS = [schema.PermissionOption(option_id="ok", kind="allow_once", name="Allow")]

    async def ask_permission(self, workspace: Path, policy: str) -> Any:
        async def act(conn: AgentSideConnection, sid: str) -> schema.RequestPermissionResponse:
            return await conn.request_permission(
                session_id=sid,
                options=self.OPTIONS,
                tool_call=schema.ToolCallUpdate(tool_call_id="tc1", title="Edit"),
            )

        return await agent_asks(act, workspace, permission_policy=policy)

    async def test_auto_allows_with_the_offered_option(self, tmp_path: Path) -> None:
        response = await self.ask_permission(tmp_path, "auto")

        assert response.outcome == schema.AllowedOutcome(option_id="ok", outcome="selected")

    async def test_deny_with_nothing_to_reject_with_is_a_clean_cancel(self, tmp_path: Path) -> None:
        # No reject option was offered, so there is nothing to pick; the answer must still be well-formed.
        response = await self.ask_permission(tmp_path, "deny")

        assert response.outcome == schema.DeniedOutcome(outcome="cancelled")


@pytest.mark.asyncio
class TestTerminal:
    async def test_a_command_runs_and_its_output_is_captured(self, tmp_path: Path) -> None:
        async def act(conn: AgentSideConnection, sid: str) -> tuple[Any, ...]:
            terminal = await conn.create_terminal(session_id=sid, command="printf", args=["hello"])
            exited = await conn.wait_for_terminal_exit(session_id=sid, terminal_id=terminal.terminal_id)
            output = await conn.terminal_output(session_id=sid, terminal_id=terminal.terminal_id)
            await conn.release_terminal(session_id=sid, terminal_id=terminal.terminal_id)
            return exited, output

        exited, output = await agent_asks(act, tmp_path)

        assert (exited.exit_code, exited.signal) == (0, None)
        assert (output.output, output.truncated) == ("hello", False)

    async def test_output_past_the_byte_limit_is_truncated(self, tmp_path: Path) -> None:
        async def act(conn: AgentSideConnection, sid: str) -> Any:
            terminal = await conn.create_terminal(
                session_id=sid, command="printf", args=["abcdefghij"], output_byte_limit=4
            )
            await conn.wait_for_terminal_exit(session_id=sid, terminal_id=terminal.terminal_id)
            return await conn.terminal_output(session_id=sid, terminal_id=terminal.terminal_id)

        output = await agent_asks(act, tmp_path)

        assert (output.output, output.truncated) == ("abcd", True)

    async def test_a_working_directory_outside_the_workspace_is_refused(self, tmp_path: Path) -> None:
        async def act(conn: AgentSideConnection, sid: str) -> Any:
            return await conn.create_terminal(session_id=sid, command="printf", args=["x"], cwd="../../etc")

        assert isinstance(await agent_asks(act, tmp_path), RequestError)

    async def test_a_running_command_can_be_killed(self, tmp_path: Path) -> None:
        async def act(conn: AgentSideConnection, sid: str) -> Any:
            terminal = await conn.create_terminal(session_id=sid, command="sleep", args=["30"])
            await conn.kill_terminal(session_id=sid, terminal_id=terminal.terminal_id)
            exited = await conn.wait_for_terminal_exit(session_id=sid, terminal_id=terminal.terminal_id)
            await conn.release_terminal(session_id=sid, terminal_id=terminal.terminal_id)
            return exited

        exited = await agent_asks(act, tmp_path)

        if sys.platform == "win32":
            # No signals on Windows: TerminateProcess surfaces as a non-zero exit code.
            assert exited.exit_code is not None
        else:
            assert exited.signal is not None
