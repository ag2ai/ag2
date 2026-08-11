# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end over a real subprocess, the way an ACP Client actually launches an Agent.

Everything else in this directory drives ``ACPAgent`` in-process. This module
spawns it with ``run_stdio()`` behind a real pipe, so the stdio transport, JSON-RPC
framing and process lifecycle are all exercised for real.
"""

import sys
from pathlib import Path
from typing import Any

import acp
import pytest
from acp import schema

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Recorder:
    """Minimal ACP Client: records notifications, implements nothing else."""

    def __init__(self) -> None:
        self.updates: list[tuple[str, Any]] = []

    def updates_for(self, session_id: str) -> list[Any]:
        return [u for sid, u in self.updates if sid == session_id]

    async def session_update(self, *, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append((session_id, update))

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None


@pytest.mark.asyncio
async def test_a_client_can_launch_and_drive_the_agent_over_stdio() -> None:
    recorder = _Recorder()

    async with acp.spawn_agent_process(
        recorder,
        sys.executable,
        "-m",
        "test.acp._stdio_agent",
        cwd=str(REPO_ROOT),
    ) as (conn, process):
        init = await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)
        assert init.agent_info.name == "workie"

        # Two independent sessions, both prompted.
        first = await conn.new_session(cwd=str(REPO_ROOT))
        second = await conn.new_session(cwd=str(REPO_ROOT))
        assert first.session_id != second.session_id

        for session in (first, second):
            response = await conn.prompt(
                session_id=session.session_id,
                prompt=[acp.text_block("what's 100 + 100")],
            )
            assert response.stop_reason == "end_turn"

        for session in (first, second):
            updates = recorder.updates_for(session.session_id)
            assert [u.session_update for u in updates] == [
                "tool_call",
                "tool_call_update",
                "agent_message_chunk",
            ]
            [text] = [u.content.text for u in updates if isinstance(u, schema.AgentMessageChunk)]
            assert text == "200"

    # The context manager shuts the subprocess down; it must not be left running.
    assert process.returncode is not None
