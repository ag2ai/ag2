# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from autogen.beta.debug.models import BreakpointType
from autogen.beta.debug.session import DebugSession


@pytest.mark.asyncio()
async def test_add_event() -> None:
    session = DebugSession("test-session")
    record = await session.add_event("ModelRequest", {"content": "hello"})
    assert record.event_type == "ModelRequest"
    assert record.event_data == {"content": "hello"}
    assert len(session.events) == 1


@pytest.mark.asyncio()
async def test_add_breakpoint_blocks_until_resumed() -> None:
    session = DebugSession("test-session")

    bp_id_holder: list[str] = []
    completed = asyncio.Event()

    async def _hit_bp() -> None:
        bp_id = await session.add_breakpoint(BreakpointType.TURN_START, "ModelRequest", {"content": "hi"})
        bp_id_holder.append(bp_id)
        completed.set()

    task = asyncio.create_task(_hit_bp())

    # Give the task a moment to start and block on the breakpoint
    await asyncio.sleep(0.01)
    assert not completed.is_set(), "Breakpoint should still be blocking"
    assert session._pending_bp_id is not None

    pending_bp_id = session._pending_bp_id
    success = await session.resume(pending_bp_id)
    assert success

    await asyncio.wait_for(task, timeout=1.0)
    assert completed.is_set()
    assert bp_id_holder[0] == pending_bp_id


@pytest.mark.asyncio()
async def test_resume_wrong_id_returns_false() -> None:
    session = DebugSession("test-session")

    task = asyncio.create_task(
        session.add_breakpoint(BreakpointType.TOOL_CALL, "ToolCallEvent", {})
    )
    await asyncio.sleep(0.01)

    result = await session.resume("nonexistent-id")
    assert result is False

    # Clean up: resume with the real id
    assert session._pending_bp_id is not None
    await session.resume(session._pending_bp_id)
    await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio()
async def test_breakpoint_record_marked_resumed() -> None:
    session = DebugSession("test-session")

    task = asyncio.create_task(
        session.add_breakpoint(BreakpointType.TURN_START, "ModelRequest", {})
    )
    await asyncio.sleep(0.01)

    bp_id = session._pending_bp_id
    assert bp_id is not None
    await session.resume(bp_id)
    await asyncio.wait_for(task, timeout=1.0)

    assert session.breakpoints[0].resumed is True
