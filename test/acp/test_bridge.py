# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from acp import schema

from ag2.acp.bridge import BridgeState, _confine, make_bridge
from ag2.acp.config import ACPConfig
from ag2.events import BaseEvent, ModelMessageChunk, ModelReasoning


def _text(text: str) -> schema.TextContentBlock:
    return schema.TextContentBlock(type="text", text=text)


class FakeContext:
    def __init__(self) -> None:
        self.sent: list[BaseEvent] = []

    async def send(self, event: BaseEvent) -> None:
        self.sent.append(event)


def _state(context: FakeContext, **cfg: object) -> BridgeState:
    st = BridgeState(ACPConfig(**cfg))  # type: ignore[arg-type]
    st.context = context  # type: ignore[assignment]
    st.begin_turn()
    return st


def test_confine_allows_inside_root(tmp_path: Path) -> None:
    p = _confine(str(tmp_path), str(tmp_path / "a.py"))
    assert p.startswith(str(tmp_path))


def test_confine_allows_relative(tmp_path: Path) -> None:
    p = _confine(str(tmp_path), "sub/a.py")
    assert p.startswith(str(tmp_path))


def test_confine_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        _confine(str(tmp_path), str(tmp_path / ".." / "etc" / "passwd"))


@pytest.mark.asyncio
async def test_handle_update_sends_event_and_accumulates_text() -> None:
    ctx = FakeContext()
    st = _state(ctx)
    await st.handle_update(schema.AgentMessageChunk(session_update="agent_message_chunk", content=_text("Hello ")))
    await st.handle_update(schema.AgentMessageChunk(session_update="agent_message_chunk", content=_text("world")))
    assert all(isinstance(e, ModelMessageChunk) for e in ctx.sent)
    assert st.turn_text == "Hello world"


@pytest.mark.asyncio
async def test_thought_is_sent_but_not_in_turn_text() -> None:
    ctx = FakeContext()
    st = _state(ctx)
    await st.handle_update(schema.AgentThoughtChunk(session_update="agent_thought_chunk", content=_text("thinking")))
    assert isinstance(ctx.sent[0], ModelReasoning)
    assert st.turn_text == ""


@pytest.mark.asyncio
async def test_unknown_update_sends_nothing() -> None:
    ctx = FakeContext()
    st = _state(ctx)
    await st.handle_update(schema.SessionInfoUpdate(session_update="session_info_update", title="x"))
    assert ctx.sent == []


@pytest.mark.asyncio
async def test_begin_turn_resets_buffer() -> None:
    ctx = FakeContext()
    st = _state(ctx)
    await st.handle_update(schema.AgentMessageChunk(session_update="agent_message_chunk", content=_text("a")))
    st.begin_turn()
    assert st.turn_text == ""


def test_fs_write_then_read(tmp_path: Path) -> None:
    st = BridgeState(ACPConfig(cwd=str(tmp_path)))
    st.write_text_file("hi\nthere\n", "notes.txt")
    assert (tmp_path / "notes.txt").read_text() == "hi\nthere\n"
    assert st.read_text_file("notes.txt") == "hi\nthere\n"


def test_fs_read_with_line_and_limit(tmp_path: Path) -> None:
    st = BridgeState(ACPConfig(cwd=str(tmp_path)))
    st.write_text_file("l1\nl2\nl3\nl4\n", "f.txt")
    assert st.read_text_file("f.txt", line=2, limit=2) == "l2\nl3\n"


def test_fs_rejects_escape(tmp_path: Path) -> None:
    st = BridgeState(ACPConfig(cwd=str(tmp_path)))
    with pytest.raises(PermissionError):
        st.read_text_file("../../etc/passwd")


@pytest.mark.asyncio
async def test_permission_auto_returns_allow_id() -> None:
    st = BridgeState(ACPConfig(permission_policy="auto"))
    chosen = await st.resolve_permission(
        [schema.PermissionOption(option_id="ok", kind="allow_once", name="Allow")],
        schema.ToolCallUpdate(tool_call_id="tc1", title="Edit"),
    )
    assert chosen == "ok"


@pytest.mark.asyncio
async def test_request_permission_allows_selected_option() -> None:
    bridge = make_bridge(ACPConfig(permission_policy="auto"))
    resp = await bridge.request_permission(
        options=[schema.PermissionOption(option_id="ok", kind="allow_once", name="Allow")],
        session_id="s",
        tool_call=schema.ToolCallUpdate(tool_call_id="tc1", title="Edit"),
    )
    assert resp.outcome == schema.AllowedOutcome(option_id="ok", outcome="selected")


@pytest.mark.asyncio
async def test_request_permission_denies_without_crashing() -> None:
    # policy="deny" but no reject option -> resolve yields None -> the bridge must
    # return a well-formed DeniedOutcome (regression: DeniedOutcome() was invalid).
    bridge = make_bridge(ACPConfig(permission_policy="deny"))
    resp = await bridge.request_permission(
        options=[schema.PermissionOption(option_id="ok", kind="allow_once", name="Allow")],
        session_id="s",
        tool_call=schema.ToolCallUpdate(tool_call_id="tc1", title="Edit"),
    )
    assert resp.outcome == schema.DeniedOutcome(outcome="cancelled")


@pytest.mark.asyncio
async def test_terminal_runs_and_captures_output(tmp_path: Path) -> None:
    st = BridgeState(ACPConfig(cwd=str(tmp_path)))
    tid = await st.terminals.create("printf", ["hello"])
    exit_status = await st.terminals.wait(tid)
    output, truncated, status = st.terminals.output(tid)
    assert output == "hello"
    assert truncated is False
    assert exit_status == schema.TerminalExitStatus(exit_code=0, signal=None)
    assert status == schema.TerminalExitStatus(exit_code=0, signal=None)
    await st.terminals.release(tid)


@pytest.mark.asyncio
async def test_terminal_output_byte_limit_truncates(tmp_path: Path) -> None:
    st = BridgeState(ACPConfig(cwd=str(tmp_path)))
    tid = await st.terminals.create("printf", ["abcdefghij"], output_byte_limit=4)
    await st.terminals.wait(tid)
    output, truncated, _ = st.terminals.output(tid)
    assert output == "abcd"
    assert truncated is True
    await st.terminals.release(tid)


@pytest.mark.asyncio
async def test_terminal_cwd_confined(tmp_path: Path) -> None:
    st = BridgeState(ACPConfig(cwd=str(tmp_path)))
    with pytest.raises(PermissionError):
        await st.terminals.create("printf", ["x"], cwd="../../etc")


@pytest.mark.asyncio
async def test_terminal_kill(tmp_path: Path) -> None:
    st = BridgeState(ACPConfig(cwd=str(tmp_path)))
    tid = await st.terminals.create("sleep", ["30"])
    await st.terminals.kill(tid)
    exit_status = await st.terminals.wait(tid)
    assert exit_status.signal is not None  # terminated by signal
    await st.terminals.release(tid)
