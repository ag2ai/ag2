# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64

from acp import schema
from dirty_equals import IsPartialDict

from ag2.acp.events import ACPAvailableCommands, ACPModeChange, ACPPlan, ACPPlanEntry
from ag2.acp.mappers import (
    content_blocks_to_files,
    content_blocks_to_text,
    map_session_update,
    map_usage,
)
from ag2.events import ModelMessageChunk, ModelReasoning
from ag2.events.tool_events import BuiltinToolCallEvent, BuiltinToolResultEvent
from ag2.events.types import Usage


def _text(text: str) -> schema.TextContentBlock:
    return schema.TextContentBlock(type="text", text=text)


def test_message_chunk() -> None:
    ev = map_session_update(schema.AgentMessageChunk(session_update="agent_message_chunk", content=_text("hello")))
    assert isinstance(ev, ModelMessageChunk)
    assert ev.content == "hello"


def test_thought_chunk() -> None:
    ev = map_session_update(schema.AgentThoughtChunk(session_update="agent_thought_chunk", content=_text("thinking")))
    assert isinstance(ev, ModelReasoning)
    assert ev.content == "thinking"


def test_tool_call_start() -> None:
    ev = map_session_update(
        schema.ToolCallStart(
            session_update="tool_call",
            tool_call_id="tc1",
            title="Edit",
            raw_input={"path": "a.py"},
            status="pending",
        )
    )
    assert isinstance(ev, BuiltinToolCallEvent)
    assert ev.id == "tc1"
    assert ev.name == "Edit"
    assert ev.serialized_arguments == {"path": "a.py"}


def test_tool_call_progress() -> None:
    ev = map_session_update(
        schema.ToolCallProgress(
            session_update="tool_call_update",
            tool_call_id="tc1",
            title="Edit",
            status="completed",
            content=[schema.ContentToolCallContent(type="content", content=_text("done"))],
        )
    )
    assert isinstance(ev, BuiltinToolResultEvent)
    assert ev.parent_id == "tc1"


def test_plan() -> None:
    plan = map_session_update(
        schema.AgentPlanUpdate(
            session_update="plan",
            entries=[schema.PlanEntry(content="do x", status="pending", priority="high")],
        )
    )
    assert isinstance(plan, ACPPlan)
    assert plan.entries == [ACPPlanEntry(content="do x", status="pending", priority="high")]


def test_mode_and_commands() -> None:
    mode = map_session_update(schema.CurrentModeUpdate(session_update="current_mode_update", current_mode_id="edit"))
    assert isinstance(mode, ACPModeChange) and mode.mode_id == "edit"

    cmds = map_session_update(
        schema.AvailableCommandsUpdate(
            session_update="available_commands_update",
            available_commands=[schema.AvailableCommand(name="/test", description="run")],
        )
    )
    assert isinstance(cmds, ACPAvailableCommands) and cmds.commands == ["/test"]


def test_unhandled_updates_return_none() -> None:
    assert map_session_update(schema.SessionInfoUpdate(session_update="session_info_update", title="x")) is None
    assert map_session_update(schema.UsageUpdate(session_update="usage_update", size=1, used=1)) is None
    assert map_session_update(schema.UserMessageChunk(session_update="user_message_chunk", content=_text("hi"))) is None


def test_content_blocks_to_text_concatenates() -> None:
    text = content_blocks_to_text([
        _text("a"),
        schema.ImageContentBlock(type="image", data="x", mime_type="image/png"),
        _text("b"),
    ])
    assert text == "ab"


def test_content_blocks_to_files_decodes_image() -> None:
    data = base64.b64encode(b"img").decode()
    files = content_blocks_to_files([schema.ImageContentBlock(type="image", data=data, mime_type="image/png")])
    assert files[0].data == b"img"
    assert files[0].metadata == IsPartialDict({"mimeType": "image/png"})


def test_map_usage() -> None:
    usage = map_usage(
        schema.Usage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cached_read_tokens=2,
            thought_tokens=3,
        )
    )
    assert usage == Usage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cache_read_input_tokens=2,
        thinking_tokens=3,
    )


def test_map_usage_none() -> None:
    assert not map_usage(None)  # falsy empty Usage
