# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A streamed hosted shell call shows both halves as they happen.

The model composes a command, then a container runs it. Until it finishes there
is nothing to show, and a shell call is the slowest hosted tool ag2 supports —
so both halves arrive as increments, and neither is allowed to contaminate the
assistant's reply stream.
"""

import json
from typing import Any

import httpx2
import pytest
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config import OpenAIResponsesConfig
from ag2.config.openai.events import OpenAIShellCommandChunk, OpenAIShellOutputChunk
from ag2.events import (
    BaseEvent,
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    ModelMessageChunk,
    ModelRequest,
    TextInput,
    is_conversational,
)
from ag2.tools.builtin.shell import SHELL_TOOL_NAME

from .test_hosted_mcp_and_shell_events import SHELL_CALL, SHELL_OUTPUT, _USAGE

_RESPONSE_SHELL: dict[str, Any] = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "model": "gpt-5",
    "status": "completed",
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "output": [SHELL_CALL, SHELL_OUTPUT],
    "usage": _USAGE,
}

STREAM_EVENTS: list[dict[str, Any]] = [
    {
        "type": "response.shell_call_command.added",
        "sequence_number": 1,
        "output_index": 0,
        "command_index": 0,
        "command": "",
    },
    {
        "type": "response.shell_call_command.delta",
        "sequence_number": 2,
        "output_index": 0,
        "command_index": 0,
        "delta": "echo ",
    },
    {
        "type": "response.shell_call_command.delta",
        "sequence_number": 3,
        "output_index": 0,
        "command_index": 0,
        "delta": "hi",
    },
    {
        "type": "response.shell_call_command.done",
        "sequence_number": 4,
        "output_index": 0,
        "command_index": 0,
        "command": "echo hi",
    },
    {
        "type": "response.shell_call_command.delta",
        "sequence_number": 5,
        "output_index": 0,
        "command_index": 1,
        "delta": "ls",
    },
    {"type": "response.output_item.done", "sequence_number": 6, "output_index": 0, "item": SHELL_CALL},
    {
        "type": "response.shell_call_output_content.delta",
        "sequence_number": 7,
        "output_index": 0,
        "command_index": 0,
        "item_id": "sho_1",
        "delta": {"stdout": "hi\n", "stderr": None},
    },
    {
        "type": "response.shell_call_output_content.delta",
        "sequence_number": 8,
        "output_index": 0,
        "command_index": 1,
        "item_id": "sho_1",
        "delta": {"stdout": None, "stderr": "warning\n"},
    },
    {"type": "response.output_item.done", "sequence_number": 9, "output_index": 1, "item": SHELL_OUTPUT},
    {
        "type": "response.output_text.delta",
        "sequence_number": 10,
        "output_index": 2,
        "item_id": "msg_1",
        "content_index": 0,
        "delta": "all done",
        "logprobs": [],
    },
    {"type": "response.completed", "sequence_number": 11, "response": _RESPONSE_SHELL},
]


def _sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


async def _streamed_events(events: list[dict[str, Any]] = STREAM_EVENTS) -> list[BaseEvent]:
    """Every event the run puts on the stream, transient ones included.

    Increments are transient by design, so history never holds them — a
    subscriber is the only place they are observable, and the only place a user
    would watch them either.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=_sse(events), headers={"content-type": "text/event-stream"})

    config = OpenAIResponsesConfig(
        model="gpt-5",
        api_key="test",
        streaming=True,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    stream = MemoryStream()
    captured: list[BaseEvent] = []

    async def capture(event: BaseEvent) -> None:
        captured.append(event)

    stream.subscribe(capture)

    await config.create()(
        messages=[ModelRequest([TextInput("run it")])],
        context=Context(stream=stream),
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )

    return captured


@pytest.mark.asyncio
class TestCommandIncrements:
    async def test_the_command_is_emitted_as_it_is_composed(self) -> None:
        chunks = [e for e in await _streamed_events() if isinstance(e, OpenAIShellCommandChunk)]

        assert [c.content for c in chunks] == ["echo ", "hi", "ls"]

    async def test_an_increment_identifies_its_command(self) -> None:
        chunks = [e for e in await _streamed_events() if isinstance(e, OpenAIShellCommandChunk)]

        assert [c.command_index for c in chunks] == [0, 0, 1]


@pytest.mark.asyncio
class TestOutputIncrements:
    async def test_output_is_emitted_as_the_container_produces_it(self) -> None:
        chunks = [e for e in await _streamed_events() if isinstance(e, OpenAIShellOutputChunk)]

        assert len(chunks) == 2

    async def test_stdout_and_stderr_are_distinguishable(self) -> None:
        chunks = [e for e in await _streamed_events() if isinstance(e, OpenAIShellOutputChunk)]

        assert [(c.stdout, c.stderr) for c in chunks] == [("hi\n", None), (None, "warning\n")]

    async def test_an_increment_identifies_its_command(self) -> None:
        chunks = [e for e in await _streamed_events() if isinstance(e, OpenAIShellOutputChunk)]

        assert [c.command_index for c in chunks] == [0, 1]


@pytest.mark.asyncio
class TestIncrementsAreTransient:
    async def test_neither_type_counts_as_conversation(self) -> None:
        assert OpenAIShellCommandChunk.__transient__ is True
        assert OpenAIShellOutputChunk.__transient__ is True

    async def test_history_management_excludes_them(self) -> None:
        events = await _streamed_events()
        increments = [e for e in events if isinstance(e, (OpenAIShellCommandChunk, OpenAIShellOutputChunk))]

        assert increments != []
        assert [e for e in increments if is_conversational(e)] == []


@pytest.mark.asyncio
class TestTheFinishedCallIsUnchanged:
    async def test_the_call_and_result_events_still_arrive(self) -> None:
        events = await _streamed_events()

        [call] = [e for e in events if isinstance(e, BuiltinToolCallEvent)]
        [result] = [e for e in events if isinstance(e, BuiltinToolResultEvent)]

        assert (call.id, call.name) == ("sh_1", SHELL_TOOL_NAME)
        assert json.loads(call.arguments) == {"commands": ["echo hi", "ls"]}
        assert (result.parent_id, result.name) == ("sh_1", SHELL_TOOL_NAME)
        assert result.result.parts == [TextInput("hi\n"), TextInput("warning\n")]

    async def test_no_shell_output_appears_among_message_chunks(self) -> None:
        events = await _streamed_events()

        assert [c.content for c in events if isinstance(c, ModelMessageChunk)] == ["all done"]
