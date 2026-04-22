# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import pytest
from ag_ui.core import RunAgentInput, UserMessage

from autogen.beta import Agent, ToolResult
from autogen.beta.ag_ui import AGUIStream
from autogen.beta.ag_ui.stream import _stringify_tool_result
from autogen.beta.events import ToolCallEvent
from autogen.beta.events.input_events import BinaryInput, DataInput, FileIdInput, UrlInput
from autogen.beta.testing import TestConfig


def _run_input(user_text: str = "hi") -> RunAgentInput:
    return RunAgentInput(
        thread_id="t1",
        run_id="r1",
        messages=[UserMessage(id="m1", content=user_text)],
        tools=[],
        context=[],
        state={},
        forwarded_props={},
    )


async def _collect(stream: AGUIStream, run_input: RunAgentInput) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async for chunk in stream.dispatch(run_input):
        # AG-UI SSE frames are `data: <json>\n\n` — strip + decode.
        for line in chunk.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line.removeprefix("data: ")))
    return events


class TestStringifyToolResult:
    """Unit-level checks for the helper that flattens a ToolResult for AG-UI."""

    def test_text_input_passes_through(self) -> None:
        assert _stringify_tool_result(ToolResult("hello")) == "hello"

    def test_dict_becomes_json(self) -> None:
        out = _stringify_tool_result(ToolResult({"stdout": "x", "exit_code": 0}))
        assert json.loads(out) == {"stdout": "x", "exit_code": 0}

    def test_multi_part_joins_with_newlines(self) -> None:
        assert _stringify_tool_result(ToolResult("a", "b", "c")) == "a\nb\nc"

    def test_url_input(self) -> None:
        assert _stringify_tool_result(ToolResult(UrlInput(kind="image", url="https://x"))) == "https://x"

    def test_file_id_input(self) -> None:
        assert _stringify_tool_result(ToolResult(FileIdInput("file_abc"))) == "[file:file_abc]"

    def test_binary_input(self) -> None:
        result = _stringify_tool_result(ToolResult(BinaryInput(b"12345", media_type="image/png")))
        assert result == "[binary:image/png 5B]"

    def test_mixed_parts(self) -> None:
        result = _stringify_tool_result(ToolResult("caption", DataInput({"k": 1})))
        assert result == 'caption\n{"k": 1}'


@pytest.mark.asyncio
class TestToolResultInStream:
    """End-to-end: ensure a tool call followed by a tool result reaches the
    AG-UI wire as a ``TOOL_CALL_RESULT`` event with the expected string
    content. This guards against the regression where ``stream.py`` read the
    pre-rename ``event.content`` field that no longer exists.
    """

    async def test_string_tool_result_is_emitted(self) -> None:
        def say_hello() -> str:
            return "/path/to/data.csv"

        agent = Agent(
            "agui_tool_test",
            config=TestConfig(
                ToolCallEvent(name="say_hello"),
                "done",
            ),
            tools=[say_hello],
        )

        events = await _collect(AGUIStream(agent), _run_input())

        result_events = [e for e in events if e["type"] == "TOOL_CALL_RESULT"]
        assert len(result_events) == 1
        assert result_events[0]["content"] == "/path/to/data.csv"
        assert result_events[0]["role"] == "tool"

        # No error event should have fired.
        assert not [e for e in events if e["type"] == "RUN_ERROR"]

    async def test_dict_tool_result_is_json_encoded(self) -> None:
        def run_code() -> dict[str, Any]:
            return {"stdout": "hi", "exit_code": 0}

        agent = Agent(
            "agui_dict_test",
            config=TestConfig(
                ToolCallEvent(name="run_code"),
                "done",
            ),
            tools=[run_code],
        )

        events = await _collect(AGUIStream(agent), _run_input())

        result_events = [e for e in events if e["type"] == "TOOL_CALL_RESULT"]
        assert len(result_events) == 1
        assert json.loads(result_events[0]["content"]) == {"stdout": "hi", "exit_code": 0}
        assert not [e for e in events if e["type"] == "RUN_ERROR"]
