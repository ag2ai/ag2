# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Hosted MCP and shell calls reach the stream as tool call/result events.

Everything here drives the real `OpenAIResponsesConfig` over a mock HTTP
transport, so what is asserted is what a user observes on their own stream.
"""

import json
from typing import Any

import httpx2
import pytest
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls

from ag2 import Agent, Context, MemoryStream, tool
from ag2.config import OpenAIResponsesConfig
from ag2.config.openai.mappers import events_to_responses_input
from ag2.events import (
    BaseEvent,
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    ModelRequest,
    TextInput,
    ToolNotFoundEvent,
)
from ag2.tools.builtin.mcp_server import MCP_SERVER_TOOL_NAME, MCPServerTool
from ag2.tools.builtin.shell import SHELL_TOOL_NAME, ShellTool

_USAGE = {
    "input_tokens": 1,
    "output_tokens": 1,
    "total_tokens": 2,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0},
}

MCP_CALL = {
    "id": "mcp_1",
    "type": "mcp_call",
    "name": "ask_question",
    "server_label": "deepwiki",
    "arguments": '{"question": "what is ag2?"}',
    "output": "an agent framework",
    "status": "completed",
}

MCP_CALL_FAILED = {
    "id": "mcp_2",
    "type": "mcp_call",
    "name": "ask_question",
    "server_label": "deepwiki",
    "arguments": "{}",
    "status": "failed",
    "error": {"type": "mcp_protocol_error", "code": -32601, "message": "Method not found"},
}

MCP_LIST_TOOLS = {
    "id": "mcpl_1",
    "type": "mcp_list_tools",
    "server_label": "deepwiki",
    "tools": [{"name": "ask_question", "input_schema": {"type": "object"}, "description": "Ask."}],
}

MCP_LIST_TOOLS_FAILED = {
    "id": "mcpl_2",
    "type": "mcp_list_tools",
    "server_label": "deepwiki",
    "tools": [],
    "error": "connection refused",
}

SHELL_CALL = {
    "id": "sh_1",
    "type": "shell_call",
    "call_id": "call_1",
    "status": "completed",
    "action": {"commands": ["echo hi", "ls"], "timeout_ms": 1000},
}

SHELL_OUTPUT = {
    "id": "sho_1",
    "type": "shell_call_output",
    "call_id": "call_1",
    "status": "completed",
    "output": [{"stdout": "hi\n", "stderr": "warning\n", "outcome": {"type": "exit", "exit_code": 0}}],
}


def _message(text: str) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _response(*output: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-5",
        "status": "completed",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": list(output),
        "usage": _USAGE,
    }


def _config(*turns: dict[str, Any]) -> OpenAIResponsesConfig:
    """A config whose transport replays one crafted Responses payload per call."""
    remaining = list(turns)

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=remaining.pop(0) if len(remaining) > 1 else remaining[0])

    return OpenAIResponsesConfig(
        model="gpt-5",
        api_key="test",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


async def _events_of(*output: dict[str, Any]) -> list[BaseEvent]:
    stream = MemoryStream()
    client = _config(_response(*output)).create()

    await client(
        messages=[ModelRequest([TextInput("go")])],
        context=Context(stream=stream),
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )

    return list(await stream.history.get_events())


def _calls(events: list[BaseEvent]) -> list[BuiltinToolCallEvent]:
    return [e for e in events if isinstance(e, BuiltinToolCallEvent)]


def _results(events: list[BaseEvent]) -> list[BuiltinToolResultEvent]:
    return [e for e in events if isinstance(e, BuiltinToolResultEvent)]


@pytest.mark.asyncio
class TestHostedMcpCall:
    async def test_completed_call_produces_a_call_and_a_result(self) -> None:
        events = await _events_of(MCP_CALL)

        [call] = _calls(events)
        [result] = _results(events)

        assert (call.id, call.name, call.arguments) == ("mcp_1", MCP_SERVER_TOOL_NAME, MCP_CALL["arguments"])
        assert (result.parent_id, result.name) == ("mcp_1", MCP_SERVER_TOOL_NAME)
        assert result.result.parts == [TextInput("an agent framework")]
        assert result.result.metadata == IsPartialDict({
            "server_label": "deepwiki",
            "tool": "ask_question",
            "status": "completed",
        })

    async def test_failure_is_observable(self) -> None:
        events = await _events_of(MCP_CALL_FAILED)

        [result] = _results(events)

        assert result.result.metadata == IsPartialDict({"status": "failed"})
        assert "error" in result.result.metadata

    async def test_listing_is_observable(self) -> None:
        events = await _events_of(MCP_LIST_TOOLS)

        [call] = _calls(events)
        [result] = _results(events)

        assert call.name == MCP_SERVER_TOOL_NAME
        assert result.result.metadata == IsPartialDict({"server_label": "deepwiki", "tools": ["ask_question"]})

    async def test_failed_listing_is_observable(self) -> None:
        events = await _events_of(MCP_LIST_TOOLS_FAILED)

        [result] = _results(events)

        assert result.result.metadata == IsPartialDict({"error": "connection refused"})


@pytest.mark.asyncio
class TestHostedShellCall:
    async def test_completed_call_produces_a_call_and_a_result(self) -> None:
        events = await _events_of(SHELL_CALL, SHELL_OUTPUT)

        [call] = _calls(events)
        [result] = _results(events)

        assert (call.id, call.name) == ("sh_1", SHELL_TOOL_NAME)
        assert json.loads(call.arguments) == {"commands": ["echo hi", "ls"]}
        assert result.parent_id == "sh_1"
        assert result.name == SHELL_TOOL_NAME

    async def test_result_carries_the_command_and_its_output(self) -> None:
        events = await _events_of(SHELL_CALL, SHELL_OUTPUT)

        [result] = _results(events)

        assert result.result.parts == [TextInput("hi\n"), TextInput("warning\n")]
        assert result.result.metadata == IsPartialDict({"commands": ["echo hi", "ls"], "status": "completed"})

    async def test_a_call_with_no_output_yet_is_still_a_call(self) -> None:
        events = await _events_of(SHELL_CALL)

        assert [c.name for c in _calls(events)] == [SHELL_TOOL_NAME]
        assert _results(events) == []


@pytest.mark.asyncio
class TestTheToolExecutorAbsorbsThem:
    async def test_neither_produces_a_tool_not_found_event(self) -> None:
        stream = MemoryStream()
        agent = Agent(
            "hosted",
            config=_config(_response(MCP_CALL, SHELL_CALL, SHELL_OUTPUT, _message("done"))),
            tools=[
                MCPServerTool(server_url="https://example.invalid/mcp", server_label="deepwiki"),
                ShellTool(),
            ],
        )

        await agent.ask("go", stream=stream)
        events = list(await stream.history.get_events())

        assert [e for e in events if isinstance(e, ToolNotFoundEvent)] == []
        assert sorted(c.name for c in _calls(events)) == [MCP_SERVER_TOOL_NAME, SHELL_TOOL_NAME]

    async def test_a_caller_function_tool_still_dispatches(self) -> None:
        called: list[str] = []

        @tool
        def note(text: str) -> str:
            """Write a note."""
            called.append(text)
            return "noted"

        function_call = {
            "id": "fc_1",
            "type": "function_call",
            "call_id": "fc_call_1",
            "name": "note",
            "arguments": '{"text": "hello"}',
            "status": "completed",
        }

        agent = Agent(
            "hosted",
            config=_config(
                _response(MCP_CALL, function_call),
                _response(_message("done")),
            ),
            tools=[MCPServerTool(server_url="https://example.invalid/mcp", server_label="deepwiki"), note],
        )

        reply = await agent.ask("go")

        assert called == ["hello"]
        assert reply.body == "done"


@pytest.mark.asyncio
class TestReplayingAHostedCall:
    """A hosted item ag2 now reports has to survive the next turn's request.

    `shell_call` is the only hosted item whose result lives in a *second* output
    item, so replaying the call alone would send the API a command with no
    outcome.
    """

    async def test_shell_call_replays_with_its_output(self) -> None:
        stream = MemoryStream()
        client = _config(_response(SHELL_CALL, SHELL_OUTPUT)).create()
        await client(
            messages=[ModelRequest([TextInput("go")])],
            context=Context(stream=stream),
            tools=[],
            response_schema=None,
            serializer=SerializerCls,
        )

        replayed = events_to_responses_input(list(await stream.history.get_events()), SerializerCls)

        assert [item["type"] for item in replayed if item.get("type", "").startswith("shell_call")] == [
            "shell_call",
            "shell_call_output",
        ]

    async def test_mcp_call_replays_whole(self) -> None:
        stream = MemoryStream()
        client = _config(_response(MCP_CALL)).create()
        await client(
            messages=[ModelRequest([TextInput("go")])],
            context=Context(stream=stream),
            tools=[],
            response_schema=None,
            serializer=SerializerCls,
        )

        replayed = events_to_responses_input(list(await stream.history.get_events()), SerializerCls)

        assert [item for item in replayed if item.get("type") == "mcp_call"] == [
            IsPartialDict({"id": "mcp_1", "name": "ask_question", "output": "an agent framework"})
        ]

    async def test_an_unanswered_shell_call_is_not_replayed(self) -> None:
        """A `shell_call` with no `shell_call_output` is not a legal input item.

        A turn that ends between the command and the container's output — an
        `incomplete` response, say — leaves the call unanswered. Replaying it
        alone sends the API a command with no outcome, and the next request 400s.
        """
        stream = MemoryStream()
        client = _config(_response(SHELL_CALL)).create()
        await client(
            messages=[ModelRequest([TextInput("go")])],
            context=Context(stream=stream),
            tools=[],
            response_schema=None,
            serializer=SerializerCls,
        )

        replayed = events_to_responses_input(list(await stream.history.get_events()), SerializerCls)

        assert [item for item in replayed if item.get("type", "").startswith("shell_call")] == []
