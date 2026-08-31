# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Hosted MCP and shell calls reach the stream as tool call/result events.

Everything here drives the real `OpenAIResponsesConfig` over a mock HTTP
transport, so what is asserted is what a user observes on their own stream.
"""

import json
from typing import Any

import pytest
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls

from ag2 import Agent, MemoryStream, tool
from ag2.config.openai.mappers import events_to_responses_input
from ag2.events import TextInput, ToolNotFoundEvent
from ag2.tools.builtin.mcp_server import MCP_SERVER_TOOL_NAME, MCPServerTool
from ag2.tools.builtin.shell import SHELL_TOOL_NAME, ContainerAutoEnvironment, ShellTool

from ._helpers import (
    MCP_CALL,
    SHELL_CALL,
    SHELL_OUTPUT,
    calls,
    config,
    events_of,
    message,
    response,
    results,
)

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


@pytest.mark.asyncio
class TestHostedMcpCall:
    async def test_completed_call_produces_a_call_and_a_result(self) -> None:
        events = await events_of(MCP_CALL)

        [call] = calls(events)
        [result] = results(events)

        assert (call.id, call.name, call.arguments) == ("mcp_1", MCP_SERVER_TOOL_NAME, MCP_CALL["arguments"])
        assert (result.parent_id, result.name) == ("mcp_1", MCP_SERVER_TOOL_NAME)
        assert result.result.parts == [TextInput("an agent framework")]
        assert result.result.metadata == IsPartialDict({
            "server_label": "deepwiki",
            "tool": "ask_question",
            "status": "completed",
        })

    async def test_failure_is_observable(self) -> None:
        events = await events_of(MCP_CALL_FAILED)

        [result] = results(events)

        assert result.result.metadata == IsPartialDict({"status": "failed"})
        assert "error" in result.result.metadata

    async def test_listing_is_observable(self) -> None:
        events = await events_of(MCP_LIST_TOOLS)

        [call] = calls(events)
        [result] = results(events)

        assert call.name == MCP_SERVER_TOOL_NAME
        assert result.result.metadata == IsPartialDict({"server_label": "deepwiki", "tools": ["ask_question"]})

    async def test_failed_listing_is_observable(self) -> None:
        events = await events_of(MCP_LIST_TOOLS_FAILED)

        [result] = results(events)

        assert result.result.metadata == IsPartialDict({
            "error": {"type": "mcp_list_tools_error", "message": "connection refused"}
        })


@pytest.mark.asyncio
class TestHostedShellCall:
    async def test_completed_call_produces_a_call_and_a_result(self) -> None:
        events = await events_of(SHELL_CALL, SHELL_OUTPUT)

        [call] = calls(events)
        [result] = results(events)

        assert (call.id, call.name) == ("sh_1", SHELL_TOOL_NAME)
        assert json.loads(call.arguments) == {"commands": ["echo hi", "ls"]}
        assert result.parent_id == "sh_1"
        assert result.name == SHELL_TOOL_NAME

    async def test_result_carries_the_command_and_its_output(self) -> None:
        events = await events_of(SHELL_CALL, SHELL_OUTPUT)

        [result] = results(events)

        assert result.result.parts == [TextInput("hi\n"), TextInput("warning\n")]
        assert result.result.metadata == IsPartialDict({"commands": ["echo hi", "ls"], "status": "completed"})

    async def test_a_call_with_no_output_yet_is_still_a_call(self) -> None:
        events = await events_of(SHELL_CALL)

        assert [c.name for c in calls(events)] == [SHELL_TOOL_NAME]
        assert results(events) == []


@pytest.mark.asyncio
class TestTheToolExecutorAbsorbsThem:
    async def test_neither_produces_a_tool_not_found_event(self) -> None:
        stream = MemoryStream()
        agent = Agent(
            "hosted",
            config=config(response(MCP_CALL, SHELL_CALL, SHELL_OUTPUT, message("done"))),
            tools=[
                MCPServerTool(server_url="https://example.invalid/mcp", server_label="deepwiki"),
                ShellTool(environment=ContainerAutoEnvironment()),
            ],
        )

        await agent.ask("go", stream=stream)
        events = list(await stream.history.get_events())

        assert [e for e in events if isinstance(e, ToolNotFoundEvent)] == []
        assert sorted(c.name for c in calls(events)) == [MCP_SERVER_TOOL_NAME, SHELL_TOOL_NAME]

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
            config=config(
                response(MCP_CALL, function_call),
                response(message("done")),
            ),
            tools=[MCPServerTool(server_url="https://example.invalid/mcp", server_label="deepwiki"), note],
        )

        reply = await agent.ask("go")

        assert called == ["hello"]
        assert reply.body == "done"


async def _replayed(*output: dict[str, Any]) -> list[dict[str, Any]]:
    """The next request's input items, after a turn whose response carried `output`."""
    return events_to_responses_input(await events_of(*output), SerializerCls)


@pytest.mark.asyncio
class TestReplayingAHostedCall:
    """A hosted item ag2 now reports has to survive the next turn's request.

    `shell_call` is the only hosted item whose result lives in a *second* output
    item, so replaying the call alone would send the API a command with no
    outcome.
    """

    async def test_shell_call_replays_with_its_output(self) -> None:
        replayed = await _replayed(SHELL_CALL, SHELL_OUTPUT)

        assert [item["type"] for item in replayed if item.get("type", "").startswith("shell_call")] == [
            "shell_call",
            "shell_call_output",
        ]

    async def test_mcp_call_replays_whole(self) -> None:
        replayed = await _replayed(MCP_CALL)

        assert [item for item in replayed if item.get("type") == "mcp_call"] == [
            IsPartialDict({"id": "mcp_1", "name": "ask_question", "output": "an agent framework"})
        ]

    async def test_an_unanswered_shell_call_is_not_replayed(self) -> None:
        """A `shell_call` with no `shell_call_output` is not a legal input item.

        A turn that ends between the command and the container's output — an
        `incomplete` response, say — leaves the call unanswered. Replaying it
        alone sends the API a command with no outcome, and the next request 400s.
        """
        replayed = await _replayed(SHELL_CALL)

        assert [item for item in replayed if item.get("type", "").startswith("shell_call")] == []

    async def test_no_replayed_item_carries_an_output_only_field(self) -> None:
        """`created_by` rides on a hosted item's way out and is rejected on the way in.

        The API answers it with `Unknown parameter: input[N].created_by`, so a
        dump of the SDK model whole has to leave it behind.

        The field has to be *set* on both items for this to test anything: it is
        optional and the API leaves it unset today, so against the plain fixtures
        `exclude_none` drops it on its own and the guard is never exercised.
        """
        replayed = await _replayed(
            {**SHELL_CALL, "created_by": "assistant"},
            {**SHELL_OUTPUT, "created_by": "assistant"},
        )

        assert [item["type"] for item in replayed if item.get("type", "").startswith("shell_call")] == [
            "shell_call",
            "shell_call_output",
        ]
        assert [item for item in replayed if "created_by" in item] == []
