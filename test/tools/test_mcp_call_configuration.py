# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import sys
from functools import partial
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("mcp")

from ag2 import Agent, Context, Variable
from ag2.events import ToolCallEvent, ToolResultEvent, ToolResultsEvent
from ag2.middleware import ToolExecution
from ag2.stream import MemoryStream
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools import MCPStdioServerConfig, MCPToolkit


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["args", "env", "cwd"])
async def test_default_factory_is_not_resolved_again_for_tool_execution(field: str, tmp_path: Path) -> None:
    script = str(Path(__file__).with_name("fixtures") / "mcp_context_server.py")
    if field == "args":
        values: list[Any] = [[script, "first"], [script, "second"]]
    elif field == "env":
        values = [{"AG2_MCP_TEST_TOOL": "first"}, {"AG2_MCP_TEST_TOOL": "second"}]
    else:
        values = [tmp_path / "first", tmp_path / "second"]
        for directory in values:
            directory.mkdir()
    options: dict[str, Any] = {
        "args": [script],
        field: Variable("launch", default_factory=partial(next, iter(values), values[-1])),
    }
    config = MCPStdioServerConfig(command=sys.executable, **options)
    call = ToolCallEvent(name="first")
    tracking = TrackingConfig(TestConfig(call, "done"))
    agent = Agent("consumer", config=tracking, tools=[MCPToolkit(config)])

    await agent.ask("Call the available tool")

    tracking.mock.assert_called_with(ToolResultsEvent([ToolResultEvent.from_call(call, result="called first")]))


@pytest.mark.asyncio
async def test_discovered_tool_keeps_its_configuration_after_rediscovery() -> None:
    script = str(Path(__file__).with_name("fixtures") / "mcp_context_server.py")
    config = MCPStdioServerConfig(command=sys.executable, args=[script, "first"])
    toolkit = MCPToolkit(config)
    context = Context(stream=MemoryStream())
    await toolkit.schemas(context)
    first_tool = cast(ToolExecution, toolkit.tools[0])
    config.args = [script, "second"]
    await toolkit.schemas(context)
    second_tool = cast(ToolExecution, toolkit.tools[0])
    first_call = ToolCallEvent(name="first")
    second_call = ToolCallEvent(name="second")

    assert await first_tool(first_call, context) == ToolResultEvent.from_call(first_call, result="called first")
    assert await second_tool(second_call, context) == ToolResultEvent.from_call(second_call, result="called second")
