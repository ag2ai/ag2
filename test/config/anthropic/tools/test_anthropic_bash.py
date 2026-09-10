# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

import pytest

from ag2 import Context, MemoryStream, ToolResult
from ag2.config.anthropic.mappers import tool_to_api
from ag2.events import ToolCallEvent, ToolResultEvent
from ag2.exceptions import UnsupportedToolError
from ag2.middleware import BaseMiddleware, ToolExecution
from ag2.tools.builtin.anthropic_bash import AnthropicBashTool
from ag2.tools.builtin.shell import ShellTool
from ag2.tools.sandbox import LocalEnvironment


@pytest.mark.asyncio
async def test_defaults(context: Context) -> None:
    tool = AnthropicBashTool()

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {"type": "bash_20250124", "name": "bash"}


@pytest.mark.asyncio
async def test_schema_type_matches_the_wire_name(context: Context) -> None:
    # ``known_tools`` stores ``schema.type`` for non-function schemas, and the
    # incoming ToolCallEvent carries Anthropic's wire name. They must be equal
    # or every call trips the not-found guard instead of reaching the executor.
    tool = AnthropicBashTool()

    [schema] = await tool.schemas(context)

    assert schema.type == tool_to_api(schema)["name"] == tool.name


@pytest.mark.asyncio
async def test_shell_tool_is_still_rejected(context: Context) -> None:
    # ShellTool is the provider-executed capability flag; Anthropic's bash is
    # client-executed. Adding the latter must not make the former mappable.
    [schema] = await ShellTool().schemas(context)

    with pytest.raises(UnsupportedToolError):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_middleware_wraps_execution(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # The tool runs model-authored shell commands, so a governance/telemetry
    # middleware must see the call. Ignoring the argument silently bypassed it.
    seen: list[str] = []

    class Deny(BaseMiddleware):
        async def on_tool_execution(
            self, call_next: ToolExecution, event: ToolCallEvent, context: Context
        ) -> ToolResultEvent:
            seen.append(event.name)
            return ToolResultEvent.from_call(event, result="denied by policy")

    stream = MemoryStream()
    context = Context(stream=stream)
    tool = AnthropicBashTool(LocalEnvironment(str(tmp_path)))
    call = ToolCallEvent(id="tc_1", name="bash", arguments='{"command": "echo hi"}')

    with ExitStack() as stack:
        tool.register(stack, context, middleware=[Deny(call, context)])
        async with context.stream.get(ToolResultEvent.parent_id == "tc_1") as pending:
            await context.send(call)
            result = await pending

    assert seen == ["bash"]
    assert result.result == ToolResult("denied by policy")
