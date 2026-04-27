# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Any

import pytest
from anthropic.types import (
    BashCodeExecutionToolResultBlock,
    Message,
    ServerToolUseBlock,
    TextBlock,
    ToolUseBlock,
    Usage,
    WebSearchToolResultBlock,
)
from anthropic.types.bash_code_execution_tool_result_error import BashCodeExecutionToolResultError

from autogen.beta import MemoryStream
from autogen.beta.config.anthropic import AnthropicClient
from autogen.beta.config.anthropic.events import AnthropicServerToolCallEvent, AnthropicServerToolResultEvent
from autogen.beta.context import ConversationContext
from autogen.beta.events import BaseEvent, ModelMessage, ModelResponse, ToolCallEvent, ToolCallsEvent
from autogen.beta.events.tool_events import ToolResult
from autogen.beta.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME
from autogen.beta.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME


async def _process(content: Iterable[Any]) -> tuple[ModelResponse, list[BaseEvent]]:
    client = AnthropicClient(api_key="test", prompt_caching=False)
    message = Message.model_construct(
        id="m1",
        type="message",
        role="assistant",
        model="claude-sonnet-4-6",
        content=list(content),
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )
    stream = MemoryStream()
    context = ConversationContext(stream=stream)
    response = await client._process_response(message, context)
    return response, list(await stream.history.get_events())


@pytest.mark.asyncio
async def test_process_response_routes_all_block_types() -> None:
    web_call = ServerToolUseBlock(id="w1", name="web_search", input={"q": "x"}, type="server_tool_use")
    web_result = WebSearchToolResultBlock(tool_use_id="w1", type="web_search_tool_result", content=[])
    bash_call = ServerToolUseBlock(id="b1", name="bash_code_execution", input={"cmd": "ls"}, type="server_tool_use")
    bash_result = BashCodeExecutionToolResultBlock(
        tool_use_id="b1",
        type="bash_code_execution_tool_result",
        content=BashCodeExecutionToolResultError(
            error_code="unavailable",
            type="bash_code_execution_tool_result_error",
        ),
    )
    user_tool = ToolUseBlock(id="tc_1", name="my_func", input={"x": 1}, type="tool_use")

    response, events = await _process([
        TextBlock(text="Searching...", type="text"),
        web_call,
        web_result,
        bash_call,
        bash_result,
        user_tool,
    ])

    assert response.message == ModelMessage("Searching...")
    assert response.tool_calls == ToolCallsEvent([
        ToolCallEvent(id="tc_1", name="my_func", arguments='{"x": 1}'),
    ])
    assert events == [
        AnthropicServerToolCallEvent(id="w1", name=WEB_SEARCH_TOOL_NAME, arguments='{"q": "x"}', block=web_call),
        AnthropicServerToolResultEvent(
            parent_id="w1", name=WEB_SEARCH_TOOL_NAME, result=ToolResult(), block=web_result
        ),
        AnthropicServerToolCallEvent(
            id="b1", name=CODE_EXECUTION_TOOL_NAME, arguments='{"cmd": "ls"}', block=bash_call
        ),
        AnthropicServerToolResultEvent(
            parent_id="b1", name=CODE_EXECUTION_TOOL_NAME, result=ToolResult(), block=bash_result
        ),
    ]
