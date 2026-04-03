# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest

from autogen.beta.events import ToolCallEvent, ToolResultEvent
from autogen.beta.middleware import approval_required


@pytest.fixture
def tool_call() -> ToolCallEvent:
    return ToolCallEvent(name="calculator", arguments='{"a": 1, "b": 2}')


@pytest.mark.asyncio()
@pytest.mark.parametrize("response", ["y", "Y", "yes", "Yes", "YES", "1"])
async def test_accepts_various_affirmative_inputs(tool_call: ToolCallEvent, response: str) -> None:
    hook = approval_required()
    context = AsyncMock()
    context.input = AsyncMock(return_value=response)

    expected = ToolResultEvent.from_call(tool_call, result="3")

    async def call_next(event: ToolCallEvent, ctx: object) -> ToolResultEvent:
        return expected

    result = await hook(call_next, tool_call, context)

    assert result == expected
    context.input.assert_awaited_once()


@pytest.mark.asyncio()
async def test_denies_on_no(tool_call: ToolCallEvent) -> None:
    hook = approval_required()
    context = AsyncMock()
    context.input = AsyncMock(return_value="n")

    call_next = AsyncMock()

    result = await hook(call_next, tool_call, context)

    call_next.assert_not_awaited()
    assert result == ToolResultEvent.from_call(tool_call, result="User denied the tool call request")


@pytest.mark.asyncio()
async def test_custom_message(tool_call: ToolCallEvent) -> None:
    custom_msg = "Approve {tool_name} with {tool_arguments}?"
    hook = approval_required(message=custom_msg)
    context = AsyncMock()
    context.input = AsyncMock(return_value="y")

    call_next = AsyncMock(return_value=ToolResultEvent.from_call(tool_call, result="ok"))

    await hook(call_next, tool_call, context)

    context.input.assert_awaited_once_with(
        'Approve calculator with {"a": 1, "b": 2}?',
        timeout=30,
    )


@pytest.mark.asyncio()
async def test_custom_timeout(tool_call: ToolCallEvent) -> None:
    hook = approval_required(timeout=60)
    context = AsyncMock()
    context.input = AsyncMock(return_value="y")

    call_next = AsyncMock(return_value=ToolResultEvent.from_call(tool_call, result="ok"))

    await hook(call_next, tool_call, context)

    _, kwargs = context.input.await_args
    assert kwargs["timeout"] == 60


@pytest.mark.asyncio()
async def test_custom_denied_message(tool_call: ToolCallEvent) -> None:
    hook = approval_required(denied_message="Rejected by user")
    context = AsyncMock()
    context.input = AsyncMock(return_value="no")

    call_next = AsyncMock(return_value=ToolResultEvent.from_call(tool_call, result="ok"))

    result = await hook(call_next, tool_call, context)

    assert result == ToolResultEvent.from_call(tool_call, result="Rejected by user")
