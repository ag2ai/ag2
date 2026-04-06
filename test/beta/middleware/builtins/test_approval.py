# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import AsyncMock

import pytest

from autogen.beta.events import ToolCallEvent, ToolResultEvent
from autogen.beta.middleware import approval_required
from autogen.beta.middleware.builtin.tools.approval import BYPASS_KEY


def make_context(
    response: str = "y",
    variables: dict[str, Any] | None = None,
) -> AsyncMock:
    context = AsyncMock()
    context.input = AsyncMock(return_value=response)
    context.variables = variables if variables is not None else {}
    return context


@pytest.fixture
def tool_call() -> ToolCallEvent:
    return ToolCallEvent(name="calculator", arguments='{"a": 1, "b": 2}')


@pytest.mark.asyncio()
@pytest.mark.parametrize("response", ["y", "Y", "yes", "Yes", "YES", "1"])
async def test_accepts_various_affirmative_inputs(tool_call: ToolCallEvent, response: str) -> None:
    hook = approval_required()
    context = make_context(response)

    expected = ToolResultEvent.from_call(tool_call, result="3")

    async def call_next(event: ToolCallEvent, ctx: object) -> ToolResultEvent:
        return expected

    result = await hook(call_next, tool_call, context)

    assert result == expected
    context.input.assert_awaited_once()


@pytest.mark.asyncio()
async def test_denies_on_no(tool_call: ToolCallEvent) -> None:
    hook = approval_required()
    context = make_context("n")

    call_next = AsyncMock()

    result = await hook(call_next, tool_call, context)

    call_next.assert_not_awaited()
    assert result == ToolResultEvent.from_call(tool_call, result="User denied the tool call request")


@pytest.mark.asyncio()
async def test_custom_message(tool_call: ToolCallEvent) -> None:
    custom_msg = "Approve {tool_name} with {tool_arguments}?"
    hook = approval_required(message=custom_msg)
    context = make_context("y")

    call_next = AsyncMock(return_value=ToolResultEvent.from_call(tool_call, result="ok"))

    await hook(call_next, tool_call, context)

    context.input.assert_awaited_once_with(
        'Approve calculator with {"a": 1, "b": 2}?',
        timeout=30,
    )


@pytest.mark.asyncio()
async def test_custom_timeout(tool_call: ToolCallEvent) -> None:
    hook = approval_required(timeout=60)
    context = make_context("y")

    call_next = AsyncMock(return_value=ToolResultEvent.from_call(tool_call, result="ok"))

    await hook(call_next, tool_call, context)

    _, kwargs = context.input.await_args
    assert kwargs["timeout"] == 60


@pytest.mark.asyncio()
async def test_custom_denied_message(tool_call: ToolCallEvent) -> None:
    hook = approval_required(denied_message="Rejected by user")
    context = make_context("no")

    call_next = AsyncMock(return_value=ToolResultEvent.from_call(tool_call, result="ok"))

    result = await hook(call_next, tool_call, context)

    assert result == ToolResultEvent.from_call(tool_call, result="Rejected by user")


@pytest.mark.asyncio()
async def test_always_sets_bypass_flag(tool_call: ToolCallEvent) -> None:
    hook = approval_required(allow_always=True)
    context = make_context("always")

    expected = ToolResultEvent.from_call(tool_call, result="ok")
    call_next = AsyncMock(return_value=expected)

    # first execution should prompt
    await hook(call_next, tool_call, context)
    context.input.assert_awaited_once()
    assert context.variables[BYPASS_KEY]["calculator"] is True

    # second execution should not prompt
    await hook(call_next, tool_call, context)
    context.input.assert_awaited_once()


@pytest.mark.asyncio()
async def test_always_is_per_tool(tool_call: ToolCallEvent) -> None:
    hook = approval_required(allow_always=True)
    context = make_context("y", variables={BYPASS_KEY: {"other_tool": True}})

    expected = ToolResultEvent.from_call(tool_call, result="ok")
    call_next = AsyncMock(return_value=expected)

    result = await hook(call_next, tool_call, context)

    assert result == expected
    # Should still prompt since "calculator" is not in the bypass dict
    context.input.assert_awaited_once()


@pytest.mark.asyncio()
async def test_always_ignored_when_disabled(tool_call: ToolCallEvent) -> None:
    hook = approval_required(allow_always=False)
    context = make_context("always")

    call_next = AsyncMock()

    result = await hook(call_next, tool_call, context)

    call_next.assert_not_awaited()
    assert result == ToolResultEvent.from_call(tool_call, result="User denied the tool call request")
