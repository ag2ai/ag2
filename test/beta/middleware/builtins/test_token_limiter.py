# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

from autogen.beta import Context
from autogen.beta.events import (
    BaseEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolCalls,
    ToolResult,
    ToolResults,
)
from autogen.beta.middleware import TokenLimiter


@pytest.mark.asyncio()
async def test_token_limiter_passes_events_through_when_within_budget(mock: MagicMock) -> None:
    events = [
        ModelRequest(content="hello"),
        ModelResponse(message=ModelMessage(content="world")),
    ]
    middleware = TokenLimiter(max_tokens=10_000)(events[-1], mock)

    async def llm_call(history: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(history)
        return ModelResponse(message=ModelMessage(content="result"))

    await middleware.on_llm_call(llm_call, events, mock)

    mock.llm_call.assert_called_once_with(events)


@pytest.mark.asyncio()
async def test_token_limiter_keeps_first_request_while_trimming(mock: MagicMock) -> None:
    events = [
        ModelRequest(content="keep-me"),
        ModelResponse(message=ModelMessage(content="drop-me-1")),
        ModelResponse(message=ModelMessage(content="drop-me-2")),
        ModelResponse(message=ModelMessage(content="keep-me-too")),
    ]
    middleware = TokenLimiter(max_tokens=30, chars_per_token=1)(events[-1], mock)

    async def llm_call(history: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(history)
        return ModelResponse(message=ModelMessage(content="result"))

    await middleware.on_llm_call(llm_call, events, mock)

    mock.llm_call.assert_called_once_with([
        ModelRequest(content="keep-me"),
        ModelResponse(message=ModelMessage(content="keep-me-too")),
    ])


@pytest.mark.asyncio()
async def test_token_limiter_trims_from_front_without_initial_request(mock: MagicMock) -> None:
    events = [
        ModelResponse(message=ModelMessage(content="drop-me-1")),
        ModelResponse(message=ModelMessage(content="drop-me-2")),
        ModelResponse(message=ModelMessage(content="keep-me")),
    ]
    middleware = TokenLimiter(max_tokens=20, chars_per_token=1)(events[-1], mock)

    async def llm_call(history: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(history)
        return ModelResponse(message=ModelMessage(content="result"))

    await middleware.on_llm_call(llm_call, events, mock)

    mock.llm_call.assert_called_once_with([ModelResponse(message=ModelMessage(content="keep-me"))])


@pytest.mark.asyncio()
async def test_token_limiter_drops_tool_results_without_parent_message(mock: MagicMock) -> None:
    tool_call = ToolCall(id="tool-call-1", name="lookup", arguments="{}")
    events = [
        ModelRequest(content="turn 1"),
        ModelResponse(tool_calls=ToolCalls(calls=[tool_call])),
        ToolResults(results=[ToolResult(parent_id=tool_call.id, name=tool_call.name, raw_content="ok")]),
        ModelResponse(message=ModelMessage(content="answer 1")),
        ModelRequest(content="turn 2"),
    ]
    budget_after_dropping_tool_call = sum(len(str(event)) for event in [events[0], events[2], events[3], events[4]])
    middleware = TokenLimiter(max_tokens=budget_after_dropping_tool_call, chars_per_token=1)(events[-1], mock)

    async def llm_call(history: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(history)
        return ModelResponse(message=ModelMessage(content="result"))

    await middleware.on_llm_call(llm_call, events, mock)

    mock.llm_call.assert_called_once_with([
        ModelRequest(content="turn 1"),
        ModelResponse(message=ModelMessage(content="answer 1")),
        ModelRequest(content="turn 2"),
    ])


@pytest.mark.asyncio()
async def test_token_limiter_drops_tool_results_without_parent_message_and_no_initial_request(
    mock: MagicMock,
) -> None:
    tool_call = ToolCall(id="tool-call-1", name="lookup", arguments="{}")
    events = [
        ModelResponse(tool_calls=ToolCalls(calls=[tool_call])),
        ToolResults(results=[ToolResult(parent_id=tool_call.id, name=tool_call.name, raw_content="ok")]),
        ModelResponse(message=ModelMessage(content="answer 1")),
    ]
    budget_after_dropping_tool_call = sum(len(str(event)) for event in [events[1], events[2]])
    middleware = TokenLimiter(max_tokens=budget_after_dropping_tool_call, chars_per_token=1)(events[-1], mock)

    async def llm_call(history: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(history)
        return ModelResponse(message=ModelMessage(content="result"))

    await middleware.on_llm_call(llm_call, events, mock)

    mock.llm_call.assert_called_once_with([ModelResponse(message=ModelMessage(content="answer 1"))])


def test_token_limiter_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="max_tokens must be greater than 0"):
        TokenLimiter(max_tokens=0)

    with pytest.raises(ValueError, match="chars_per_token must be greater than 0"):
        TokenLimiter(max_tokens=1, chars_per_token=0)
