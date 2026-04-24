# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
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
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from autogen.beta.middleware import HistoryLimiter


@pytest.mark.asyncio()
async def test_history_limiter(mock: MagicMock) -> None:
    history_limiter = HistoryLimiter(max_events=3)

    middleware = history_limiter(ModelRequest([TextInput("Hi!")]), mock)

    async def llm_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(events)
        return ModelResponse(ModelMessage("result"))

    await middleware.on_llm_call(llm_call, [ModelRequest([TextInput("Hi!")])], mock)

    mock.llm_call.assert_called_once_with([ModelRequest([TextInput("Hi!")])])


@pytest.mark.asyncio()
async def test_history_limiter_saves_first_turn(mock: MagicMock) -> None:
    history_limiter = HistoryLimiter(max_events=3)

    middleware = history_limiter(ModelRequest([TextInput("turn 3")]), mock)
    events = [
        ModelRequest([TextInput("turn 1")]),
        ModelResponse(ModelMessage("answer 1")),
        ModelRequest([TextInput("turn 2")]),
        ModelResponse(ModelMessage("answer 2")),
        ModelRequest([TextInput("turn 3")]),
    ]

    async def llm_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(events)
        return ModelResponse(ModelMessage("result"))

    await middleware.on_llm_call(llm_call, events, mock)

    mock.llm_call.assert_called_once_with([
        ModelRequest([TextInput("turn 1")]),
        ModelResponse(ModelMessage("answer 2")),
        ModelRequest([TextInput("turn 3")]),
    ])


@pytest.mark.asyncio()
async def test_no_history_limiter(mock: MagicMock) -> None:
    history_limiter = HistoryLimiter(max_events=1)

    middleware = history_limiter(ModelRequest([TextInput("turn 3")]), mock)
    events = [
        ModelRequest([TextInput("turn 1")]),
        ModelResponse(ModelMessage("answer 1")),
        ModelRequest([TextInput("turn 2")]),
        ModelResponse(ModelMessage("answer 2")),
        ModelRequest([TextInput("turn 3")]),
    ]

    async def llm_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(events)
        return ModelResponse(ModelMessage("result"))

    await middleware.on_llm_call(llm_call, events, mock)

    mock.llm_call.assert_called_once_with([ModelRequest([TextInput("turn 1")])])


@pytest.mark.asyncio()
async def test_history_limiter_drops_overlapping_turns(mock: MagicMock) -> None:
    history_limiter = HistoryLimiter(max_events=3)

    middleware = history_limiter(ModelRequest([TextInput("turn 3")]), mock)
    events = [
        ModelResponse(ModelMessage("answer 0")),
        ModelRequest([TextInput("turn 1")]),
        ModelResponse(ModelMessage("answer 1")),
        ModelRequest([TextInput("turn 2")]),
        ModelResponse(ModelMessage("answer 2")),
        ModelRequest([TextInput("turn 3")]),
    ]

    async def llm_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(events)
        return ModelResponse(ModelMessage("result"))

    await middleware.on_llm_call(llm_call, events, mock)

    mock.llm_call.assert_called_once_with([
        ModelRequest([TextInput("turn 2")]),
        ModelResponse(ModelMessage("answer 2")),
        ModelRequest([TextInput("turn 3")]),
    ])


@pytest.mark.asyncio()
async def test_history_limiter_drops_incomplete_tool_interaction(mock: MagicMock) -> None:
    history_limiter = HistoryLimiter(max_events=4)

    tool_call = ToolCallEvent(id="tool-call-1", name="lookup", arguments="{}")
    middleware = history_limiter(ModelRequest([TextInput("turn 2")]), mock)
    events = [
        ModelRequest([TextInput("turn 1")]),
        ModelResponse(tool_calls=ToolCallsEvent([tool_call])),
        ToolResultsEvent([ToolResultEvent.from_call(tool_call, result="ok")]),
        ModelResponse(ModelMessage("answer 1")),
        ModelRequest([TextInput("turn 2")]),
    ]

    async def llm_call(history: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(history)
        return ModelResponse(ModelMessage("result"))

    await middleware.on_llm_call(llm_call, events, mock)

    mock.llm_call.assert_called_once_with([
        ModelRequest([TextInput("turn 1")]),
        ModelResponse(ModelMessage("answer 1")),
        ModelRequest([TextInput("turn 2")]),
    ])
