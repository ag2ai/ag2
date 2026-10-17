# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

from ag2 import Context
from ag2.events import (
    BaseEvent,
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResult,
    ToolResultEvent,
    ToolResultsEvent,
)
from ag2.middleware import HistoryLimiter
from test._helpers import DurableReasoning


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


@pytest.mark.asyncio()
async def test_history_limiter_never_orphans_a_builtin_tool_call(mock: MagicMock) -> None:
    """A builtin call replayed without its reasoning item is a hard provider error.

    Verified live against ``gpt-5.6-terra`` + ``WebSearchTool``: the tail this
    middleware used to produce is rejected with ``400 invalid_request_error —
    Item 'ws_…' of type 'web_search_call' was provided without its required
    'reasoning' item: 'rs_…'``. Skipping leading ``ToolResultsEvent``s did not
    cover it; the call event is neither a tool result nor named in
    ``ModelResponse.tool_calls``.
    """
    events = [
        ModelRequest([TextInput("q")]),
        DurableReasoning("plan"),
        BuiltinToolCallEvent(id="ws_1", name="web_search", arguments="{}"),
        BuiltinToolResultEvent(parent_id="ws_1", name="web_search", result=ToolResult("ok")),
        ModelResponse(ModelMessage("answer")),
        ModelRequest([TextInput("next")]),
    ]
    middleware = HistoryLimiter(max_events=5)(events[-1], mock)

    async def llm_call(history: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(history)
        return ModelResponse(ModelMessage("result"))

    await middleware.on_llm_call(llm_call, events, mock)

    # The whole group goes with its anchor: no web_search_call is left behind.
    mock.llm_call.assert_called_once_with([events[0], events[4], events[5]])


@pytest.mark.asyncio()
async def test_history_limiter_keeps_an_intact_builtin_group(mock: MagicMock) -> None:
    events = [
        ModelRequest([TextInput("q")]),
        ModelResponse(ModelMessage("old")),
        DurableReasoning("plan"),
        BuiltinToolCallEvent(id="ws_1", name="web_search", arguments="{}"),
        BuiltinToolResultEvent(parent_id="ws_1", name="web_search", result=ToolResult("ok")),
    ]
    middleware = HistoryLimiter(max_events=4)(events[-1], mock)

    async def llm_call(history: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(history)
        return ModelResponse(ModelMessage("result"))

    await middleware.on_llm_call(llm_call, events, mock)

    mock.llm_call.assert_called_once_with([events[0], *events[2:]])
