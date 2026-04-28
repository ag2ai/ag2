# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import ToolResult
from autogen.beta.context import ConversationContext as Context
from autogen.beta.events import (
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from autogen.beta.policies.sliding_window import SlidingWindowPolicy


def _tool_response(call_id: str = "tc_1", name: str = "get") -> ModelResponse:
    return ModelResponse(
        message=None,
        tool_calls=ToolCallsEvent(calls=[ToolCallEvent(id=call_id, name=name, arguments="{}")]),
    )


def _tool_results(parent_id: str = "tc_1", name: str = "get") -> ToolResultsEvent:
    return ToolResultsEvent(
        results=[ToolResultEvent(parent_id=parent_id, name=name, result=ToolResult("ok"))],
    )


class TestNoTrimming:
    @pytest.mark.asyncio
    async def test_events_within_limit_are_unchanged(self, context: Context) -> None:
        events = [ModelRequest([TextInput("a")]), ModelRequest([TextInput("b")])]
        policy = SlidingWindowPolicy(max_events=5)

        prompts, result = await policy.apply([], events, context)

        assert result == events
        assert prompts == []

    @pytest.mark.asyncio
    async def test_events_at_exact_limit(self, context: Context) -> None:
        events = [ModelRequest([TextInput("a")]), ModelRequest([TextInput("b")])]
        policy = SlidingWindowPolicy(max_events=2)

        _, result = await policy.apply([], events, context)

        assert result == events


class TestTrimming:
    @pytest.mark.asyncio
    async def test_keeps_last_n_events(self, context: Context) -> None:
        events = [ModelRequest([TextInput(str(i))]) for i in range(5)]
        policy = SlidingWindowPolicy(max_events=2)

        _, result = await policy.apply([], events, context)

        assert len(result) == 2
        assert result[0].parts[0].content == "3"
        assert result[1].parts[0].content == "4"

    @pytest.mark.asyncio
    async def test_transparent_adds_prompt(self, context: Context) -> None:
        events = [ModelRequest([TextInput(str(i))]) for i in range(5)]
        policy = SlidingWindowPolicy(max_events=2, transparent=True)

        prompts, result = await policy.apply(["existing"], events, context)

        assert len(result) == 2
        assert len(prompts) == 2
        assert prompts[0] == "existing"
        assert "2" in prompts[1] and "5" in prompts[1]


class TestOrphanedToolResults:
    """After trimming, leading ToolResultsEvents without a matching tool_use should be dropped."""

    @pytest.mark.asyncio
    async def test_leading_orphaned_tool_result_is_skipped(self, context: Context) -> None:
        events = [
            _tool_response("tc_1"),  # will be trimmed
            _tool_results("tc_1"),  # orphaned after trim — should be skipped
            ModelRequest([TextInput("next")]),
            ModelRequest([TextInput("final")]),
        ]
        policy = SlidingWindowPolicy(max_events=3)

        _, result = await policy.apply([], events, context)

        # The ToolResultsEvent should be dropped, leaving 2 events
        assert len(result) == 2
        assert isinstance(result[0], ModelRequest)
        assert result[0].parts[0].content == "next"

    @pytest.mark.asyncio
    async def test_multiple_leading_orphaned_tool_results_are_skipped(self, context: Context) -> None:
        events = [
            _tool_response("tc_1"),
            _tool_response("tc_2"),
            _tool_results("tc_1"),
            _tool_results("tc_2"),
            ModelRequest([TextInput("hello")]),
        ]
        policy = SlidingWindowPolicy(max_events=3)

        _, result = await policy.apply([], events, context)

        # Both orphaned ToolResultsEvents should be dropped
        assert len(result) == 1
        assert isinstance(result[0], ModelRequest)

    @pytest.mark.asyncio
    async def test_non_leading_tool_result_is_kept(self, context: Context) -> None:
        """ToolResultsEvent after a non-ToolResultsEvent should be preserved."""
        events = [
            ModelRequest([TextInput("old")]),
            _tool_response("tc_1"),
            _tool_results("tc_1"),
            ModelRequest([TextInput("new")]),
        ]
        policy = SlidingWindowPolicy(max_events=3)

        _, result = await policy.apply([], events, context)

        # trim keeps last 3: [tool_response, tool_results, request]
        assert len(result) == 3
        assert isinstance(result[0], ModelResponse)
        assert isinstance(result[1], ToolResultsEvent)
        assert isinstance(result[2], ModelRequest)

    @pytest.mark.asyncio
    async def test_transparent_count_reflects_skipped_orphans(self, context: Context) -> None:
        events = [
            _tool_response("tc_1"),
            _tool_results("tc_1"),
            ModelRequest([TextInput("a")]),
            ModelRequest([TextInput("b")]),
        ]
        policy = SlidingWindowPolicy(max_events=3, transparent=True)

        prompts, result = await policy.apply([], events, context)

        assert len(result) == 2
        # Prompt should reflect actual count after orphan removal
        assert "2" in prompts[-1] and "4" in prompts[-1]
