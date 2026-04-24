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
from autogen.beta.policies.token_budget import TokenBudgetPolicy


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
    async def test_events_within_budget_are_unchanged(self, context: Context) -> None:
        events = [ModelRequest([TextInput("hi")])]
        policy = TokenBudgetPolicy(max_tokens=100_000)

        prompts, result = await policy.apply([], events, context)

        assert result == events
        assert prompts == []


class TestTrimming:
    @pytest.mark.asyncio
    async def test_retains_most_recent_events(self, context: Context) -> None:
        # Use a very tight budget so only the last event fits
        last = ModelRequest([TextInput("z")])
        events = [ModelRequest([TextInput("a" * 200)]), last]
        budget_tokens = (len(str(last)) // 4) + 1
        policy = TokenBudgetPolicy(max_tokens=budget_tokens)

        _, result = await policy.apply([], events, context)

        assert result[-1].parts[0].content == "z"

    @pytest.mark.asyncio
    async def test_transparent_adds_prompt(self, context: Context) -> None:
        events = [ModelRequest([TextInput("a" * 200)]), ModelRequest([TextInput("b")])]
        budget_tokens = (len(str(events[-1])) // 4) + 1
        policy = TokenBudgetPolicy(max_tokens=budget_tokens, transparent=True)

        prompts, _ = await policy.apply(["existing"], events, context)

        assert len(prompts) == 2
        assert prompts[0] == "existing"
        assert "token budget" in prompts[1]


class TestOrphanedToolResults:
    """After budget trimming, leading ToolResultsEvents without a matching tool_use should be dropped."""

    @pytest.mark.asyncio
    async def test_leading_orphaned_tool_result_is_skipped(self, context: Context) -> None:
        tool_result = _tool_results("tc_1")
        request = ModelRequest([TextInput("next")])
        # Budget enough for tool_result + request but NOT for the preceding tool_response
        events = [
            ModelRequest([TextInput("a" * 5000)]),
            _tool_response("tc_1"),
            tool_result,
            request,
        ]
        # Tight budget: fits tool_result + request but not the big first event and tool_response
        budget_chars = len(str(tool_result)) + len(str(request)) + 10
        policy = TokenBudgetPolicy(max_tokens=budget_chars // 4 + 1)

        _, result = await policy.apply([], events, context)

        # The orphaned leading ToolResultsEvent should be dropped
        for event in result:
            if isinstance(event, ToolResultsEvent):
                # If a ToolResultsEvent survived, it must not be the first element
                assert result[0] is not event

    @pytest.mark.asyncio
    async def test_multiple_leading_orphaned_tool_results_are_skipped(self, context: Context) -> None:
        tr1 = _tool_results("tc_1")
        tr2 = _tool_results("tc_2")
        request = ModelRequest([TextInput("hello")])
        events = [
            ModelRequest([TextInput("a" * 5000)]),
            _tool_response("tc_1"),
            _tool_response("tc_2"),
            tr1,
            tr2,
            request,
        ]
        budget_chars = len(str(tr1)) + len(str(tr2)) + len(str(request)) + 10
        policy = TokenBudgetPolicy(max_tokens=budget_chars // 4 + 1)

        _, result = await policy.apply([], events, context)

        # Both orphaned ToolResultsEvents should be dropped
        assert not isinstance(result[0], ToolResultsEvent)

    @pytest.mark.asyncio
    async def test_transparent_count_reflects_skipped_orphans(self, context: Context) -> None:
        tr = _tool_results("tc_1")
        request = ModelRequest([TextInput("b")])
        events = [
            ModelRequest([TextInput("a" * 5000)]),
            _tool_response("tc_1"),
            tr,
            request,
        ]
        budget_chars = len(str(tr)) + len(str(request)) + 10
        policy = TokenBudgetPolicy(max_tokens=budget_chars // 4 + 1, transparent=True)

        prompts, result = await policy.apply([], events, context)

        # The prompt count should reflect that the orphan was removed
        assert str(len(result)) in prompts[-1]
