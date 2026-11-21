# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Context, ToolResult
from ag2.events import (
    BaseEvent,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
    estimated_tokens,
)
from ag2.policies.token_budget import TokenBudgetPolicy


def _tokens(*events: BaseEvent) -> int:
    """Size of the given events in the tokens the policy counts."""
    return sum(estimated_tokens(e) for e in events)


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
        events: list[BaseEvent] = [ModelRequest([TextInput("hi")])]
        policy = TokenBudgetPolicy(max_tokens=100_000)

        prompts, result = await policy.apply([], events, context)

        assert result == events
        assert prompts == []


class TestTrimming:
    @pytest.mark.asyncio
    async def test_retains_most_recent_events(self, context: Context) -> None:
        # Use a very tight budget so only the last event fits
        last = ModelRequest([TextInput("z")])
        events: list[BaseEvent] = [ModelRequest([TextInput("a" * 200)]), last]
        budget_tokens = _tokens(last) + 1
        policy = TokenBudgetPolicy(max_tokens=budget_tokens)

        _, result = await policy.apply([], events, context)

        assert result[-1] == last

    @pytest.mark.asyncio
    async def test_transparent_adds_prompt(self, context: Context) -> None:
        events: list[BaseEvent] = [ModelRequest([TextInput("a" * 200)]), ModelRequest([TextInput("b")])]
        budget_tokens = _tokens(events[-1]) + 1
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
        policy = TokenBudgetPolicy(max_tokens=_tokens(tool_result, request))

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
        policy = TokenBudgetPolicy(max_tokens=_tokens(tr1, tr2, request))

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
        policy = TokenBudgetPolicy(max_tokens=_tokens(tr, request), transparent=True)

        prompts, result = await policy.apply([], events, context)

        # The prompt count should reflect that the orphan was removed
        assert str(len(result)) in prompts[-1]

    @pytest.mark.asyncio
    async def test_mid_window_orphaned_tool_result_is_dropped(self, context: Context) -> None:
        """An orphaned ToolResultsEvent must be dropped even when not at the head of the window."""
        tr = _tool_results("tc_1")
        req_a = ModelRequest([TextInput("a")])
        req_b = ModelRequest([TextInput("b")])
        events = [
            ModelRequest([TextInput("a" * 5000)]),
            _tool_response("tc_1"),
            req_a,
            tr,
            req_b,
        ]
        policy = TokenBudgetPolicy(max_tokens=_tokens(req_a, tr, req_b))

        _, result = await policy.apply([], events, context)

        assert result == [req_a, req_b]


class TestContentSizing:
    """The budget counts an event's full content, not its truncated repr."""

    @pytest.mark.asyncio
    async def test_long_message_is_trimmed(self, context: Context) -> None:
        # str(event) truncates a long message to a few hundred characters, so a
        # repr-sized budget would keep this whole history.
        big = ModelRequest([TextInput("x" * 10_000)])
        last = ModelRequest([TextInput("next")])
        policy = TokenBudgetPolicy(max_tokens=1000)

        _, result = await policy.apply([], [big, last], context)

        assert result == [last]

    @pytest.mark.asyncio
    async def test_long_tool_result_is_trimmed(self, context: Context) -> None:
        call = _tool_response("tc_1")
        big = ToolResultsEvent(
            results=[ToolResultEvent(parent_id="tc_1", name="get", result=ToolResult("y" * 10_000))],
        )
        last = ModelRequest([TextInput("next")])
        policy = TokenBudgetPolicy(max_tokens=1000)

        _, result = await policy.apply([], [call, big, last], context)

        assert result == [last]

    @pytest.mark.asyncio
    async def test_history_within_budget_is_unchanged(self, context: Context) -> None:
        events: list[BaseEvent] = [ModelRequest([TextInput("x" * 400)]), ModelRequest([TextInput("next")])]
        policy = TokenBudgetPolicy(max_tokens=_tokens(*events))

        _, result = await policy.apply([], events, context)

        assert result == events

    @pytest.mark.asyncio
    async def test_chars_per_token_scales_the_estimate(self, context: Context) -> None:
        events: list[BaseEvent] = [ModelRequest([TextInput("x" * 400)]), ModelRequest([TextInput("next")])]
        policy = TokenBudgetPolicy(max_tokens=150, chars_per_token=2)

        _, result = await policy.apply([], events, context)

        assert result == events[1:]

    @pytest.mark.parametrize("chars_per_token", [0, -1])
    def test_rejects_non_positive_chars_per_token(self, chars_per_token: int) -> None:
        with pytest.raises(ValueError, match="chars_per_token"):
            TokenBudgetPolicy(max_tokens=100, chars_per_token=chars_per_token)
