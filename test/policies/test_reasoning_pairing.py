# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Trimming policies must not split a builtin tool call from its reasoning item."""

import pytest

from ag2 import Context, ToolResult
from ag2.events import (
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    ModelReasoning,
    ModelRequest,
    TextInput,
)
from ag2.policies.sliding_window import SlidingWindowPolicy
from ag2.policies.token_budget import TokenBudgetPolicy


class DurableReasoning(ModelReasoning):
    """Provider reasoning item that must be replayed, like OpenAIReasoningEvent."""

    __transient__ = False


def _call(call_id: str) -> BuiltinToolCallEvent:
    return BuiltinToolCallEvent(id=call_id, name="web_search", arguments="{}")


def _result(parent_id: str) -> BuiltinToolResultEvent:
    return BuiltinToolResultEvent(parent_id=parent_id, name="web_search", result=ToolResult("ok"))


def _budget_for(events: list) -> int:
    """Token budget that fits exactly the given events."""
    return sum(len(str(e)) for e in events) // 4 + 1


@pytest.mark.asyncio
class TestSlidingWindow:
    async def test_orphaned_builtin_call_is_dropped(self, context: Context) -> None:
        events = [
            DurableReasoning("plan"),
            _call("ws_1"),
            _result("ws_1"),
            ModelRequest([TextInput("next")]),
        ]
        policy = SlidingWindowPolicy(max_events=3)

        _, result = await policy.apply([], events, context)

        assert result == [events[-1]]

    async def test_intact_group_is_kept(self, context: Context) -> None:
        events = [
            ModelRequest([TextInput("old")]),
            DurableReasoning("plan"),
            _call("ws_1"),
            _result("ws_1"),
        ]
        policy = SlidingWindowPolicy(max_events=3)

        _, result = await policy.apply([], events, context)

        assert result == events[1:]

    async def test_builtin_calls_without_reasoning_are_kept(self, context: Context) -> None:
        # Non-reasoning models emit no reasoning item, so nothing can be orphaned.
        events = [
            ModelRequest([TextInput("old")]),
            _call("ws_1"),
            _result("ws_1"),
            ModelRequest([TextInput("next")]),
        ]
        policy = SlidingWindowPolicy(max_events=3)

        _, result = await policy.apply([], events, context)

        assert result == events[1:]

    async def test_later_group_keeps_its_own_anchor(self, context: Context) -> None:
        # The cut splits group one; group two carries its own reasoning item.
        events = [
            DurableReasoning("plan one"),
            _call("ws_1"),
            DurableReasoning("plan two"),
            _call("ws_2"),
        ]
        policy = SlidingWindowPolicy(max_events=3)

        _, result = await policy.apply([], events, context)

        assert result == events[2:]

    async def test_transparent_count_reflects_dropped_group(self, context: Context) -> None:
        events = [
            DurableReasoning("plan"),
            _call("ws_1"),
            _result("ws_1"),
            ModelRequest([TextInput("next")]),
        ]
        policy = SlidingWindowPolicy(max_events=3, transparent=True)

        prompts, result = await policy.apply([], events, context)

        assert len(result) == 1
        assert "last 1 of 4" in prompts[-1]


@pytest.mark.asyncio
class TestTokenBudget:
    async def test_orphaned_builtin_call_is_dropped(self, context: Context) -> None:
        events = [
            DurableReasoning("plan"),
            _call("ws_1"),
            _result("ws_1"),
            ModelRequest([TextInput("next")]),
        ]
        policy = TokenBudgetPolicy(max_tokens=_budget_for(events[1:]))

        _, result = await policy.apply([], events, context)

        assert result == [events[-1]]

    async def test_stays_within_budget_after_advancing_the_cut(self, context: Context) -> None:
        events = [
            DurableReasoning("plan"),
            _call("ws_1"),
            _result("ws_1"),
            ModelRequest([TextInput("next")]),
        ]
        budget_chars = _budget_for(events[1:]) * 4
        policy = TokenBudgetPolicy(max_tokens=_budget_for(events[1:]))

        _, result = await policy.apply([], events, context)

        assert sum(len(str(e)) for e in result) <= budget_chars

    async def test_intact_group_is_kept(self, context: Context) -> None:
        events = [
            ModelRequest([TextInput("a" * 5000)]),
            DurableReasoning("plan"),
            _call("ws_1"),
            _result("ws_1"),
        ]
        policy = TokenBudgetPolicy(max_tokens=_budget_for(events[1:]))

        _, result = await policy.apply([], events, context)

        assert result == events[1:]
