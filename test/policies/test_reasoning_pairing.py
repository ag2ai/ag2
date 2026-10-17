# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Trimming policies must not split a retained event from the provider-native
item it needs — a builtin tool call from its reasoning item, or a response
carrying tool calls from the turn object that is the only record of them."""

import pytest

from ag2 import Context, ToolResult
from ag2.events import (
    BaseEvent,
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    ModelMessage,
    ModelReasoning,
    ModelRequest,
    ModelResponse,
    ProviderReplay,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
    UsageEvent,
)
from ag2.policies.sliding_window import SlidingWindowPolicy
from ag2.policies.token_budget import TokenBudgetPolicy
from test._helpers import DurableReasoning, ProviderTurnState


def _call(call_id: str) -> BuiltinToolCallEvent:
    return BuiltinToolCallEvent(id=call_id, name="web_search", arguments="{}")


def _result(parent_id: str) -> BuiltinToolResultEvent:
    return BuiltinToolResultEvent(parent_id=parent_id, name="web_search", result=ToolResult("ok"))


def _chars(events: list[BaseEvent]) -> int:
    """Size of the given events in the characters the policy counts."""
    return sum(len(str(e)) for e in events)


def _budget_for(events: list[BaseEvent]) -> int:
    """Token budget that fits exactly the given events."""
    return _chars(events) // 4 + 1


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

    async def test_anchor_does_not_leak_past_a_response_boundary(self, context: Context) -> None:
        # The second group is self-contained: it has no reasoning item of its own,
        # so nothing about it was orphaned and the first group's anchor must not
        # condemn it. Live, the same model emits a reasoning item only when asked
        # for a summary, so one history really can mix anchored and unanchored
        # builtin calls.
        events = [
            DurableReasoning("plan one"),
            _call("ws_1"),
            _result("ws_1"),
            ModelRequest([TextInput("next")]),
            _call("ws_2"),
            _result("ws_2"),
        ]
        policy = SlidingWindowPolicy(max_events=2)

        _, result = await policy.apply([], events, context)

        assert result == events[4:]

    async def test_anchor_does_not_leak_past_a_model_response(self, context: Context) -> None:
        # A ModelResponse closes the response that emitted the anchor, so a builtin
        # call after it belongs to a later response and needs its own anchor.
        events = [
            DurableReasoning("plan one"),
            _call("ws_1"),
            ModelResponse(tool_calls=ToolCallsEvent(calls=[])),
            _call("ws_2"),
            _result("ws_2"),
        ]
        policy = SlidingWindowPolicy(max_events=2)

        _, result = await policy.apply([], events, context)

        assert result == events[3:]

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

    async def test_group_is_kept_whole_when_no_smaller_window_is_legal(self, context: Context) -> None:
        # Nothing follows the group, so advancing the cut would leave an empty
        # request. Overshooting the window beats sending nothing.
        events = [
            ModelRequest([TextInput("q")]),
            DurableReasoning("plan"),
            _call("ws_1"),
            _result("ws_1"),
        ]
        policy = SlidingWindowPolicy(max_events=2)

        _, result = await policy.apply([], events, context)

        assert result == events[1:]

    async def test_anchor_is_found_across_interleaved_events(self, context: Context) -> None:
        # UsageEvent is persisted but not conversation, so it can sit between a
        # reasoning item and the call it anchors. The anchor is still its anchor.
        events = [
            DurableReasoning("plan"),
            UsageEvent(),
            _call("ws_1"),
            _result("ws_1"),
            ModelRequest([TextInput("next")]),
        ]
        policy = SlidingWindowPolicy(max_events=3)

        _, result = await policy.apply([], events, context)

        assert result == [events[-1]]

    async def test_window_widens_past_stray_local_call_events(self, context: Context) -> None:
        # A local call is announced by the ModelResponse that requested it; the
        # standalone call events map to no provider item. A window holding only
        # those maps to an empty request, so it must widen to reach the response.
        call = ToolCallEvent(id="c_1", name="convert", arguments="{}")
        events = [
            ModelRequest([TextInput("q")]),
            ModelResponse(tool_calls=ToolCallsEvent(calls=[call])),
            ToolCallsEvent(calls=[call]),
            call,
            ToolResultsEvent(results=[ToolResultEvent(parent_id="c_1", name="convert", result=ToolResult("ok"))]),
        ]
        policy = SlidingWindowPolicy(max_events=3)

        _, result = await policy.apply([], events, context)

        assert result == events[1:]

    async def test_orphaned_builtin_result_is_dropped(self, context: Context) -> None:
        # A builtin call never appears in ModelResponse.tool_calls, so its result
        # needs the call event itself to survive the cut.
        events = [
            _call("ws_1"),
            _result("ws_1"),
            ModelRequest([TextInput("next")]),
        ]
        policy = SlidingWindowPolicy(max_events=2)

        _, result = await policy.apply([], events, context)

        assert result == [events[-1]]


@pytest.mark.asyncio
class TestProviderTurnItem:
    """A response's tool calls may live only in a provider-native turn object."""

    async def test_response_orphaned_from_its_turn_item_is_dropped(self, context: Context) -> None:
        # Keeping the response without the turn object rebuilds the turn text-only:
        # the tool calls vanish and the results below reference calls the model was
        # never told it made. Dropping the response takes the results with it.
        events = [
            ProviderTurnState(),
            ModelResponse(tool_calls=ToolCallsEvent(calls=[ToolCallEvent(id="tc_1", name="multiply")])),
            ToolResultsEvent(results=[ToolResultEvent(parent_id="tc_1", name="multiply", result=ToolResult("ok"))]),
            ModelRequest([TextInput("next")]),
        ]
        policy = SlidingWindowPolicy(max_events=3)

        _, result = await policy.apply([], events, context)

        assert result == [events[-1]]

    async def test_intact_turn_is_kept(self, context: Context) -> None:
        events = [
            ModelRequest([TextInput("old")]),
            ProviderTurnState(),
            ModelResponse(tool_calls=ToolCallsEvent(calls=[ToolCallEvent(id="tc_1", name="multiply")])),
            ToolResultsEvent(results=[ToolResultEvent(parent_id="tc_1", name="multiply", result=ToolResult("ok"))]),
        ]
        policy = SlidingWindowPolicy(max_events=3)

        _, result = await policy.apply([], events, context)

        assert result == events[1:]

    async def test_plain_response_survives_without_its_turn_item(self, context: Context) -> None:
        # Only tool calls live exclusively in the turn object; the text is on the
        # response, so an answer with no tool calls loses nothing.
        events = [
            ProviderTurnState(),
            ModelResponse(message=ModelMessage("hi")),
            ModelRequest([TextInput("next")]),
        ]
        policy = SlidingWindowPolicy(max_events=2)

        _, result = await policy.apply([], events, context)

        assert result == events[1:]

    async def test_turn_item_does_not_leak_to_a_later_response(self, context: Context) -> None:
        # A turn object belongs to the response that follows it. A later response
        # that never had one must not be condemned by the earlier one being cut.
        events = [
            ProviderTurnState(),
            ModelResponse(tool_calls=ToolCallsEvent(calls=[ToolCallEvent(id="tc_1", name="multiply")])),
            ModelRequest([TextInput("next")]),
            ModelResponse(tool_calls=ToolCallsEvent(calls=[ToolCallEvent(id="tc_2", name="multiply")])),
            ToolResultsEvent(results=[ToolResultEvent(parent_id="tc_2", name="multiply", result=ToolResult("ok"))]),
        ]
        policy = SlidingWindowPolicy(max_events=3)

        _, result = await policy.apply([], events, context)

        assert result == events[2:]


class ReasoningShapedTurnState(ModelReasoning, ProviderReplay):
    """A turn object that happens to subclass ``ModelReasoning``.

    Nothing stops a provider from shaping its turn carrier this way, and the two
    roles have opposite remedies — so the role has to be read off the event's own
    declaration rather than inferred from what it inherits.
    """

    __transient__ = False
    __replay_role__ = "turn"


def test_marker_requires_a_declared_role() -> None:
    with pytest.raises(TypeError, match="__replay_role__"):

        class Undeclared(BaseEvent, ProviderReplay):
            pass


@pytest.mark.asyncio
async def test_declared_role_decides_the_remedy_not_the_base_class(context: Context) -> None:
    # Read as an anchor, this event would condemn builtin calls and leave the
    # response standing — rebuilding the turn text-only and losing its tool calls.
    events = [
        ReasoningShapedTurnState("state"),
        ModelResponse(tool_calls=ToolCallsEvent(calls=[ToolCallEvent(id="tc_1", name="multiply")])),
        ToolResultsEvent(results=[ToolResultEvent(parent_id="tc_1", name="multiply", result=ToolResult("ok"))]),
        ModelRequest([TextInput("next")]),
    ]
    policy = SlidingWindowPolicy(max_events=3)

    _, result = await policy.apply([], events, context)

    assert result == [events[-1]]


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

    async def test_stays_within_budget_after_dropping_the_group(self, context: Context) -> None:
        events = [
            DurableReasoning("plan"),
            _call("ws_1"),
            _result("ws_1"),
            ModelRequest([TextInput("next")]),
        ]
        budget = _budget_for(events[1:])
        policy = TokenBudgetPolicy(max_tokens=budget)

        _, result = await policy.apply([], events, context)

        assert _chars(result) <= budget * 4

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

    async def test_budget_is_overshot_when_no_smaller_span_is_legal(self, context: Context) -> None:
        # The budget fits only the result, whose call is outside it. Dropping the
        # orphan would leave nothing, so the span widens past the budget instead —
        # an oversized request is recoverable, an empty one is rejected outright.
        events = [
            DurableReasoning("plan"),
            _call("ws_1"),
            _result("ws_1"),
        ]
        budget = _budget_for(events[-1:])
        policy = TokenBudgetPolicy(max_tokens=budget)

        _, result = await policy.apply([], events, context)

        assert result == events
        assert _chars(result) > budget * 4
