# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Agent, Context, MemoryStream, ToolResult
from ag2.events import (
    ImageInput,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
    estimated_tokens,
)
from ag2.policies import ConversationPolicy, TokenBudgetPolicy
from ag2.testing import TestConfig, TrackingConfig


def _tool_response(call_id: str = "tc_1", name: str = "get") -> ModelResponse:
    return ModelResponse(
        message=None,
        tool_calls=ToolCallsEvent(calls=[ToolCallEvent(id=call_id, name=name, arguments="{}")]),
    )


def _tool_results(parent_id: str = "tc_1", name: str = "get") -> ToolResultsEvent:
    return ToolResultsEvent(
        results=[ToolResultEvent(parent_id=parent_id, name=name, result=ToolResult("ok"))],
    )


@pytest.mark.asyncio
class TestContentBudget:
    @pytest.mark.parametrize("chars_per_token", [1, 4])
    async def test_long_user_content_is_counted_in_full(self, chars_per_token: int) -> None:
        latest = ModelRequest([TextInput("next")])
        events = [ModelRequest([TextInput("x" * 10_000)]), latest]
        policy = TokenBudgetPolicy(max_tokens=1000, chars_per_token=chars_per_token)

        _, result = await policy.apply([], events, Context(stream=MemoryStream()))

        assert result == [latest]

    async def test_long_tool_result_is_counted_in_full(self) -> None:
        latest = ModelRequest([TextInput("next")])
        events = [
            _tool_response(),
            ToolResultsEvent([ToolResultEvent(parent_id="tc_1", name="get", result=ToolResult("x" * 10_000))]),
            latest,
        ]
        policy = TokenBudgetPolicy(max_tokens=1000)

        _, result = await policy.apply([], events, Context(stream=MemoryStream()))

        assert result == [latest]

    async def test_image_uses_a_modality_budget(self) -> None:
        latest = ModelRequest([TextInput("next")])
        events = [ModelRequest([ImageInput(data=b"\x89PNG0000", media_type="image/png")]), latest]
        policy = TokenBudgetPolicy(max_tokens=500)

        _, result = await policy.apply([], events, Context(stream=MemoryStream()))

        assert result == [latest]

    @pytest.mark.parametrize("max_tokens", [0, 1000])
    async def test_oversized_latest_event_is_preserved(self, max_tokens: int) -> None:
        latest = ModelRequest([TextInput("x" * 10_000)])
        events = [ModelRequest([TextInput("older")]), latest]
        policy = TokenBudgetPolicy(max_tokens=max_tokens)

        _, result = await policy.apply([], events, Context(stream=MemoryStream()))

        assert result == [latest]

    async def test_short_content_does_not_pay_for_display_scaffolding(self) -> None:
        events = [ModelRequest([TextInput("abcd")]), ModelRequest([TextInput("efgh")])]
        policy = TokenBudgetPolicy(max_tokens=2, transparent=True)

        prompts, result = await policy.apply(["original"], events, Context(stream=MemoryStream()))

        assert result == events
        assert prompts == ["original"]

    async def test_agent_assembly_trims_long_history_before_the_model_call(self) -> None:
        stream = MemoryStream()
        await stream.history.replace([
            ModelRequest([TextInput("x" * 10_000)]),
            ModelResponse(ModelMessage("acknowledged")),
        ])
        config = TrackingConfig(TestConfig("done"))
        agent = Agent(
            "budgeted",
            config=config,
            assembly=[ConversationPolicy(), TokenBudgetPolicy(1000, transparent=True)],
        )

        reply = await agent.ask("next", stream=stream)

        assert reply.body == "done"
        assert config.calls[-1].prompt == ("[token_budget] Showing 2 of 3 events (token budget).",)


@pytest.mark.parametrize("chars_per_token", [0, -1])
def test_rejects_nonpositive_chars_per_token(chars_per_token: int) -> None:
    with pytest.raises(ValueError, match="chars_per_token must be greater than 0"):
        TokenBudgetPolicy(max_tokens=1000, chars_per_token=chars_per_token)


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
        budget_tokens = estimated_tokens(last)
        policy = TokenBudgetPolicy(max_tokens=budget_tokens)

        _, result = await policy.apply([], events, context)

        assert result[-1].parts[0].content == "z"

    @pytest.mark.asyncio
    async def test_transparent_adds_prompt(self, context: Context) -> None:
        events = [ModelRequest([TextInput("a" * 200)]), ModelRequest([TextInput("b")])]
        budget_tokens = estimated_tokens(events[-1])
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
        budget_tokens = estimated_tokens(tool_result) + estimated_tokens(request)
        policy = TokenBudgetPolicy(max_tokens=budget_tokens)

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
        budget_tokens = sum(estimated_tokens(event) for event in [tr1, tr2, request])
        policy = TokenBudgetPolicy(max_tokens=budget_tokens)

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
        budget_tokens = estimated_tokens(tr) + estimated_tokens(request)
        policy = TokenBudgetPolicy(max_tokens=budget_tokens, transparent=True)

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
        budget_tokens = sum(estimated_tokens(event) for event in [req_a, tr, req_b])
        policy = TokenBudgetPolicy(max_tokens=budget_tokens)

        _, result = await policy.apply([], events, context)

        assert all(not isinstance(e, ToolResultsEvent) for e in result)
        assert [e.parts[0].content for e in result if isinstance(e, ModelRequest)] == ["a", "b"]
