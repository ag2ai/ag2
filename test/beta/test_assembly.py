# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for assembly policies (replacement for old ContextHarness tests)."""

import pytest

from autogen.beta.assembly import AssemblerMiddleware
from autogen.beta.compact import CompactionSummary
from autogen.beta.context import ConversationContext as Context
from autogen.beta.events import ModelMessage, ModelRequest, ModelResponse, TextInput
from autogen.beta.events.alert import ObserverAlert, Severity
from autogen.beta.events.tool_events import ToolCallEvent, ToolResultEvent
from autogen.beta.knowledge import KnowledgeStore, MemoryKnowledgeStore
from autogen.beta.policies import (
    ConversationPolicy,
    EpisodicMemoryPolicy,
    SlidingWindowPolicy,
    TokenBudgetPolicy,
    WorkingMemoryPolicy,
)
from autogen.beta.stream import MemoryStream


class TestConversationPolicy:
    @pytest.mark.asyncio
    async def test_filters_to_conversation_events(self) -> None:
        policy = ConversationPolicy()
        events = [
            ModelRequest([TextInput("hello")]),
            ModelResponse(message=ModelMessage(content="hi")),
            ToolCallEvent(name="search", arguments="{}"),
            ToolResultEvent(id="1", name="search", content="result"),
            ObserverAlert(source="mon", severity=Severity.WARNING, message="warn"),
        ]
        ctx = Context(stream=MemoryStream())
        prompts, filtered = await policy.apply([], events, ctx)
        assert len(filtered) == 4
        assert all(not isinstance(e, ObserverAlert) for e in filtered)

    @pytest.mark.asyncio
    async def test_includes_compaction_summary(self) -> None:
        policy = ConversationPolicy()
        summary = CompactionSummary(summary="Earlier context...", event_count=50)
        events = [summary, ModelRequest([TextInput("hello")])]
        ctx = Context(stream=MemoryStream())
        _, filtered = await policy.apply([], events, ctx)
        assert summary in filtered


# NOTE: TestNetworkPolicy (V2 NetworkPolicy + FormattedEvent wrapping of
# DelegationResult / SchedulerTriggerFired / TopicMessage) was removed with
# the V2 rewrite. V3 will reintroduce equivalent assembly glue in Phase 2 /
# Phase 4 when network Tasks and the multi-participant session types land.


class TestSlidingWindowPolicy:
    @pytest.mark.asyncio
    async def test_no_trim_below_max(self) -> None:
        policy = SlidingWindowPolicy(max_events=10)
        events = [ModelRequest([TextInput(f"msg-{i}")]) for i in range(5)]
        ctx = Context(stream=MemoryStream())
        prompts, filtered = await policy.apply([], events, ctx)
        assert len(filtered) == 5
        assert prompts == []

    @pytest.mark.asyncio
    async def test_trims_to_max(self) -> None:
        policy = SlidingWindowPolicy(max_events=3)
        events = [ModelRequest([TextInput(f"msg-{i}")]) for i in range(10)]
        ctx = Context(stream=MemoryStream())
        _, filtered = await policy.apply([], events, ctx)
        assert len(filtered) == 3
        assert filtered[0].parts[0].content == "msg-7"

    @pytest.mark.asyncio
    async def test_transparent_adds_note(self) -> None:
        policy = SlidingWindowPolicy(max_events=3, transparent=True)
        events = [ModelRequest([TextInput(f"msg-{i}")]) for i in range(10)]
        ctx = Context(stream=MemoryStream())
        prompts, _ = await policy.apply([], events, ctx)
        assert len(prompts) == 1
        assert "3 of 10" in prompts[0]


class TestTokenBudgetPolicy:
    @pytest.mark.asyncio
    async def test_no_trim_within_budget(self) -> None:
        policy = TokenBudgetPolicy(max_tokens=10000)
        events = [ModelRequest([TextInput("short")])]
        ctx = Context(stream=MemoryStream())
        _, filtered = await policy.apply([], events, ctx)
        assert len(filtered) == 1

    @pytest.mark.asyncio
    async def test_trims_to_budget(self) -> None:
        policy = TokenBudgetPolicy(max_tokens=10, chars_per_token=1)
        events = [ModelRequest([TextInput("a" * 20)]), ModelRequest([TextInput("b" * 5)])]
        ctx = Context(stream=MemoryStream())
        _, filtered = await policy.apply([], events, ctx)
        # Should keep at least the last event that fits
        assert len(filtered) >= 1
        assert filtered[-1].parts[0].content == "b" * 5


class TestAssemblerMiddleware:
    @pytest.mark.asyncio
    async def test_applies_policies_in_order(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        initial_event = ModelRequest([TextInput("start")])

        policy = ConversationPolicy()
        mw = AssemblerMiddleware(initial_event, ctx, policies=[policy])

        all_events = [
            ModelRequest([TextInput("hello")]),
            ModelResponse(message=ModelMessage(content="hi")),
            ObserverAlert(source="mon", severity=Severity.WARNING, message="warn"),
        ]

        received_events = None

        async def mock_llm_call(events, context):
            nonlocal received_events
            received_events = list(events)
            return ModelResponse(message=ModelMessage(content="response"))

        await mw.on_llm_call(mock_llm_call, all_events, ctx)

        assert received_events is not None
        assert len(received_events) == 2
        assert all(not isinstance(e, ObserverAlert) for e in received_events)

    @pytest.mark.asyncio
    async def test_restores_prompts_after_call(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream, prompt=["original"])
        initial_event = ModelRequest([TextInput("start")])

        class _PromptAdder:
            name = "adder"

            async def apply(self, prompts, events, context):
                return prompts + ["injected"], events

        mw = AssemblerMiddleware(initial_event, ctx, policies=[_PromptAdder()])

        async def mock_llm_call(events, context):
            assert "injected" in context.prompt
            return ModelResponse(message=ModelMessage(content="ok"))

        await mw.on_llm_call(mock_llm_call, [ModelRequest([TextInput("hi")])], ctx)
        assert ctx.prompt == ["original"]

    def test_validate_order_warns_on_bad_ordering(self) -> None:
        class _FakePolicy:
            def __init__(self, n):
                self.name = n

            async def apply(self, p, e, c):
                return p, e

        policies = [_FakePolicy("sliding_window"), _FakePolicy("episodic_memory")]
        warnings = AssemblerMiddleware.validate_order(policies)
        assert len(warnings) == 1
        assert "sliding_window" in warnings[0]

    def test_validate_order_no_warnings_for_correct_order(self) -> None:
        class _FakePolicy:
            def __init__(self, n):
                self.name = n

            async def apply(self, p, e, c):
                return p, e

        policies = [_FakePolicy("episodic_memory"), _FakePolicy("sliding_window")]
        warnings = AssemblerMiddleware.validate_order(policies)
        assert len(warnings) == 0

    @pytest.mark.asyncio
    async def test_restores_prompts_on_exception(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream, prompt=["original"])
        initial_event = ModelRequest([TextInput("start")])

        class _PromptAdder:
            name = "adder"

            async def apply(self, prompts, events, context):
                return prompts + ["injected"], events

        mw = AssemblerMiddleware(initial_event, ctx, policies=[_PromptAdder()])

        async def failing_llm_call(events, context):
            raise RuntimeError("LLM failed")

        with pytest.raises(RuntimeError):
            await mw.on_llm_call(failing_llm_call, [ModelRequest([TextInput("hi")])], ctx)

        # Prompts must be restored even after exception
        assert ctx.prompt == ["original"]


class TestEpisodicMemoryPolicy:
    @pytest.mark.asyncio
    async def test_injects_summaries_from_store(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/memory/conversations/20260101T120000_abc.md", "Summary of session 1.")
        await store.write("/memory/conversations/20260102T120000_def.md", "Summary of session 2.")

        policy = EpisodicMemoryPolicy(max_episodes=5)
        ctx = Context(stream=MemoryStream())
        ctx.dependencies[KnowledgeStore] = store

        prompts, events = await policy.apply([], [ModelRequest([TextInput("hi")])], ctx)
        assert any("Past Conversations" in p for p in prompts)
        assert any("Summary of session 1" in p for p in prompts)
        assert any("Summary of session 2" in p for p in prompts)

    @pytest.mark.asyncio
    async def test_limits_to_max_episodes(self) -> None:
        store = MemoryKnowledgeStore()
        for i in range(10):
            await store.write(f"/memory/conversations/2026010{i}T120000_s{i}.md", f"Summary {i}")

        policy = EpisodicMemoryPolicy(max_episodes=3)
        ctx = Context(stream=MemoryStream())
        ctx.dependencies[KnowledgeStore] = store

        prompts, _ = await policy.apply([], [ModelRequest([TextInput("hi")])], ctx)
        # Should have the last 3 (most recent by sorted name)
        combined = " ".join(prompts)
        assert "Summary 7" in combined
        assert "Summary 8" in combined
        assert "Summary 9" in combined
        assert "Summary 0" not in combined

    @pytest.mark.asyncio
    async def test_no_op_without_store(self) -> None:
        policy = EpisodicMemoryPolicy()
        ctx = Context(stream=MemoryStream())
        prompts, events = await policy.apply(["existing"], [ModelRequest([TextInput("hi")])], ctx)
        assert prompts == ["existing"]

    @pytest.mark.asyncio
    async def test_no_op_when_no_summaries(self) -> None:
        store = MemoryKnowledgeStore()
        policy = EpisodicMemoryPolicy()
        ctx = Context(stream=MemoryStream())
        ctx.dependencies[KnowledgeStore] = store
        prompts, _ = await policy.apply([], [ModelRequest([TextInput("hi")])], ctx)
        assert prompts == []


class TestWorkingMemoryPolicy:
    @pytest.mark.asyncio
    async def test_injects_working_memory(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/memory/working.md", "Current state: working on project X.")

        policy = WorkingMemoryPolicy()
        ctx = Context(stream=MemoryStream())
        ctx.dependencies[KnowledgeStore] = store

        prompts, _ = await policy.apply([], [ModelRequest([TextInput("hi")])], ctx)
        assert any("Working Memory" in p for p in prompts)
        assert any("project X" in p for p in prompts)

    @pytest.mark.asyncio
    async def test_no_op_without_store(self) -> None:
        policy = WorkingMemoryPolicy()
        ctx = Context(stream=MemoryStream())
        prompts, _ = await policy.apply([], [ModelRequest([TextInput("hi")])], ctx)
        assert prompts == []

    @pytest.mark.asyncio
    async def test_no_op_without_working_memory_file(self) -> None:
        store = MemoryKnowledgeStore()
        policy = WorkingMemoryPolicy()
        ctx = Context(stream=MemoryStream())
        ctx.dependencies[KnowledgeStore] = store
        prompts, _ = await policy.apply([], [ModelRequest([TextInput("hi")])], ctx)
        assert prompts == []


# NOTE: TestTopicInboxPolicy removed — TopicInboxPolicy was V2-network-specific.
# Equivalent SessionInboxPolicy coverage lives with the V3 network tests.
