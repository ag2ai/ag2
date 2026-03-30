# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for assembly policies (replacement for old ContextHarness tests)."""

import pytest

from autogen.beta.context import Context
from autogen.beta.events import ModelMessage, ModelRequest, ModelResponse
from autogen.beta.events.tool_events import ToolCallEvent, ToolResultEvent
from autogen.beta.network.assembler import AssemblerMiddleware
from autogen.beta.network.events import DelegationResult, SchedulerTriggerFired, TopicMessage
from autogen.beta.network.hub import Hub
from autogen.beta.network.policies.conversation import ConversationPolicy
from autogen.beta.network.policies.episodic_memory import EpisodicMemoryPolicy
from autogen.beta.network.policies.network import FormattedEvent, NetworkPolicy
from autogen.beta.network.policies.sliding_window import SlidingWindowPolicy
from autogen.beta.network.policies.token_budget import TokenBudgetPolicy
from autogen.beta.network.policies.topic_inbox import TopicInboxPolicy, TopicOverflow
from autogen.beta.network.policies.working_memory import WorkingMemoryPolicy
from autogen.beta.network.primitives.compact import CompactionSummary
from autogen.beta.network.primitives.knowledge import KnowledgeStore, MemoryKnowledgeStore
from autogen.beta.network.primitives.signal import Severity, Signal
from autogen.beta.stream import MemoryStream


class TestConversationPolicy:
    @pytest.mark.asyncio
    async def test_filters_to_conversation_events(self) -> None:
        policy = ConversationPolicy()
        events = [
            ModelRequest(content="hello"),
            ModelResponse(message=ModelMessage(content="hi")),
            ToolCallEvent(name="search", arguments="{}"),
            ToolResultEvent(id="1", name="search", content="result"),
            Signal(source="mon", severity=Severity.WARNING, message="warn"),
        ]
        ctx = Context(stream=MemoryStream())
        prompts, filtered = await policy.apply([], events, ctx)
        assert len(filtered) == 4
        assert all(not isinstance(e, Signal) for e in filtered)

    @pytest.mark.asyncio
    async def test_includes_compaction_summary(self) -> None:
        policy = ConversationPolicy()
        summary = CompactionSummary(summary="Earlier context...", event_count=50)
        events = [summary, ModelRequest(content="hello")]
        ctx = Context(stream=MemoryStream())
        _, filtered = await policy.apply([], events, ctx)
        assert summary in filtered


class TestNetworkPolicy:
    @pytest.mark.asyncio
    async def test_includes_signals(self) -> None:
        policy = NetworkPolicy()
        signal = Signal(source="mon", severity=Severity.CRITICAL, message="alert")
        events = [ModelRequest(content="hello"), signal]
        ctx = Context(stream=MemoryStream())
        _, filtered = await policy.apply([], events, ctx)
        # Signal should be formatted as FormattedEvent
        formatted = [e for e in filtered if isinstance(e, FormattedEvent)]
        assert len(formatted) == 1
        assert "[SIGNAL/CRITICAL]" in formatted[0].content

    @pytest.mark.asyncio
    async def test_formats_delegation_result(self) -> None:
        policy = NetworkPolicy()
        dr = DelegationResult(source="researcher", target="writer", result="report written")
        events = [ModelRequest(content="hello"), dr]
        ctx = Context(stream=MemoryStream())
        _, filtered = await policy.apply([], events, ctx)
        formatted = [e for e in filtered if isinstance(e, FormattedEvent)]
        assert len(formatted) == 1
        assert "[DELEGATION RESULT]" in formatted[0].content
        assert "researcher" in formatted[0].content

    @pytest.mark.asyncio
    async def test_formats_scheduler_trigger(self) -> None:
        policy = NetworkPolicy()
        st = SchedulerTriggerFired(watch_id="w1", target="monitor", task="check")
        events = [ModelRequest(content="hello"), st]
        ctx = Context(stream=MemoryStream())
        _, filtered = await policy.apply([], events, ctx)
        formatted = [e for e in filtered if isinstance(e, FormattedEvent)]
        assert len(formatted) == 1
        assert "[SCHEDULED]" in formatted[0].content

    @pytest.mark.asyncio
    async def test_formats_topic_message(self) -> None:
        policy = NetworkPolicy()
        msg = TopicMessage(topic="findings", sender="researcher", message="Found X")
        events = [ModelRequest(content="hello"), msg]
        ctx = Context(stream=MemoryStream())
        _, filtered = await policy.apply([], events, ctx)
        formatted = [e for e in filtered if isinstance(e, FormattedEvent)]
        assert len(formatted) == 1
        assert "[TOPIC/findings]" in formatted[0].content

    @pytest.mark.asyncio
    async def test_does_not_format_conversation_events(self) -> None:
        policy = NetworkPolicy()
        req = ModelRequest(content="hello")
        events = [req]
        ctx = Context(stream=MemoryStream())
        _, filtered = await policy.apply([], events, ctx)
        assert filtered[0] is req  # Not wrapped in FormattedEvent


class TestSlidingWindowPolicy:
    @pytest.mark.asyncio
    async def test_no_trim_below_max(self) -> None:
        policy = SlidingWindowPolicy(max_events=10)
        events = [ModelRequest(content=f"msg-{i}") for i in range(5)]
        ctx = Context(stream=MemoryStream())
        prompts, filtered = await policy.apply([], events, ctx)
        assert len(filtered) == 5
        assert prompts == []

    @pytest.mark.asyncio
    async def test_trims_to_max(self) -> None:
        policy = SlidingWindowPolicy(max_events=3)
        events = [ModelRequest(content=f"msg-{i}") for i in range(10)]
        ctx = Context(stream=MemoryStream())
        _, filtered = await policy.apply([], events, ctx)
        assert len(filtered) == 3
        assert filtered[0].content == "msg-7"

    @pytest.mark.asyncio
    async def test_transparent_adds_note(self) -> None:
        policy = SlidingWindowPolicy(max_events=3, transparent=True)
        events = [ModelRequest(content=f"msg-{i}") for i in range(10)]
        ctx = Context(stream=MemoryStream())
        prompts, _ = await policy.apply([], events, ctx)
        assert len(prompts) == 1
        assert "3 of 10" in prompts[0]


class TestTokenBudgetPolicy:
    @pytest.mark.asyncio
    async def test_no_trim_within_budget(self) -> None:
        policy = TokenBudgetPolicy(max_tokens=10000)
        events = [ModelRequest(content="short")]
        ctx = Context(stream=MemoryStream())
        _, filtered = await policy.apply([], events, ctx)
        assert len(filtered) == 1

    @pytest.mark.asyncio
    async def test_trims_to_budget(self) -> None:
        policy = TokenBudgetPolicy(max_tokens=10, chars_per_token=1)
        events = [ModelRequest(content="a" * 20), ModelRequest(content="b" * 5)]
        ctx = Context(stream=MemoryStream())
        _, filtered = await policy.apply([], events, ctx)
        # Should keep at least the last event that fits
        assert len(filtered) >= 1
        assert filtered[-1].content == "b" * 5


class TestAssemblerMiddleware:
    @pytest.mark.asyncio
    async def test_applies_policies_in_order(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        initial_event = ModelRequest(content="start")

        policy = ConversationPolicy()
        mw = AssemblerMiddleware(initial_event, ctx, policies=[policy])

        all_events = [
            ModelRequest(content="hello"),
            ModelResponse(message=ModelMessage(content="hi")),
            Signal(source="mon", severity=Severity.WARNING, message="warn"),
        ]

        received_events = None

        async def mock_llm_call(events, context):
            nonlocal received_events
            received_events = list(events)
            return ModelResponse(message=ModelMessage(content="response"))

        await mw.on_llm_call(mock_llm_call, all_events, ctx)

        assert received_events is not None
        assert len(received_events) == 2
        assert all(not isinstance(e, Signal) for e in received_events)

    @pytest.mark.asyncio
    async def test_restores_prompts_after_call(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream, prompt=["original"])
        initial_event = ModelRequest(content="start")

        class _PromptAdder:
            name = "adder"
            async def apply(self, prompts, events, context):
                return prompts + ["injected"], events

        mw = AssemblerMiddleware(initial_event, ctx, policies=[_PromptAdder()])

        async def mock_llm_call(events, context):
            assert "injected" in context.prompt
            return ModelResponse(message=ModelMessage(content="ok"))

        await mw.on_llm_call(mock_llm_call, [ModelRequest(content="hi")], ctx)
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
        initial_event = ModelRequest(content="start")

        class _PromptAdder:
            name = "adder"
            async def apply(self, prompts, events, context):
                return prompts + ["injected"], events

        mw = AssemblerMiddleware(initial_event, ctx, policies=[_PromptAdder()])

        async def failing_llm_call(events, context):
            raise RuntimeError("LLM failed")

        with pytest.raises(RuntimeError):
            await mw.on_llm_call(failing_llm_call, [ModelRequest(content="hi")], ctx)

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

        prompts, events = await policy.apply([], [ModelRequest(content="hi")], ctx)
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

        prompts, _ = await policy.apply([], [ModelRequest(content="hi")], ctx)
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
        prompts, events = await policy.apply(["existing"], [ModelRequest(content="hi")], ctx)
        assert prompts == ["existing"]

    @pytest.mark.asyncio
    async def test_no_op_when_no_summaries(self) -> None:
        store = MemoryKnowledgeStore()
        policy = EpisodicMemoryPolicy()
        ctx = Context(stream=MemoryStream())
        ctx.dependencies[KnowledgeStore] = store
        prompts, _ = await policy.apply([], [ModelRequest(content="hi")], ctx)
        assert prompts == []


class TestWorkingMemoryPolicy:
    @pytest.mark.asyncio
    async def test_injects_working_memory(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/memory/working.md", "Current state: working on project X.")

        policy = WorkingMemoryPolicy()
        ctx = Context(stream=MemoryStream())
        ctx.dependencies[KnowledgeStore] = store

        prompts, _ = await policy.apply([], [ModelRequest(content="hi")], ctx)
        assert any("Working Memory" in p for p in prompts)
        assert any("project X" in p for p in prompts)

    @pytest.mark.asyncio
    async def test_no_op_without_store(self) -> None:
        policy = WorkingMemoryPolicy()
        ctx = Context(stream=MemoryStream())
        prompts, _ = await policy.apply([], [ModelRequest(content="hi")], ctx)
        assert prompts == []

    @pytest.mark.asyncio
    async def test_no_op_without_working_memory_file(self) -> None:
        store = MemoryKnowledgeStore()
        policy = WorkingMemoryPolicy()
        ctx = Context(stream=MemoryStream())
        ctx.dependencies[KnowledgeStore] = store
        prompts, _ = await policy.apply([], [ModelRequest(content="hi")], ctx)
        assert prompts == []


class TestTopicInboxPolicy:
    @pytest.mark.asyncio
    async def test_injects_topic_messages(self) -> None:
        hub = Hub()
        await hub.subscribe_topic("actor", "news")
        await hub.publish("writer", "news", "Breaking news!")

        policy = TopicInboxPolicy(hub, "actor")
        ctx = Context(stream=MemoryStream())
        prompts, _ = await policy.apply([], [], ctx)
        assert any("Breaking news!" in p for p in prompts)

    @pytest.mark.asyncio
    async def test_no_op_without_messages(self) -> None:
        hub = Hub()
        await hub.subscribe_topic("actor", "news")
        policy = TopicInboxPolicy(hub, "actor")
        ctx = Context(stream=MemoryStream())
        prompts, _ = await policy.apply(["existing"], [], ctx)
        assert prompts == ["existing"]

    @pytest.mark.asyncio
    async def test_newest_overflow_drops_oldest(self) -> None:
        hub = Hub()
        await hub.subscribe_topic("actor", "news")
        for i in range(10):
            await hub.publish("w", "news", f"msg-{i}")

        policy = TopicInboxPolicy(hub, "actor", max_messages=3, overflow=TopicOverflow.NEWEST)
        ctx = Context(stream=MemoryStream())
        prompts, _ = await policy.apply([], [], ctx)

        combined = " ".join(prompts)
        assert "msg-7" in combined
        assert "msg-8" in combined
        assert "msg-9" in combined
        assert "msg-0" not in combined

        # All messages consumed (cursor fully advanced)
        remaining = await hub.peek_topic("actor", "news")
        assert len(remaining) == 0

    @pytest.mark.asyncio
    async def test_oldest_overflow_preserves_newer_messages(self) -> None:
        """OLDEST overflow should keep old messages and defer newer ones.

        This is the fix for Bug C: previously the cursor advanced past all
        messages, permanently losing the deferred ones.
        """
        hub = Hub()
        await hub.subscribe_topic("actor", "news")
        for i in range(10):
            await hub.publish("w", "news", f"msg-{i}")

        policy = TopicInboxPolicy(hub, "actor", max_messages=3, overflow=TopicOverflow.OLDEST)
        ctx = Context(stream=MemoryStream())
        prompts, _ = await policy.apply([], [], ctx)

        combined = " ".join(prompts)
        # Should see oldest messages
        assert "msg-0" in combined
        assert "msg-1" in combined
        assert "msg-2" in combined
        assert "msg-9" not in combined

        # Newer messages should still be available (cursor only advanced by 3)
        remaining = await hub.peek_topic("actor", "news")
        assert len(remaining) == 7
        assert remaining[0].message == "msg-3"

    @pytest.mark.asyncio
    async def test_summary_overflow_advances_all(self) -> None:
        hub = Hub()
        await hub.subscribe_topic("actor", "news")
        for i in range(10):
            await hub.publish("w", "news", f"msg-{i}")

        # No summary_config — falls back to truncated list
        policy = TopicInboxPolicy(hub, "actor", max_messages=3, overflow=TopicOverflow.SUMMARY)
        ctx = Context(stream=MemoryStream())
        prompts, _ = await policy.apply([], [], ctx)

        assert any("summarized" in p.lower() for p in prompts)
        # All messages consumed
        remaining = await hub.peek_topic("actor", "news")
        assert len(remaining) == 0
