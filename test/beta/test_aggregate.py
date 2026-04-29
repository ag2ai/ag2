# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for AggregateStrategy, AggregateTrigger, and built-in strategies."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autogen.beta import Agent
from autogen.beta.agent import KnowledgeConfig
from autogen.beta.aggregate import (
    AggregateTrigger,
    ConversationSummaryAggregate,
    WorkingMemoryAggregate,
)
from autogen.beta.context import ConversationContext as Context
from autogen.beta.events import ModelMessage, ModelRequest, ModelResponse, TextInput
from autogen.beta.events.lifecycle import AggregationCompleted
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.stream import MemoryStream
from autogen.beta.testing import TestConfig


class TestAggregateTrigger:
    def test_defaults(self) -> None:
        trigger = AggregateTrigger()
        assert trigger.every_n_turns == 0
        assert trigger.every_n_events == 0
        assert trigger.on_end is False

    def test_custom_values(self) -> None:
        trigger = AggregateTrigger(every_n_turns=5, every_n_events=50, on_end=True)
        assert trigger.every_n_turns == 5
        assert trigger.every_n_events == 50
        assert trigger.on_end is True


class TestConversationSummaryAggregate:
    @pytest.mark.asyncio
    async def test_writes_timestamped_summary(self) -> None:
        """Summary files should be prefixed with a timestamp for chronological sorting."""
        mock_response = MagicMock()
        mock_response.content = "This conversation covered X and Y."
        mock_response.usage = {"input_tokens": 100, "output_tokens": 50}

        mock_client = AsyncMock(return_value=mock_response)
        mock_config = MagicMock()
        mock_config.create.return_value = mock_client

        strategy = ConversationSummaryAggregate(config=mock_config)
        store = MemoryKnowledgeStore()
        stream = MemoryStream()
        ctx = Context(stream=stream)
        events = [ModelRequest([TextInput("hello")]), ModelRequest([TextInput("world")])]

        await strategy.aggregate(events, ctx, store)

        entries = await store.list("/memory/conversations/")
        assert len(entries) == 1
        filename = entries[0]
        # Filename should start with a timestamp: YYYYMMDDTHHmmss_
        assert filename[8] == "T"  # ISO date separator
        assert filename[15] == "_"  # separator before stream ID
        assert filename.endswith(".md")

    @pytest.mark.asyncio
    async def test_skips_empty_events(self) -> None:
        mock_config = MagicMock()
        strategy = ConversationSummaryAggregate(config=mock_config)
        store = MemoryKnowledgeStore()
        ctx = Context(stream=MemoryStream())

        await strategy.aggregate([], ctx, store)

        entries = await store.list("/memory/conversations/")
        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_stores_usage(self) -> None:
        mock_response = MagicMock()
        mock_response.content = "Summary"
        mock_response.usage = {"input_tokens": 200}

        mock_client = AsyncMock(return_value=mock_response)
        mock_config = MagicMock()
        mock_config.create.return_value = mock_client

        strategy = ConversationSummaryAggregate(config=mock_config)
        store = MemoryKnowledgeStore()
        ctx = Context(stream=MemoryStream())

        await strategy.aggregate([ModelRequest([TextInput("hi")])], ctx, store)
        assert strategy.last_usage == {"input_tokens": 200}

    @pytest.mark.asyncio
    async def test_chronological_ordering_of_summaries(self) -> None:
        """Multiple summaries should sort chronologically by filename."""
        mock_response = MagicMock()
        mock_response.content = "Summary"
        mock_response.usage = {}

        mock_client = AsyncMock(return_value=mock_response)
        mock_config = MagicMock()
        mock_config.create.return_value = mock_client

        strategy = ConversationSummaryAggregate(config=mock_config)
        store = MemoryKnowledgeStore()

        # Write two summaries with different streams
        stream1 = MemoryStream()
        ctx1 = Context(stream=stream1)
        with patch("autogen.beta.aggregate.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20260101T120000"
            mock_dt.side_effect = None
            await strategy.aggregate([ModelRequest([TextInput("first")])], ctx1, store)

        stream2 = MemoryStream()
        ctx2 = Context(stream=stream2)
        with patch("autogen.beta.aggregate.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20260201T120000"
            mock_dt.side_effect = None
            await strategy.aggregate([ModelRequest([TextInput("second")])], ctx2, store)

        entries = await store.list("/memory/conversations/")
        assert len(entries) == 2
        # Alphabetical sort == chronological sort because of timestamp prefix
        assert entries[0] < entries[1]
        assert entries[0].startswith("20260101")
        assert entries[1].startswith("20260201")


class TestWorkingMemoryAggregate:
    @pytest.mark.asyncio
    async def test_writes_working_memory(self) -> None:
        mock_response = MagicMock()
        mock_response.content = "Updated working memory content."
        mock_response.usage = {"input_tokens": 100}

        mock_client = AsyncMock(return_value=mock_response)
        mock_config = MagicMock()
        mock_config.create.return_value = mock_client

        strategy = WorkingMemoryAggregate(config=mock_config)
        store = MemoryKnowledgeStore()
        ctx = Context(stream=MemoryStream())

        await strategy.aggregate([ModelRequest([TextInput("hi")])], ctx, store)

        content = await store.read("/memory/working.md")
        assert content == "Updated working memory content."

    @pytest.mark.asyncio
    async def test_merges_with_existing(self) -> None:
        """Should pass existing working memory to LLM for merging."""
        mock_response = MagicMock()
        mock_response.content = "Merged memory."
        mock_response.usage = {}

        mock_client = AsyncMock(return_value=mock_response)
        mock_config = MagicMock()
        mock_config.create.return_value = mock_client

        strategy = WorkingMemoryAggregate(config=mock_config)
        store = MemoryKnowledgeStore()
        await store.write("/memory/working.md", "Existing context about project X.")
        ctx = Context(stream=MemoryStream())

        await strategy.aggregate([ModelRequest([TextInput("update")])], ctx, store)

        # Verify the LLM was called with existing memory in the prompt
        call_args = mock_client.call_args
        prompt_event = call_args[0][0][0]
        assert any(
            isinstance(inp, TextInput) and "Existing context about project X." in inp.content
            for inp in prompt_event.parts
        )

        content = await store.read("/memory/working.md")
        assert content == "Merged memory."

    @pytest.mark.asyncio
    async def test_skips_empty_events(self) -> None:
        mock_config = MagicMock()
        strategy = WorkingMemoryAggregate(config=mock_config)
        store = MemoryKnowledgeStore()
        ctx = Context(stream=MemoryStream())

        await strategy.aggregate([], ctx, store)
        assert await store.read("/memory/working.md") is None

    @pytest.mark.asyncio
    async def test_falls_back_to_existing_on_empty_response(self) -> None:
        mock_response = MagicMock()
        mock_response.content = ""  # LLM returns empty
        mock_response.usage = {}

        mock_client = AsyncMock(return_value=mock_response)
        mock_config = MagicMock()
        mock_config.create.return_value = mock_client

        strategy = WorkingMemoryAggregate(config=mock_config)
        store = MemoryKnowledgeStore()
        await store.write("/memory/working.md", "Existing content.")
        ctx = Context(stream=MemoryStream())

        await strategy.aggregate([ModelRequest([TextInput("update")])], ctx, store)

        # Should fall back to existing content when LLM returns empty
        content = await store.read("/memory/working.md")
        assert content == "Existing content."


class TestConversationSummaryStreamId:
    @pytest.mark.asyncio
    async def test_filename_uses_full_stream_id(self) -> None:
        """The summary filename must contain the full stream UUID, not a prefix."""
        store = MemoryKnowledgeStore()
        stream = MemoryStream()
        ctx = Context(stream=stream)
        full_id = str(stream.id)

        mock_response = MagicMock()
        mock_response.content = "summary"
        mock_response.usage = {}
        mock_client = AsyncMock(return_value=mock_response)
        mock_config = MagicMock()
        mock_config.create.return_value = mock_client

        agg = ConversationSummaryAggregate(config=mock_config)
        await agg.aggregate([ModelRequest([TextInput("hello")])], ctx, store)

        entries = await store.list("/memory/conversations/")
        assert len(entries) == 1
        assert full_id in entries[0]


class _RecordingAggregate:
    """In-process AggregateStrategy that records calls without LLM traffic."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_usage: dict = {}

    async def aggregate(self, events, context, store) -> None:
        self.calls += 1
        await store.write(f"/memory/runs/{self.calls}.md", "rolled up")


class TestAggregationWiredOnAgent:
    """End-to-end behaviour of the aggregation middleware on an Agent."""

    @pytest.mark.asyncio
    async def test_on_end_fires_once_per_ask(self) -> None:
        store = MemoryKnowledgeStore()
        strategy = _RecordingAggregate()
        stream = MemoryStream()
        completions: list[AggregationCompleted] = []
        stream.where(AggregationCompleted).subscribe(lambda e: completions.append(e))

        agent = Agent(
            "roller",
            config=TestConfig(ModelResponse(ModelMessage("done"))),
            knowledge=KnowledgeConfig(
                store=store,
                aggregate=strategy,
                aggregate_trigger=AggregateTrigger(on_end=True),
            ),
        )
        await agent.ask("go", stream=stream)

        assert strategy.calls == 1
        assert len(completions) == 1
        assert completions[0].agent == "roller"

    @pytest.mark.asyncio
    async def test_on_end_does_not_double_fire_with_other_triggers(self) -> None:
        """``on_end=True`` plus ``every_n_turns=1`` still aggregates once per ask."""
        store = MemoryKnowledgeStore()
        strategy = _RecordingAggregate()
        agent = Agent(
            "once",
            config=TestConfig(ModelResponse(ModelMessage("ok"))),
            knowledge=KnowledgeConfig(
                store=store,
                aggregate=strategy,
                aggregate_trigger=AggregateTrigger(every_n_turns=1, on_end=True),
            ),
        )
        await agent.ask("go")

        assert strategy.calls == 1

    @pytest.mark.asyncio
    async def test_on_end_runs_even_when_turn_raises(self) -> None:
        """If the LLM call errors, ``on_end`` aggregation still fires.

        TestConfig with no canned responses raises on the first call,
        simulating a failed turn.
        """
        store = MemoryKnowledgeStore()
        strategy = _RecordingAggregate()
        agent = Agent(
            "boom",
            config=TestConfig(),
            knowledge=KnowledgeConfig(
                store=store,
                aggregate=strategy,
                aggregate_trigger=AggregateTrigger(on_end=True),
            ),
        )

        with pytest.raises(Exception):
            await agent.ask("go")

        assert strategy.calls == 1
