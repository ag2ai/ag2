# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for CompactStrategy, CompactTrigger, and built-in strategies."""

import pytest

from autogen.beta import Agent, Context
from autogen.beta.agent import KnowledgeConfig
from autogen.beta.compact import CompactTrigger, CompactionSummary, TailWindowCompact
from autogen.beta.events import (
    CompactionCompleted,
    CompactionFailed,
    CompactionStarted,
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
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.stream import MemoryStream
from autogen.beta.testing import TestConfig


class TestTailWindowCompact:
    @pytest.mark.asyncio
    async def test_no_op_below_target(self) -> None:
        compact = TailWindowCompact(target=10)
        events = [ModelRequest([TextInput(f"msg-{i}")]) for i in range(5)]
        ctx = Context(stream=MemoryStream())
        result = await compact.compact(events, ctx, None)
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_truncates_above_target(self) -> None:
        compact = TailWindowCompact(target=3)
        events = [ModelRequest([TextInput(f"msg-{i}")]) for i in range(10)]
        ctx = Context(stream=MemoryStream())
        result = await compact.compact(events, ctx, None)
        assert len(result) == 3
        assert result[0].parts[0].content == "msg-7"

    @pytest.mark.asyncio
    async def test_persists_dropped_to_store(self) -> None:
        store = MemoryKnowledgeStore()
        compact = TailWindowCompact(target=3)
        events = [ModelRequest([TextInput(f"msg-{i}")]) for i in range(10)]
        stream = MemoryStream()
        ctx = Context(stream=stream)
        result = await compact.compact(events, ctx, store)
        assert len(result) == 3

        # Check that dropped events were persisted
        entries = await store.list("/log/")
        dropped = [e for e in entries if "dropped" in e]
        assert len(dropped) == 1

    @pytest.mark.asyncio
    async def test_no_persist_without_store(self) -> None:
        compact = TailWindowCompact(target=3)
        events = [ModelRequest([TextInput(f"msg-{i}")]) for i in range(10)]
        ctx = Context(stream=MemoryStream())
        result = await compact.compact(events, ctx, None)
        assert len(result) == 3


def _cycle(cid: str, name: str = "t") -> tuple[ModelResponse, ToolResultsEvent]:
    """A (ModelResponse tool call, ToolResultsEvent) pair linked by id."""
    call = ToolCallEvent(id=cid, name=name, arguments="{}")
    mr = ModelResponse(tool_calls=ToolCallsEvent(calls=[call]))
    res = ToolResultsEvent(results=[ToolResultEvent(parent_id=cid, name=name, result=ToolResult("ok"))])
    return mr, res


@pytest.mark.asyncio
class TestTailWindowToolCycleBoundary:
    """A tool-call/result cycle must never be split across the compaction
    boundary — a retained orphan result would crash strict providers."""

    async def test_split_cycle_compacts_whole(self) -> None:
        mr, res = _cycle("c1")
        events = [
            ModelRequest([TextInput("u0")]),
            mr,
            res,
            ModelRequest([TextInput("u1")]),
            ModelResponse(ModelMessage("done")),
        ]
        # target=3 would cut between the call and its result
        result = await TailWindowCompact(target=3).compact(events, Context(stream=MemoryStream()), None)
        assert result == [events[3], events[4]]

    async def test_clean_cycle_boundary_kept(self) -> None:
        mr, res = _cycle("c1")
        events = [ModelRequest([TextInput("u0")]), mr, res]
        result = await TailWindowCompact(target=2).compact(events, Context(stream=MemoryStream()), None)
        assert result == [mr, res]

    async def test_split_second_of_chained_cycles(self) -> None:
        mr1, res1 = _cycle("c1")
        mr2, res2 = _cycle("c2")
        events = [mr1, res1, mr2, res2, ModelResponse(ModelMessage("end"))]
        # target=2 would cut between the second call and its result
        result = await TailWindowCompact(target=2).compact(events, Context(stream=MemoryStream()), None)
        assert result == [events[4]]

    async def test_split_cycle_persisted_whole(self) -> None:
        store = MemoryKnowledgeStore()
        mr, res = _cycle("c1")
        events = [
            ModelRequest([TextInput("u0")]),
            mr,
            res,
            ModelRequest([TextInput("u1")]),
            ModelResponse(ModelMessage("done")),
        ]
        result = await TailWindowCompact(target=3).compact(events, Context(stream=MemoryStream()), store)
        assert mr not in result and res not in result
        entries = await store.list("/log/")
        assert [e for e in entries if "dropped" in e]


class TestCompactionSummary:
    def test_is_base_event(self) -> None:
        summary = CompactionSummary(summary="Earlier work...", event_count=50)
        assert summary.summary == "Earlier work..."
        assert summary.event_count == 50


class TestCompactTrigger:
    def test_defaults(self) -> None:
        trigger = CompactTrigger()
        assert trigger.max_events == 0
        assert trigger.max_tokens == 0
        assert trigger.chars_per_token == 4

    def test_custom_values(self) -> None:
        trigger = CompactTrigger(max_events=100, max_tokens=50000)
        assert trigger.max_events == 100
        assert trigger.max_tokens == 50000

    def test_custom_chars_per_token(self) -> None:
        trigger = CompactTrigger(max_tokens=100, chars_per_token=2)
        assert trigger.chars_per_token == 2


class TestCompactionWiredOnAgent:
    """End-to-end: an Agent configured with compaction emits CompactionCompleted
    on the stream and shrinks history once the trigger threshold is crossed."""

    @pytest.mark.asyncio
    async def test_fires_when_threshold_crossed(self) -> None:
        store = MemoryKnowledgeStore()
        stream = MemoryStream()
        completions: list[CompactionCompleted] = []
        stream.where(CompactionCompleted).subscribe(lambda e: completions.append(e))

        agent = Agent(
            "compactor",
            config=TestConfig(
                ModelResponse(ModelMessage("a")),
                ModelResponse(ModelMessage("b")),
                ModelResponse(ModelMessage("c")),
                ModelResponse(ModelMessage("d")),
            ),
            knowledge=KnowledgeConfig(
                store=store,
                compact=TailWindowCompact(target=2),
                compact_trigger=CompactTrigger(max_events=3),
            ),
        )

        # Four turns on the same stream — history grows past max_events=3
        reply = await agent.ask("turn-1", stream=stream)
        for q in ("turn-2", "turn-3", "turn-4"):
            reply = await reply.ask(q)

        assert len(completions) >= 1
        assert completions[0].agent == "compactor"
        assert completions[0].strategy == "TailWindowCompact"
        assert completions[0].events_after <= 2

    @pytest.mark.asyncio
    async def test_does_not_fire_below_threshold(self) -> None:
        store = MemoryKnowledgeStore()
        stream = MemoryStream()
        completions: list[CompactionCompleted] = []
        stream.where(CompactionCompleted).subscribe(lambda e: completions.append(e))

        agent = Agent(
            "compactor",
            config=TestConfig(ModelResponse(ModelMessage("hi"))),
            knowledge=KnowledgeConfig(
                store=store,
                compact=TailWindowCompact(target=2),
                compact_trigger=CompactTrigger(max_events=100),
            ),
        )
        await agent.ask("once", stream=stream)

        assert completions == []


class _RaisingCompact:
    """CompactStrategy that always raises — for failure-path tests."""

    last_usage: dict = {}

    async def compact(self, events, context, store) -> list:
        raise RuntimeError("compact boom")


@pytest.mark.asyncio
class TestCompactionLifecycleEvents:
    """Started + Failed events must reach the stream so failures are
    observable without configuring Python logging."""

    async def test_started_event_fires_before_strategy_runs(self) -> None:
        store = MemoryKnowledgeStore()
        stream = MemoryStream()
        started: list[CompactionStarted] = []
        stream.where(CompactionStarted).subscribe(lambda e: started.append(e))

        agent = Agent(
            "compactor",
            config=TestConfig(
                ModelResponse(ModelMessage("a")),
                ModelResponse(ModelMessage("b")),
                ModelResponse(ModelMessage("c")),
                ModelResponse(ModelMessage("d")),
            ),
            knowledge=KnowledgeConfig(
                store=store,
                compact=TailWindowCompact(target=2),
                compact_trigger=CompactTrigger(max_events=3),
            ),
        )
        reply = await agent.ask("turn-1", stream=stream)
        for q in ("turn-2", "turn-3", "turn-4"):
            reply = await reply.ask(q)

        assert started
        assert started[0].agent == "compactor"
        assert started[0].strategy == "TailWindowCompact"

    async def test_failed_event_fires_when_strategy_raises(self) -> None:
        store = MemoryKnowledgeStore()
        stream = MemoryStream()
        failures: list[CompactionFailed] = []
        completions: list[CompactionCompleted] = []
        stream.where(CompactionFailed).subscribe(lambda e: failures.append(e))
        stream.where(CompactionCompleted).subscribe(lambda e: completions.append(e))

        agent = Agent(
            "broken-compactor",
            config=TestConfig(
                ModelResponse(ModelMessage("a")),
                ModelResponse(ModelMessage("b")),
            ),
            knowledge=KnowledgeConfig(
                store=store,
                compact=_RaisingCompact(),
                compact_trigger=CompactTrigger(max_events=1),
            ),
        )
        # The turn itself succeeds — only compaction failed.
        await agent.ask("turn-1", stream=stream)

        assert len(failures) == 1
        assert failures[0].agent == "broken-compactor"
        assert failures[0].strategy == "_RaisingCompact"
        assert failures[0].error_type == "RuntimeError"
        assert "compact boom" in failures[0].error
        assert completions == []
