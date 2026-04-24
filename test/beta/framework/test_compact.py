# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for CompactStrategy, CompactTrigger, and built-in strategies."""

import pytest

from autogen.beta.compact import CompactTrigger, CompactionSummary, TailWindowCompact
from autogen.beta.context import ConversationContext as Context
from autogen.beta.events import ModelRequest, TextInput
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.stream import MemoryStream


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
