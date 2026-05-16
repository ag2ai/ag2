# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for MemoryToolkit."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from autogen.beta import Agent, Context
from autogen.beta.events import ToolCallEvent, ToolResultsEvent
from autogen.beta.testing import TestConfig, TrackingConfig
from autogen.beta.tools import MemoryToolkit
from autogen.beta.tools.toolkits.memory import _InMemoryStore, _SQLiteStore

# ---------------------------------------------------------------------------
# _InMemoryStore — unit tests
# ---------------------------------------------------------------------------


class TestInMemoryStore:
    def test_store_and_retrieve(self) -> None:
        s = _InMemoryStore()
        s.store("k1", "hello world")
        entry = s.retrieve("k1")
        assert entry is not None
        content, _ts = entry
        assert content == "hello world"

    def test_retrieve_missing_key(self) -> None:
        s = _InMemoryStore()
        assert s.retrieve("ghost") is None

    def test_search_finds_content(self) -> None:
        s = _InMemoryStore()
        s.store("a", "Paris is the capital of France")
        s.store("b", "Berlin is the capital of Germany")
        hits = s.search("Paris", 10)
        assert len(hits) == 1
        assert hits[0][0] == "a"

    def test_search_finds_key(self) -> None:
        s = _InMemoryStore()
        s.store("special-key", "some content")
        hits = s.search("special-key", 10)
        assert any(h[0] == "special-key" for h in hits)

    def test_search_case_insensitive(self) -> None:
        s = _InMemoryStore()
        s.store("k", "Python is great")
        assert s.search("python", 5)
        assert s.search("PYTHON", 5)

    def test_search_respects_max_results(self) -> None:
        s = _InMemoryStore()
        for i in range(10):
            s.store(f"item-{i}", f"item number {i}")
        hits = s.search("item", 3)
        assert len(hits) <= 3

    def test_delete_existing(self) -> None:
        s = _InMemoryStore()
        s.store("temp", "gone soon")
        assert s.delete("temp") is True
        assert s.retrieve("temp") is None

    def test_delete_nonexistent(self) -> None:
        s = _InMemoryStore()
        assert s.delete("ghost") is False

    def test_overwrite(self) -> None:
        s = _InMemoryStore()
        s.store("k", "original")
        s.store("k", "updated")
        content, _ = s.retrieve("k")  # type: ignore[misc]
        assert content == "updated"

    def test_list_all_empty(self) -> None:
        s = _InMemoryStore()
        assert s.list_all() == []

    def test_list_all_returns_all_entries(self) -> None:
        s = _InMemoryStore()
        s.store("a", "alpha")
        s.store("b", "beta")
        all_entries = s.list_all()
        keys = {e[0] for e in all_entries}
        assert keys == {"a", "b"}

    def test_independent_instances(self) -> None:
        s1 = _InMemoryStore()
        s2 = _InMemoryStore()
        s1.store("shared-key", "only in s1")
        assert s2.search("only in s1", 5) == []


# ---------------------------------------------------------------------------
# _SQLiteStore — unit tests
# ---------------------------------------------------------------------------


class TestSQLiteStore:
    def test_store_and_retrieve(self, tmp_path: Path) -> None:
        s = _SQLiteStore(tmp_path / "test.db")
        s.store("k1", "hello sqlite")
        content, _ = s.retrieve("k1")  # type: ignore[misc]
        assert content == "hello sqlite"

    def test_retrieve_missing(self, tmp_path: Path) -> None:
        s = _SQLiteStore(tmp_path / "test.db")
        assert s.retrieve("ghost") is None

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        db = tmp_path / "persist.db"
        _SQLiteStore(db).store("durable", "survived restart")
        content, _ = _SQLiteStore(db).retrieve("durable")  # type: ignore[misc]
        assert content == "survived restart"

    def test_search(self, tmp_path: Path) -> None:
        s = _SQLiteStore(tmp_path / "search.db")
        s.store("a", "unique searchable text")
        hits = s.search("unique searchable", 5)
        assert len(hits) == 1
        assert hits[0][0] == "a"

    def test_delete(self, tmp_path: Path) -> None:
        s = _SQLiteStore(tmp_path / "del.db")
        s.store("gone", "byebye")
        assert s.delete("gone") is True
        assert s.retrieve("gone") is None

    def test_delete_nonexistent(self, tmp_path: Path) -> None:
        s = _SQLiteStore(tmp_path / "del2.db")
        assert s.delete("ghost") is False

    def test_list_all(self, tmp_path: Path) -> None:
        s = _SQLiteStore(tmp_path / "list.db")
        s.store("x", "ex")
        s.store("y", "why")
        keys = {e[0] for e in s.list_all()}
        assert keys == {"x", "y"}

    def test_delete_persists_across_instances(self, tmp_path: Path) -> None:
        db = tmp_path / "forgetter.db"
        _SQLiteStore(db).store("gone", "will be deleted")
        _SQLiteStore(db).delete("gone")
        assert _SQLiteStore(db).retrieve("gone") is None


# ---------------------------------------------------------------------------
# MemoryToolkit schemas
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_toolkit_schemas(async_mock: AsyncMock) -> None:
    toolkit = MemoryToolkit()
    schemas = list(await toolkit.schemas(Context(async_mock)))
    names = {s.function.name for s in schemas}
    assert names == {"remember", "recall", "forget", "list_memories"}


def test_toolkit_name() -> None:
    assert MemoryToolkit().name == "memory_toolkit"


def test_toolkit_tools_dict() -> None:
    m = MemoryToolkit()
    assert set(m._tools.keys()) == {"remember", "recall", "forget", "list_memories"}


# ---------------------------------------------------------------------------
# End-to-end: agent invokes remember, then recall
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_remember_call() -> None:
    toolkit = MemoryToolkit()

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(
                name="remember",
                arguments=json.dumps({"content": "The sky is blue.", "key": "sky"}),
            ),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[toolkit])
    await agent.ask("remember something")

    # Second mock call receives the tool result
    tool_result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    result_text = tool_result_msg.results[0].result.parts[0].content
    assert "sky" in result_text


@pytest.mark.asyncio
async def test_agent_recall_call() -> None:
    toolkit = MemoryToolkit()
    toolkit._store.store("sky", "The sky is blue.")

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(
                name="recall",
                arguments=json.dumps({"query": "sky", "max_results": 3}),
            ),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[toolkit])
    await agent.ask("recall something")

    tool_result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    result_text = tool_result_msg.results[0].result.parts[0].content
    assert "sky" in result_text
    assert "The sky is blue." in result_text
