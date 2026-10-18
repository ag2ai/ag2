# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("watchdog")

from ag2.knowledge import SqliteKnowledgeStore

# How long the contending appends get to reach their read before the writer lock is released.
_CONTENTION_WINDOW_S = 0.2


@pytest.mark.asyncio
@pytest.mark.parametrize("initial", ["", "existing\n"])
async def test_append_is_atomic_across_store_instances(tmp_path: Path, initial: str) -> None:
    database = tmp_path / "knowledge.sqlite"
    first = SqliteKnowledgeStore(database)
    second = SqliteKnowledgeStore(database)
    await first.write("/log.jsonl", initial)
    # Open both connections (and finish schema setup) before contention starts.
    await second.read("/log.jsonl")

    # Hold the database's writer lock while both stores try to append. An append that
    # reads before taking that lock sees the same stale content in both stores.
    blocker = sqlite3.connect(database)
    blocker.execute("BEGIN IMMEDIATE")
    payloads = ["alpha\n", "\u4e2d\u6587\n"]
    appends = [asyncio.create_task(store.append("/log.jsonl", text)) for store, text in zip((first, second), payloads)]

    try:
        # Give both appends time to reach the point where they would have read.
        await asyncio.sleep(_CONTENTION_WINDOW_S)
        blocker.rollback()
        offsets = await asyncio.gather(*appends)
        assert await first.read("/log.jsonl") in (
            initial + payloads[0] + payloads[1],
            initial + payloads[1] + payloads[0],
        )
        assert len(set(offsets)) == len(offsets)
        for offset, text in zip(offsets, payloads):
            assert await first.read_range("/log.jsonl", offset, offset + len(text.encode("utf-8"))) == text
    finally:
        blocker.rollback()
        await asyncio.gather(*appends, return_exceptions=True)
        first.close()
        second.close()
        blocker.close()


@pytest.mark.asyncio
async def test_concurrent_appends_from_many_stores_are_all_kept(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.sqlite"
    stores = [SqliteKnowledgeStore(database) for _ in range(8)]
    lines = [f"line-{i}\n" for i in range(len(stores))]
    try:
        await stores[0].write("/log.jsonl", "")
        offsets = await asyncio.gather(*(store.append("/log.jsonl", line) for store, line in zip(stores, lines)))

        content = await stores[0].read("/log.jsonl")
        assert content is not None
        assert sorted(content.splitlines(keepends=True)) == lines
        assert sorted(offsets) == [i * len(lines[0]) for i in range(len(lines))]
    finally:
        for store in stores:
            store.close()


@pytest.mark.asyncio
async def test_failed_append_rolls_back_and_allows_retry(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.sqlite"
    store = SqliteKnowledgeStore(database)
    await store.write("/log.jsonl", "before\n")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TRIGGER reject_append BEFORE INSERT ON entries "
            "WHEN instr(CAST(NEW.content AS TEXT), 'reject') > 0 "
            "BEGIN SELECT RAISE(ABORT, 'rejected append'); END"
        )
    try:
        with pytest.raises(sqlite3.IntegrityError, match="rejected append"):
            await store.append("/log.jsonl", "reject\n")
        assert await store.read("/log.jsonl") == "before\n"
        assert await store.append("/log.jsonl", "after\n") == len(b"before\n")
        assert await store.read("/log.jsonl") == "before\nafter\n"
    finally:
        store.close()
