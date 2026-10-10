# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("watchdog")

from ag2.knowledge import SqliteKnowledgeStore


@pytest.mark.asyncio
@pytest.mark.parametrize("initial", ["", "existing\n"])
async def test_append_is_atomic_across_store_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, initial: str
) -> None:
    database = tmp_path / "knowledge.sqlite"
    seed = SqliteKnowledgeStore(database)
    await seed.write("/log.jsonl", initial)
    seed.close()

    # Hold the database's writer lock while two independent clients try to append.
    # If they read before acquiring that lock, both prepare a write from stale data.
    blocker = sqlite3.connect(database)
    blocker.execute("BEGIN IMMEDIATE")
    both_writing = threading.Event()
    counter_lock = threading.Lock()
    writers = 0
    original_connect = sqlite3.connect

    def trace(statement: str) -> None:
        nonlocal writers
        if statement.startswith("INSERT OR REPLACE INTO entries"):
            with counter_lock:
                writers += 1
                if writers == 2:
                    both_writing.set()

    def connect(path: str, **kwargs: Any) -> sqlite3.Connection:
        connection: sqlite3.Connection = original_connect(path, **kwargs)
        connection.set_trace_callback(trace)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    first = SqliteKnowledgeStore(database)
    second = SqliteKnowledgeStore(database)
    # Finish schema setup before the competing appends so only the writes under test contend.
    blocker.rollback()
    await first.read("/log.jsonl")
    await second.read("/log.jsonl")
    blocker.execute("BEGIN IMMEDIATE")
    payloads = ["alpha\n", "\u4e2d\u6587\n"]
    appends = [asyncio.create_task(store.append("/log.jsonl", text)) for store, text in zip((first, second), payloads)]

    try:
        # A serialized implementation waits before reading, so this wait can time out.
        # Release the real SQLite lock in either case; no store internals are patched.
        await asyncio.to_thread(both_writing.wait, 0.25)
        blocker.rollback()
        offsets = await asyncio.gather(*appends)
        actual = await first.read("/log.jsonl")
        assert actual in (initial + payloads[0] + payloads[1], initial + payloads[1] + payloads[0])
        for offset, text in zip(offsets, payloads):
            assert await first.read_range("/log.jsonl", offset, offset + len(text.encode("utf-8"))) == text
    finally:
        blocker.rollback()
        await asyncio.gather(*appends, return_exceptions=True)
        first.close()
        second.close()
        blocker.close()


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
