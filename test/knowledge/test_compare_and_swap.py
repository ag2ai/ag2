# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sqlite3
import sys

import pytest

from ag2.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore, SqliteKnowledgeStore


@pytest.fixture(params=["memory", "sqlite", "disk"])
def stores(request, tmp_path):
    if request.param == "memory":
        first = second = MemoryKnowledgeStore()
    elif request.param == "sqlite":
        first, second = [SqliteKnowledgeStore(tmp_path / "store.sqlite") for _ in range(2)]
    else:
        pytest.importorskip("watchdog")
        if sys.platform == "win32":
            pytest.skip("DiskKnowledgeStore is POSIX-only")
        first, second = [DiskKnowledgeStore(tmp_path / "files") for _ in range(2)]
    try:
        yield first, second
    finally:
        if request.param == "sqlite":
            first.close()
            second.close()


@pytest.mark.asyncio
async def test_missing_and_empty_are_distinct(stores):
    first, second = stores
    assert await first.compare_and_swap("/note", None, "") is True
    assert await second.compare_and_swap("/note", None, "wrong") is False
    assert await second.compare_and_swap("/note", "", "雪☃") is True
    assert await first.read("/note") == "雪☃"


@pytest.mark.asyncio
async def test_only_one_independent_handle_can_replace_same_snapshot(stores):
    first, second = stores
    await first.write("/note", "before")
    outcomes = await asyncio.gather(
        first.compare_and_swap("note", "before", "first"),
        second.compare_and_swap("//note/", "before", "second"),
    )
    assert sorted(outcomes) == [False, True]
    assert await first.read("/note") == ("first" if outcomes[0] else "second")


@pytest.mark.asyncio
async def test_failed_comparison_preserves_content_and_index(stores):
    first, second = stores
    await first.write("/folder/note", "kept")
    assert await second.compare_and_swap("/folder/note", "stale", "lost") is False
    assert await second.compare_and_swap("/missing", "", "wrong") is False
    assert await first.read("/folder/note") == "kept"
    assert await first.list("/") == ["folder/"]


@pytest.mark.asyncio
async def test_sqlite_failed_commit_rolls_back_and_allows_retry(tmp_path):
    path = tmp_path / "store.sqlite"
    store = SqliteKnowledgeStore(path)
    try:
        await store.write("/note", "before")
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TRIGGER reject_content BEFORE INSERT ON entries "
                "WHEN CAST(NEW.content AS TEXT) = 'reject' "
                "BEGIN SELECT RAISE(ABORT, 'rejected content'); END"
            )
        with pytest.raises(sqlite3.IntegrityError, match="rejected content"):
            await store.compare_and_swap("/note", "before", "reject")
        assert await store.read("/note") == "before"
        assert await store.compare_and_swap("/note", "before", "after") is True
    finally:
        store.close()


@pytest.mark.asyncio
async def test_sqlite_cancelled_commit_finishes_before_connection_is_reused(tmp_path):
    path = tmp_path / "store.sqlite"
    store = SqliteKnowledgeStore(path)
    await store.write("/note", "before")
    blocker = sqlite3.connect(path)
    blocker.execute("BEGIN IMMEDIATE")
    operation = asyncio.create_task(store.compare_and_swap("/note", "before", "after"))
    reader = None
    try:
        await asyncio.sleep(0.05)
        operation.cancel()
        await asyncio.sleep(0)
        operation.cancel()  # repeated cancellation must not release the connection lock
        reader = asyncio.create_task(store.read("/note"))
        await asyncio.sleep(0.01)
        assert not reader.done()
        blocker.rollback()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert await reader == "after"
    finally:
        blocker.rollback()
        await asyncio.gather(operation, *([reader] if reader is not None else []), return_exceptions=True)
        store.close()
        blocker.close()
