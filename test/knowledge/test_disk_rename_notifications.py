# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("watchdog")

from ag2.knowledge import DiskKnowledgeStore


@pytest.mark.skipif(sys.platform == "win32", reason="DiskKnowledgeStore is POSIX-only")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "watched, expected",
    [
        ("/notes.md", {"/notes.md"}),
        ("/", {"/notes.md", "/archive.md"}),
        ("/archive.md", {"/archive.md"}),
    ],
    ids=["source-file", "both-paths", "destination-file"],
)
async def test_rename_notifies_affected_paths(tmp_path: Path, watched: str, expected: set[str]) -> None:
    store = DiskKnowledgeStore(tmp_path)
    await store.write("/notes.md", "agent knowledge")
    received: set[str] = set()
    complete = asyncio.Event()

    async def callback(path: str) -> None:
        received.add(path)
        if expected <= received:
            complete.set()

    subscription = await store.on_change(watched, callback)
    try:
        (tmp_path / "notes.md").rename(tmp_path / "archive.md")
        await asyncio.wait_for(complete.wait(), timeout=5.0)
        assert received == expected
        assert await store.read("/notes.md") is None
        assert await store.read("/archive.md") == "agent knowledge"
    finally:
        await subscription.close()
