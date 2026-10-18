# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("watchdog")

from ag2.knowledge import DiskKnowledgeStore


# Without the fix, a subscriber on the old name never hears about a rename, but only on backends
# that report a rename as a single event (inotify). macOS also reports a modification of the
# source, so these cases pass there either way.
@pytest.mark.skipif(sys.platform == "win32", reason="DiskKnowledgeStore is POSIX-only")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "watched, expected",
    [
        ("/notes.md", {"/notes.md"}),
        ("/", {"/notes.md", "/archive.md"}),
        ("/archive.md", {"/archive.md"}),
    ],
    ids=["old-name", "directory", "new-name"],
)
async def test_rename_notifies_subscribers_of_both_names(tmp_path: Path, watched: str, expected: set[str]) -> None:
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
    finally:
        await subscription.close()
