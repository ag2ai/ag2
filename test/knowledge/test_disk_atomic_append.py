# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from pathlib import Path
from threading import Barrier

import pytest

pytest.importorskip("watchdog")

from ag2.knowledge import DiskKnowledgeStore


@pytest.mark.skipif(sys.platform == "win32", reason="DiskKnowledgeStore is POSIX-only")
@pytest.mark.asyncio
async def test_concurrent_append_offsets_identify_each_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_a = DiskKnowledgeStore(tmp_path)
    store_b = DiskKnowledgeStore(tmp_path)
    initial = "雪\n"
    await store_a.write("/log", initial)
    opened = Barrier(2, timeout=5)
    original_open = Path.open

    def open_together(path: Path, mode: str = "r", *args, **kwargs):
        handle = original_open(path, mode, *args, **kwargs)
        if path == tmp_path / "log" and mode == "ab":
            try:
                # Start both appends with file handles positioned at the same EOF.
                opened.wait()
            except BaseException:
                handle.close()
                raise
        return handle

    def append(store: DiskKnowledgeStore, payload: str) -> int:
        return asyncio.run(store.append("/log", payload))

    monkeypatch.setattr(Path, "open", open_together)
    payloads = ("猫\n", "dog\n")
    offsets = await asyncio.gather(
        asyncio.to_thread(append, store_a, payloads[0]),
        asyncio.to_thread(append, store_b, payloads[1]),
    )

    assert min(offsets) == len(initial.encode())
    for offset, payload in zip(offsets, payloads, strict=True):
        assert await store_a.read_range("/log", offset, offset + len(payload.encode())) == payload
