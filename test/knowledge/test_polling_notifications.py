# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from typing import Literal

import pytest

from ag2.knowledge import SqliteKnowledgeStore


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["write", "append", "delete", "mixed"])
@pytest.mark.parametrize("close_in_callback", [False, True])
async def test_polling_batch_respects_subscription_close(
    tmp_path: Path,
    operation: Literal["write", "append", "delete", "mixed"],
    close_in_callback: bool,
) -> None:
    store = SqliteKnowledgeStore(tmp_path / "knowledge.db", poll_interval_s=0.05)
    paths = ["/watched/first.txt", "/watched/second.txt"]
    received: list[str] = []
    delivered = asyncio.Event()

    async def changed(path: str) -> None:
        if path == "/watched/trigger":
            # The watcher cannot take another snapshot until this callback returns,
            # so both mutations are guaranteed to belong to the same next batch.
            if operation == "delete":
                for target in paths:
                    await store.delete(target)
            elif operation == "mixed":
                await store.write(paths[0], "updated")
                await store.delete(paths[1])
            else:
                for target in paths:
                    if operation == "append":
                        await store.append(target, "appended")
                    else:
                        await store.write(target, "updated")
            return

        received.append(path)
        if close_in_callback:
            await subscription.close()
        if close_in_callback or len(received) == len(paths):
            delivered.set()

    try:
        for path in paths:
            await store.write(path, "initial")
        subscription = await store.on_change("/watched", changed)
        try:
            await store.write("/watched/trigger", "start batch")
            await asyncio.wait_for(delivered.wait(), timeout=5)

            if close_in_callback:
                assert len(received) == 1
                assert received[0] in paths
            else:
                assert sorted(received) == paths
        finally:
            await subscription.close()
            await subscription.close()
    finally:
        store.close()


@pytest.mark.asyncio
async def test_polling_continues_after_callback_error(tmp_path: Path) -> None:
    store = SqliteKnowledgeStore(tmp_path / "knowledge.db", poll_interval_s=0.05)
    failed = asyncio.Event()
    recovered = asyncio.Event()
    received: list[str] = []

    async def changed(path: str) -> None:
        received.append(path)
        if path == "/watched/first.txt":
            failed.set()
            raise ValueError("consumer failed")
        recovered.set()

    subscription = await store.on_change("/watched", changed)
    try:
        await store.write("/watched/first.txt", "first")
        await asyncio.wait_for(failed.wait(), timeout=5)
        await store.write("/watched/second.txt", "second")
        await asyncio.wait_for(recovered.wait(), timeout=5)
        assert received == ["/watched/first.txt", "/watched/second.txt"]
    finally:
        await subscription.close()
        store.close()
