# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
from pathlib import Path
from typing import Literal

import pytest
from dirty_equals import IsList, IsOneOf

from ag2.knowledge import SqliteKnowledgeStore

TRIGGER = "/watched/trigger"
PATHS = ["/watched/first.txt", "/watched/second.txt"]

Mutation = tuple[Literal["write", "append", "delete"], str]

# Each batch mutates both watched paths, so the next poll delivers two changes.
BATCHES: dict[str, list[Mutation]] = {
    "write": [("write", PATHS[0]), ("write", PATHS[1])],
    "append": [("append", PATHS[0]), ("append", PATHS[1])],
    "delete": [("delete", PATHS[0]), ("delete", PATHS[1])],
    "mixed": [("write", PATHS[0]), ("delete", PATHS[1])],
}


async def _apply(store: SqliteKnowledgeStore, batch: list[Mutation]) -> None:
    for operation, path in batch:
        if operation == "delete":
            await store.delete(path)
        elif operation == "append":
            await store.append(path, "appended")
        else:
            await store.write(path, "updated")


async def _seed(store: SqliteKnowledgeStore) -> None:
    for path in PATHS:
        await store.write(path, "initial")


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", BATCHES.values(), ids=BATCHES.keys())
@pytest.mark.parametrize("close_in_callback", [False, True])
async def test_polling_batch_respects_subscription_close(
    tmp_path: Path,
    batch: list[Mutation],
    close_in_callback: bool,
) -> None:
    store = SqliteKnowledgeStore(tmp_path / "knowledge.db", poll_interval_s=0.05)
    received: list[str] = []
    delivered = asyncio.Event()

    async def changed(path: str) -> None:
        if path == TRIGGER:
            # The watcher cannot take another snapshot until this callback returns,
            # so both mutations are guaranteed to belong to the same next batch.
            await _apply(store, batch)
            return

        received.append(path)
        if close_in_callback:
            await subscription.close()
        if close_in_callback or len(received) == len(PATHS):
            delivered.set()

    try:
        await _seed(store)
        subscription = await store.on_change("/watched", changed)
        try:
            await store.write(TRIGGER, "start batch")
            await asyncio.wait_for(delivered.wait(), timeout=5)

            if close_in_callback:
                assert received == [IsOneOf(*PATHS)]
            else:
                assert received == IsList(*PATHS, check_order=False)
        finally:
            await subscription.close()
    finally:
        store.close()


@pytest.mark.asyncio
async def test_polling_batch_stops_when_closed_from_another_task(tmp_path: Path) -> None:
    store = SqliteKnowledgeStore(tmp_path / "knowledge.db", poll_interval_s=0.05)
    received: list[str] = []
    in_first_callback = asyncio.Event()
    release = asyncio.Event()

    async def changed(path: str) -> None:
        if path == TRIGGER:
            await _apply(store, BATCHES["write"])
            return

        received.append(path)
        if len(received) == 1:
            in_first_callback.set()
            # The first delivery is still in flight when close() is called and
            # finishes normally (a consumer may absorb an interruption); the
            # rest of the batch must not start either way.
            with contextlib.suppress(asyncio.CancelledError):
                await release.wait()

    try:
        await _seed(store)
        subscription = await store.on_change("/watched", changed)
        try:
            await store.write(TRIGGER, "start batch")
            await asyncio.wait_for(in_first_callback.wait(), timeout=5)
            closing = asyncio.create_task(subscription.close())
            release.set()
            await asyncio.wait_for(closing, timeout=5)

            assert received == [IsOneOf(*PATHS)]
        finally:
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
