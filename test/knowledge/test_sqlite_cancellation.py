# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import Executor
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("watchdog")

from ag2.knowledge import SqliteKnowledgeStore


class PausedExecutor:
    """Pause one real executor worker without replacing store internals."""

    def __init__(self, loop: asyncio.AbstractEventLoop, fail: bool) -> None:
        self.loop = loop
        self.submit = loop.run_in_executor
        self.started = asyncio.Event()
        self.release = threading.Event()
        self.paused = False
        self.fail = fail

    def run_in_executor(self, executor: Executor | None, func: Callable[..., Any], *args: Any) -> asyncio.Future[Any]:
        if self.paused:
            return self.submit(executor, func, *args)
        self.paused = True
        return self.submit(executor, self.execute, func, args)

    def execute(self, func: Callable[..., Any], args: tuple[Any, ...]) -> Any:
        self.loop.call_soon_threadsafe(self.started.set)
        if not self.release.wait(timeout=5):
            raise TimeoutError("Executor worker was not released")
        if self.fail:
            raise sqlite3.OperationalError("write failed")
        return func(*args)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_count", [1, 2])
@pytest.mark.parametrize("worker_fails", [False, True])
async def test_cancelled_write_settles_before_later_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_count: int, worker_fails: bool
) -> None:
    store = SqliteKnowledgeStore(tmp_path / "knowledge.sqlite")
    await store.write("/value", "initial")
    loop = asyncio.get_running_loop()
    executor = PausedExecutor(loop, worker_fails)
    monkeypatch.setattr(loop, "run_in_executor", executor.run_in_executor)
    first = asyncio.create_task(store.write("/value", "first"))
    second: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(executor.started.wait(), timeout=5)
        for _ in range(cancel_count):
            first.cancel()
            await asyncio.sleep(0)
        second = asyncio.create_task(store.write("/value", "second"))
        await asyncio.sleep(0)
        assert not first.done(), "Cancellation returned while its executor worker was still active"
        completed, _ = await asyncio.wait({second}, timeout=0.05)
        assert not completed, "A later write overtook the cancelled operation"
        executor.release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        assert await store.read("/value") == "second"
    finally:
        executor.release.set()
        tasks = [first] if second is None else [first, second]
        await asyncio.gather(*tasks, return_exceptions=True)
        store.close()
