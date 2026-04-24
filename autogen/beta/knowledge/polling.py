# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
from typing import Any

from .base import ChangeCallback, _normalize


class PollingChangeWatcher:
    """Polls a store for changes and dispatches to an async callback.

    Used by backends that cannot observe changes natively (Sqlite,
    Redis). The hub-facing semantics mirror the Memory + Disk stores:

    * Subscribe to a path prefix.
    * Receive ``callback(changed_path)`` asynchronously when the
      backend reports a newer version for any key under that prefix.
    * Close the subscription via :meth:`close` to stop delivery.

    Two interfaces are required from the backend:

    * ``list_versions_under(prefix)`` — returns ``dict[path, version]``
      for every key beneath ``prefix``. The version can be an mtime
      float, a monotonic int, or anything comparable for equality.
    * Normal polling fires at ``interval_s`` (default 0.5s). The
      watcher diffs the current snapshot against its last-seen
      snapshot and fires the callback once per changed key.
    """

    def __init__(
        self,
        *,
        backend: Any,
        prefix: str,
        callback: ChangeCallback,
        interval_s: float = 0.5,
    ) -> None:
        self._backend = backend
        self._prefix = _normalize(prefix)
        self._callback = callback
        self._interval_s = max(0.05, float(interval_s))
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._last_snapshot: dict[str, Any] = {}

    async def start(self) -> None:
        self._last_snapshot = await self._backend.list_versions_under(self._prefix)
        self._task = asyncio.create_task(self._run(), name="store-poll-watcher")

    async def _run(self) -> None:
        while not self._closed:
            try:
                await asyncio.sleep(self._interval_s)
            except asyncio.CancelledError:
                return
            if self._closed:
                return
            try:
                current = await self._backend.list_versions_under(self._prefix)
            except Exception:  # pragma: no cover
                continue
            for path, version in current.items():
                prior = self._last_snapshot.get(path)
                if prior is None or prior != version:
                    await self._safe_fire(path)
            for path in self._last_snapshot:
                if path not in current:
                    await self._safe_fire(path)
            self._last_snapshot = current

    async def _safe_fire(self, path: str) -> None:
        try:  # noqa: SIM105
            await self._callback(path)
        except Exception:  # pragma: no cover
            pass  # callbacks must never crash the watcher

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
