# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Internal sweepers — TTL + expectations.

Both built on the ``_IntervalSweeper`` primitive. Sweepers are spawned
by ``Hub.start()`` and cancelled by ``Hub.close()``. ``Hub.open()``
calls ``start()`` automatically; tests that don't want background
timers can construct the hub with ``ttl_sweep_interval=0`` (disables
the TTL sweeper entirely) or ``expectation_sweep_interval=0`` (disables
the expectation sweeper).
"""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable

__all__ = ("_IntervalSweeper",)


class _IntervalSweeper:
    """Run a coroutine on a fixed interval until cancelled.

    Exceptions in the coroutine are swallowed so a transient failure
    in one tick does not kill the sweeper.
    """

    def __init__(
        self,
        name: str,
        interval: float,
        fn: Callable[[], Awaitable[None]],
    ) -> None:
        # __init__ stores params; start() spawns the task.
        self._name = name
        self._interval = interval
        self._fn = fn
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    @property
    def name(self) -> str:
        return self._name

    def start(self) -> None:
        """Spawn the loop. Idempotent — second call is a no-op."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop(), name=f"sweeper:{self._name}")

    async def _loop(self) -> None:
        while not self._stopping:
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                return
            if self._stopping:
                return
            try:
                await self._fn()
            except asyncio.CancelledError:
                return
            except Exception:
                # Swallow — a sweeper tick failure must not kill the loop.
                pass

    async def stop(self) -> None:
        """Cancel the loop. Idempotent."""
        self._stopping = True
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._task
        self._task = None
