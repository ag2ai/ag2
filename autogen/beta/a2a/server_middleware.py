# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Protocol, TypeAlias

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue

ExecutorCall: TypeAlias = Callable[[RequestContext, EventQueue], Awaitable[None]]
"""``call_next`` shape — the inner-most call ends up running ``AG2AgentExecutor.execute``."""


class ExecutorMiddleware(Protocol):
    """Wrapper around ``AG2AgentExecutor.execute``.

    Compose multiple middleware by passing a list to ``A2AServer(executor_middleware=...)``.
    Each middleware receives ``call_next`` already wrapped by every later entry,
    so list order = outer → inner (first entry runs first on the way in, last on
    the way out — same as Starlette).
    """

    async def __call__(
        self,
        call_next: ExecutorCall,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None: ...


def compose(
    middleware: Iterable[ExecutorMiddleware],
    inner: ExecutorCall,
) -> ExecutorCall:
    """Fold ``middleware`` into a single call chain ending in ``inner``.

    Order: the first item in ``middleware`` is the outer-most wrapper.
    """
    chain = inner
    for mw in reversed(list(middleware)):
        chain = _wrap(mw, chain)
    return chain


def _wrap(mw: ExecutorMiddleware, call_next: ExecutorCall) -> ExecutorCall:
    async def wrapped(context: RequestContext, event_queue: EventQueue) -> None:
        await mw(call_next, context, event_queue)

    return wrapped


class ExecutorLoggingMiddleware:
    """Log task lifecycle events (start, end, duration) to a stdlib logger."""

    __slots__ = ("_logger",)

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("autogen.beta.a2a.executor")

    async def __call__(
        self,
        call_next: ExecutorCall,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        task_id = context.task_id or "<new>"
        context_id = context.context_id or "<new>"
        self._logger.info("task=%s context=%s start", task_id, context_id)
        started = time.perf_counter()
        try:
            await call_next(context, event_queue)
        except Exception:
            elapsed = time.perf_counter() - started
            self._logger.exception("task=%s context=%s failed after %.3fs", task_id, context_id, elapsed)
            raise
        else:
            elapsed = time.perf_counter() - started
            self._logger.info("task=%s context=%s done in %.3fs", task_id, context_id, elapsed)


class ExecutorMetricsMiddleware:
    """Accumulate task counts and total elapsed seconds in-process.

    Suitable as a starting point — wire to Prometheus / OpenTelemetry by
    overriding ``_record`` or by reading the ``stats`` property out-of-band.

    .. warning::
       ``stats[k] += value`` on a shared ``dict`` is **not** atomic across
       concurrent task executions. Numbers may drift under load. For
       production telemetry, replace this middleware with one that uses an
       ``asyncio.Lock`` or an external thread-safe accumulator.
    """

    __slots__ = ("stats",)

    def __init__(self) -> None:
        self.stats: dict[str, float] = {
            "started": 0.0,
            "completed": 0.0,
            "failed": 0.0,
            "elapsed_total": 0.0,
        }

    async def __call__(
        self,
        call_next: ExecutorCall,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        self.stats["started"] += 1
        started = time.perf_counter()
        try:
            await call_next(context, event_queue)
        except Exception:
            self.stats["failed"] += 1
            raise
        else:
            self.stats["completed"] += 1
        finally:
            self.stats["elapsed_total"] += time.perf_counter() - started


__all__ = (
    "ExecutorCall",
    "ExecutorLoggingMiddleware",
    "ExecutorMetricsMiddleware",
    "ExecutorMiddleware",
    "compose",
)
