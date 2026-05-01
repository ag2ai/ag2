# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

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


__all__ = (
    "ExecutorCall",
    "ExecutorMiddleware",
    "compose",
)
