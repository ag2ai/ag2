# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import TYPE_CHECKING, Any, Generic

from typing_extensions import TypeVar as TypeVar313

from .context import Stream, StreamId
from .events import Input
from .types import SendableMessage

if TYPE_CHECKING:
    from .agent import AgentReply

__all__ = ("RunHandle", "RunRegistry")

TResult = TypeVar313("TResult", default=str)


class RunHandle(Generic[TResult]):
    """A handle to an agent turn started via :meth:`Agent.run`.

    The handle exposes:

    - :attr:`task_id` — the stream id, used to address this run in a registry.
    - :meth:`send` — push a follow-up message into the live run's inbox. It is
      read at the next step boundary; it does **not** interrupt the current
      model call. Safe to call from any thread.
    - :meth:`result` — await the turn's final :class:`AgentReply`.
    - :meth:`done` / :meth:`cancel`.

    ``send`` after the run has already finished is **not** an error: the message
    is durably appended to ``stream.pending_messages`` and consumed by the next
    ``ask``/``run`` on the same stream. Check :meth:`done` if you need to know.
    """

    __slots__ = ("stream", "_task", "_loop")

    def __init__(
        self,
        stream: Stream,
        task: "asyncio.Task[AgentReply[TResult, Any]]",
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.stream = stream
        self._task = task
        self._loop = loop

    @property
    def task_id(self) -> StreamId:
        """Stable id of this run — the underlying stream's id."""
        return self.stream.id

    def send(self, *content: "SendableMessage | Input") -> None:
        """Push follow-up content into the running turn's inbox (thread-safe).

        Read at the next step boundary, not mid-LLM-call. When invoked from a
        thread other than the run's event loop, the enqueue is marshalled onto
        that loop via ``call_soon_threadsafe`` so the plain-list inbox is only
        ever mutated from its owning loop.
        """
        if not content:
            return

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is self._loop:
            self.stream.enqueue(*content)
        else:
            self._loop.call_soon_threadsafe(self._enqueue, content)

    def _enqueue(self, content: "tuple[SendableMessage | Input, ...]") -> None:
        # call_soon_threadsafe passes a single positional arg; unpack here so
        # the scheduled callback runs on the owning loop.
        self.stream.enqueue(*content)

    def done(self) -> bool:
        """Whether the underlying turn has finished (completed/failed/cancelled)."""
        return self._task.done()

    def cancel(self) -> bool:
        """Request cancellation of the underlying turn."""
        return self._task.cancel()

    async def result(self) -> "AgentReply[TResult, Any]":
        """Await and return the turn's final :class:`AgentReply`."""
        return await self._task

    def __await__(self) -> "Any":
        return self._task.__await__()


class RunRegistry:
    """Maps ``task_id`` to the live :class:`RunHandle` for in-flight runs.

    Instantiate one per server (not a module global) to avoid leaking run state
    across servers or tests. Handles are auto-removed when their turn finishes.
    """

    __slots__ = ("_runs",)

    def __init__(self) -> None:
        self._runs: dict[StreamId, RunHandle[Any]] = {}

    def register(self, handle: "RunHandle[Any]") -> None:
        """Track ``handle`` and arrange auto-removal when its turn finishes."""
        self._runs[handle.task_id] = handle
        handle._task.add_done_callback(lambda _t, tid=handle.task_id: self._runs.pop(tid, None))

    def get(self, task_id: StreamId) -> "RunHandle[Any] | None":
        return self._runs.get(task_id)

    def send(self, task_id: StreamId, *content: "SendableMessage | Input") -> bool:
        """Route ``content`` to the run with ``task_id``.

        Returns ``True`` if a live run was found and the message was pushed,
        ``False`` if no such run is registered (e.g. already finished/removed).
        """
        handle = self._runs.get(task_id)
        if handle is None:
            return False
        handle.send(*content)
        return True

    def active(self) -> "list[StreamId]":
        """Ids of currently-registered (in-flight) runs."""
        return list(self._runs)
