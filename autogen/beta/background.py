# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Background agent tasks — non-blocking :meth:`~autogen.beta.Agent.ask` invocations.

:func:`run_in_background` starts an agent call without blocking the caller.
It returns a :class:`BackgroundTask` immediately.  Await the task to collect
the :class:`~autogen.beta.AgentReply` when the agent finishes.

Example::

    import asyncio
    from autogen.beta import Agent
    from autogen.beta.background import run_in_background
    from autogen.beta.config import OpenAIConfig


    async def main():
        agent = Agent("analyst", config=OpenAIConfig("gpt-4o-mini"))

        # Start without blocking
        task = run_in_background(agent, "Summarise the Q1 results.")

        # Do other awaitable work concurrently
        await asyncio.sleep(0)

        # Collect the result when ready
        reply = await task
        print(await reply.content())


    asyncio.run(main())

To run *many* agents in parallel and collect all results at once, use
:func:`~autogen.beta.fanout` instead.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from .agent import Agent, AgentReply

__all__ = ("BackgroundTask", "run_in_background")

_T = TypeVar("_T")


class BackgroundTask(Generic[_T]):
    """A handle for an :meth:`~autogen.beta.Agent.ask` call running concurrently.

    Wraps an :class:`asyncio.Task`.  Await it to get the
    :class:`~autogen.beta.AgentReply` when the agent finishes.

    Example::

        task = run_in_background(agent, "Long analysis…")

        # Continue doing other async work…
        await asyncio.sleep(0)

        # Collect result (waits until the agent finishes if still running)
        reply = await task

    Args:
        task:
            An :class:`asyncio.Task` wrapping :meth:`~autogen.beta.Agent.ask`.
    """

    __slots__ = ("_task",)

    def __init__(self, task: asyncio.Task[AgentReply[_T, Any]]) -> None:
        self._task = task

    # ------------------------------------------------------------------
    # Task control
    # ------------------------------------------------------------------

    def cancel(self, msg: str | None = None) -> bool:
        """Request cancellation of the background agent call.

        Returns ``True`` if the task was successfully requested to cancel,
        ``False`` if it is already done.
        """
        return self._task.cancel(msg)  # type: ignore[call-arg]

    # ------------------------------------------------------------------
    # Status inspection
    # ------------------------------------------------------------------

    def done(self) -> bool:
        """Return ``True`` if the agent has finished (successfully or not)."""
        return self._task.done()

    def cancelled(self) -> bool:
        """Return ``True`` if the task was cancelled via :meth:`cancel`."""
        return self._task.cancelled()

    def result(self) -> AgentReply[_T, Any]:
        """Return the :class:`~autogen.beta.AgentReply`.

        Raises :exc:`asyncio.InvalidStateError` if the task is not yet done,
        :exc:`asyncio.CancelledError` if it was cancelled, or propagates any
        exception raised by the agent call.
        """
        return self._task.result()

    def exception(self) -> BaseException | None:
        """Return the exception raised by the agent call, or ``None`` on success.

        Raises :exc:`asyncio.InvalidStateError` if not done, or
        :exc:`asyncio.CancelledError` if it was cancelled.
        """
        return self._task.exception()

    # ------------------------------------------------------------------
    # Awaitable
    # ------------------------------------------------------------------

    def __await__(self):  # type: ignore[override]
        return self._task.__await__()


def run_in_background(
    agent: Agent[_T],
    *msg: Any,
    **kwargs: Any,
) -> BackgroundTask[_T]:
    """Start an :meth:`~autogen.beta.Agent.ask` call as a background asyncio task.

    Returns a :class:`BackgroundTask` immediately.  The agent runs concurrently
    with the caller — you can do other async work, then ``await`` the task to
    collect the result.

    Must be called from an async context (a running asyncio event loop).

    Example::

        task = run_in_background(agent, "Analyse the data.")
        other_result = await other_coroutine()
        reply = await task  # waits until agent finishes

    Args:
        agent:
            The :class:`~autogen.beta.Agent` to invoke.
        *msg:
            Positional arguments forwarded verbatim to
            :meth:`~autogen.beta.Agent.ask`.
        **kwargs:
            Keyword arguments forwarded verbatim to
            :meth:`~autogen.beta.Agent.ask`.

    Returns:
        A :class:`BackgroundTask` wrapping the underlying asyncio task.

    Raises:
        RuntimeError: If called outside a running event loop.
    """
    task: asyncio.Task[Any] = asyncio.create_task(agent.ask(*msg, **kwargs))
    return BackgroundTask(task)
