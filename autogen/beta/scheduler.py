# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""AgentScheduler — run agent invocations on a schedule.

Use :class:`AgentScheduler` to drive ``agent.ask()`` calls on a time-based
schedule backed by the :mod:`autogen.beta.watch` primitives.

Example::

    from autogen.beta import Agent, AgentScheduler
    from autogen.beta.config import OpenAIConfig

    agent = Agent("assistant", config=OpenAIConfig("gpt-4o-mini"))

    scheduler = AgentScheduler()
    scheduler.cron("0 9 * * MON", agent=agent, prompt="Give me a Monday briefing.")
    scheduler.interval(300, agent=agent, prompt="System health check — report status.")

    async with scheduler:
        await asyncio.sleep(3600)  # run for an hour
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from .stream import MemoryStream
from .watch import CronWatch, DelayWatch, IntervalWatch, Watch, WatchCallback

if TYPE_CHECKING:
    from .agent import Agent, AgentReply

__all__ = ("AgentScheduler", "ScheduledJob")

logger = logging.getLogger(__name__)

# Callback invoked after each scheduled agent turn with its reply.
OnReplyCallback = Callable[["AgentReply[Any, Any]"], Awaitable[None]]

# Prompt can be a plain string, a zero-arg sync callable, or a zero-arg async callable.
PromptFactory = str | Callable[[], str] | Callable[[], Awaitable[str]]


class ScheduledJob:
    """A single agent-invocation job registered with :class:`AgentScheduler`.

    Obtain instances via :meth:`AgentScheduler.cron`, :meth:`~AgentScheduler.interval`,
    :meth:`~AgentScheduler.delay`, or :meth:`~AgentScheduler.add`.  Call
    :meth:`cancel` to stop this job without stopping the whole scheduler.
    """

    def __init__(
        self,
        watch: Watch,
        agent: Agent[Any],
        prompt: PromptFactory,
        on_reply: OnReplyCallback | None = None,
        *,
        name: str = "",
    ) -> None:
        self._watch = watch
        self._agent = agent
        self._prompt = prompt
        self._on_reply = on_reply
        self.name: str = name or watch.__class__.__name__

    @property
    def id(self) -> str:
        """Unique job identifier (delegates to the underlying Watch id)."""
        return self._watch.id

    @property
    def is_armed(self) -> bool:
        """``True`` while this job is running."""
        return self._watch.is_armed

    def arm(self, stream: MemoryStream) -> None:
        """Start the underlying watch. Called by :class:`AgentScheduler`."""
        self._watch.arm(stream, self._make_callback())

    def cancel(self) -> None:
        """Stop this job. The scheduler continues running other jobs."""
        self._watch.disarm()

    def _make_callback(self) -> WatchCallback:
        async def _run(events: list[Any], ctx: Any) -> None:
            prompt_text = await _resolve_prompt(self._prompt)
            try:
                reply = await self._agent.ask(prompt_text)
            except Exception:
                logger.exception("ScheduledJob %r: agent.ask() failed", self.name)
                return
            if self._on_reply is not None:
                try:
                    await self._on_reply(reply)
                except Exception:
                    logger.exception("ScheduledJob %r: on_reply callback failed", self.name)

        return _run


async def _resolve_prompt(prompt: PromptFactory) -> str:
    """Evaluate a prompt: string passthrough, or call a sync/async factory."""
    if isinstance(prompt, str):
        return prompt
    result = prompt()
    if hasattr(result, "__await__"):
        return await result  # type: ignore[return-value]
    return result  # type: ignore[return-value]


class AgentScheduler:
    """Schedule periodic :meth:`Agent.ask` calls using Watch primitives.

    Register jobs with :meth:`cron`, :meth:`interval`, or :meth:`delay`, then
    call :meth:`start` (or use as an async context manager) to arm them all.

    Example::

        scheduler = AgentScheduler()
        scheduler.interval(60, agent=agent, prompt="Ping.")

        async with scheduler:
            await asyncio.sleep(300)  # runs for 5 minutes

    Jobs added *after* :meth:`start` are armed immediately.
    """

    def __init__(self) -> None:
        self._jobs: list[ScheduledJob] = []
        # Internal stream required by Watch.arm(); not exposed to callers.
        self._stream: MemoryStream = MemoryStream()
        self._running: bool = False

    # ------------------------------------------------------------------
    # Job registration
    # ------------------------------------------------------------------

    def cron(
        self,
        expression: str,
        *,
        agent: Agent[Any],
        prompt: PromptFactory,
        on_reply: OnReplyCallback | None = None,
        name: str = "",
    ) -> ScheduledJob:
        """Schedule ``agent.ask(prompt)`` on a cron expression.

        Args:
            expression: Standard 5-field cron expression, e.g. ``"0 9 * * MON"``.
            agent: The agent to invoke.
            prompt: Message text (or a callable that produces it).
            on_reply: Optional async callback receiving the :class:`AgentReply`.
            name: Human-readable label for logs.

        Returns:
            The registered :class:`ScheduledJob`.

        Example::

            scheduler.cron("0 9 * * *", agent=agent, prompt="Daily briefing.")
        """
        job = ScheduledJob(CronWatch(expression), agent, prompt, on_reply, name=name or f"cron:{expression}")
        return self._register(job)

    def interval(
        self,
        seconds: float,
        *,
        agent: Agent[Any],
        prompt: PromptFactory,
        on_reply: OnReplyCallback | None = None,
        name: str = "",
    ) -> ScheduledJob:
        """Schedule ``agent.ask(prompt)`` every *seconds* seconds.

        Args:
            seconds: Interval in seconds between invocations.
            agent: The agent to invoke.
            prompt: Message text (or a callable that produces it).
            on_reply: Optional async callback receiving the :class:`AgentReply`.
            name: Human-readable label for logs.

        Returns:
            The registered :class:`ScheduledJob`.

        Example::

            scheduler.interval(300, agent=agent, prompt="Health check.")
        """
        job = ScheduledJob(IntervalWatch(seconds), agent, prompt, on_reply, name=name or f"interval:{seconds}s")
        return self._register(job)

    def delay(
        self,
        seconds: float,
        *,
        agent: Agent[Any],
        prompt: PromptFactory,
        on_reply: OnReplyCallback | None = None,
        name: str = "",
    ) -> ScheduledJob:
        """Schedule a one-shot ``agent.ask(prompt)`` after *seconds* seconds.

        The job auto-cancels after firing once.

        Args:
            seconds: Delay in seconds before the single invocation.
            agent: The agent to invoke.
            prompt: Message text (or a callable that produces it).
            on_reply: Optional async callback receiving the :class:`AgentReply`.
            name: Human-readable label for logs.

        Returns:
            The registered :class:`ScheduledJob`.

        Example::

            scheduler.delay(30, agent=agent, prompt="Follow up after 30 s.")
        """
        job = ScheduledJob(DelayWatch(seconds), agent, prompt, on_reply, name=name or f"delay:{seconds}s")
        return self._register(job)

    def add(
        self,
        watch: Watch,
        *,
        agent: Agent[Any],
        prompt: PromptFactory,
        on_reply: OnReplyCallback | None = None,
        name: str = "",
    ) -> ScheduledJob:
        """Register any :class:`~autogen.beta.watch.Watch` as a job trigger.

        Use this when the built-in convenience methods don't cover your
        trigger (e.g. :class:`~autogen.beta.watch.CadenceWatch` or a custom
        Watch implementation).

        Args:
            watch: Any armed-able Watch instance.
            agent: The agent to invoke.
            prompt: Message text (or a callable that produces it).
            on_reply: Optional async callback receiving the :class:`AgentReply`.
            name: Human-readable label for logs.

        Returns:
            The registered :class:`ScheduledJob`.
        """
        job = ScheduledJob(watch, agent, prompt, on_reply, name=name)
        return self._register(job)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Arm all registered jobs and begin scheduling.

        Idempotent — calling :meth:`start` on an already-running scheduler
        is a no-op.
        """
        if self._running:
            return
        self._running = True
        for job in self._jobs:
            job.arm(self._stream)

    async def stop(self) -> None:
        """Disarm all jobs and stop the scheduler."""
        self._running = False
        for job in self._jobs:
            job.cancel()

    async def __aenter__(self) -> AgentScheduler:
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _register(self, job: ScheduledJob) -> ScheduledJob:
        self._jobs.append(job)
        if self._running:
            job.arm(self._stream)
        return job
