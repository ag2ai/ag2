# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Watch primitives — unified abstraction for all reactive behavior.

A Watch encapsulates a condition, internal state, check logic, and firing
semantics. It is the universal trigger for event monitoring, time-based
scheduling, and composite conditions.
"""

import asyncio
import datetime
import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from autogen.beta.annotations import Context
from autogen.beta.context import ConversationContext as ContextType
from autogen.beta.context import Stream, SubId
from autogen.beta.events import BaseEvent
from autogen.beta.events.conditions import Condition, TypeCondition
from autogen.beta.types import ClassInfo

logger = logging.getLogger(__name__)

WatchCallback = Callable[[list[BaseEvent], Context], Awaitable[None]]


@runtime_checkable
class Watch(Protocol):
    """A condition that can be armed on a stream and fires a callback when met."""

    @property
    def id(self) -> str: ...

    def arm(self, stream: Stream, callback: WatchCallback) -> None:
        """Start watching. Subscribe to relevant events or start timers."""
        ...

    def disarm(self) -> None:
        """Stop watching. Clean up subscriptions and timers."""
        ...

    @property
    def is_armed(self) -> bool: ...


class _BaseWatch:
    """Common base for watch implementations."""

    def __init__(self) -> None:
        self._id: str = uuid4().hex
        self._sub_id: SubId | None = None
        self._stream: Stream | None = None
        self._armed: bool = False

    @property
    def id(self) -> str:
        return self._id

    @property
    def is_armed(self) -> bool:
        return self._armed

    def disarm(self) -> None:
        if self._stream is not None and self._sub_id is not None:
            self._stream.unsubscribe(self._sub_id)
        self._sub_id = None
        self._stream = None
        self._armed = False


class EventWatch(_BaseWatch):
    """Fire immediately for each matching event.

    Example::

        EventWatch(ModelResponse)
        EventWatch(ToolCallEvent.name == "search")
    """

    def __init__(self, condition: ClassInfo | Condition) -> None:
        super().__init__()
        if not isinstance(condition, Condition):
            condition = TypeCondition(condition)
        self._condition = condition
        self._callback: WatchCallback | None = None

    def arm(self, stream: Stream, callback: WatchCallback) -> None:
        self._callback = callback
        self._stream = stream
        self._sub_id = stream.subscribe(self._handle_event, condition=self._condition)
        self._armed = True

    async def _handle_event(self, event: BaseEvent, ctx: Context) -> None:
        if self._callback is not None:
            await self._callback([event], ctx)


class CadenceWatch(_BaseWatch):
    """Fire when ``n`` matching events have been buffered or ``max_wait`` seconds
    have elapsed since the first buffered event — whichever comes first.

    At least one of ``n`` or ``max_wait`` must be set. Supplying both produces
    a size-or-time batch. Only ``n`` gives a pure count batch; only ``max_wait``
    gives a pure time window.

    Example::

        CadenceWatch(n=10)  # count-only batch
        CadenceWatch(max_wait=60)  # time-only window
        CadenceWatch(n=10, max_wait=60)  # size OR time
        CadenceWatch(n=5, condition=ModelResponse)
    """

    def __init__(
        self,
        n: int | None = None,
        max_wait: float | None = None,
        *,
        condition: ClassInfo | Condition | None = None,
    ) -> None:
        if n is None and max_wait is None:
            raise ValueError("CadenceWatch requires at least one of 'n' or 'max_wait'")
        if n is not None and n <= 0:
            raise ValueError(f"CadenceWatch 'n' must be positive, got {n}")
        if max_wait is not None and max_wait <= 0:
            raise ValueError(f"CadenceWatch 'max_wait' must be positive, got {max_wait}")
        super().__init__()
        self._n = n
        self._max_wait = max_wait
        self._condition: Condition | None = None
        if condition is not None:
            self._condition = condition if isinstance(condition, Condition) else TypeCondition(condition)
        self._buffer: list[BaseEvent] = []
        self._timer_task: asyncio.Task[Any] | None = None
        self._callback: WatchCallback | None = None

    def arm(self, stream: Stream, callback: WatchCallback) -> None:
        self._stream = stream
        self._callback = callback
        self._sub_id = stream.subscribe(self._handle_cadence_event, condition=self._condition)
        self._armed = True

    async def _handle_cadence_event(self, event: BaseEvent, ctx: Context) -> None:
        self._buffer.append(event)
        if self._n is not None and len(self._buffer) >= self._n:
            self._cancel_timer()
            await self._fire(ctx)
            return
        if self._max_wait is not None and (self._timer_task is None or self._timer_task.done()):
            assert self._stream is not None
            self._timer_task = asyncio.ensure_future(self._wait_and_fire(self._stream))

    async def _wait_and_fire(self, stream: Stream) -> None:
        assert self._max_wait is not None
        try:
            await asyncio.sleep(self._max_wait)
        except asyncio.CancelledError:
            return
        if self._buffer and self._callback is not None:
            ctx = ContextType(stream=stream)
            await self._fire(ctx)

    async def _fire(self, ctx: Context) -> None:
        if not self._buffer or self._callback is None:
            return
        batch = self._buffer[:]
        self._buffer.clear()
        await self._callback(batch, ctx)

    def _cancel_timer(self) -> None:
        if self._timer_task is not None:
            self._timer_task.cancel()
            self._timer_task = None

    def disarm(self) -> None:
        self._cancel_timer()
        self._buffer.clear()
        self._callback = None
        super().disarm()


class IntervalWatch(_BaseWatch):
    """Fire periodically at a fixed interval.

    Example::

        IntervalWatch(300)  # Every 5 minutes
    """

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self._seconds = seconds
        self._task: asyncio.Task[Any] | None = None
        self._callback: WatchCallback | None = None
        self._context: Context | None = None

    def arm(self, stream: Stream, callback: WatchCallback) -> None:
        self._stream = stream
        self._callback = callback
        self._armed = True
        self._task = asyncio.ensure_future(self._run(stream))

    async def _run(self, stream: Stream) -> None:
        while True:
            await asyncio.sleep(self._seconds)
            if self._callback is not None:
                try:
                    ctx = ContextType(stream=stream)
                    await self._callback([], ctx)
                except Exception:
                    logger.exception("IntervalWatch %s callback failed", self._id)

    def disarm(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self._callback = None
        super().disarm()


class DelayWatch(_BaseWatch):
    """Fire once after a delay.

    Example::

        DelayWatch(30)  # Fire after 30 seconds
    """

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self._seconds = seconds
        self._task: asyncio.Task[Any] | None = None

    def arm(self, stream: Stream, callback: WatchCallback) -> None:
        self._stream = stream
        self._armed = True
        self._task = asyncio.ensure_future(self._run(stream, callback))

    async def _run(self, stream: Stream, callback: WatchCallback) -> None:
        try:
            await asyncio.sleep(self._seconds)
            ctx = ContextType(stream=stream)
            await callback([], ctx)
        except Exception:
            logger.exception("DelayWatch %s callback failed", self._id)
        finally:
            # Auto-disarm after firing once (or on error)
            self.disarm()

    def disarm(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
        super().disarm()


class AllOf(_BaseWatch):
    """Fire when ALL sub-watches have fired at least once.

    Example::

        AllOf(
            EventWatch(DelegationResult.target == "monitor"),
            IntervalWatch(60),
        )
    """

    def __init__(self, *watches: Watch) -> None:
        super().__init__()
        self._watches = watches
        self._fired: set[str] = set()
        self._event_buffer: dict[str, list[BaseEvent]] = {}
        self._callback: WatchCallback | None = None

    def arm(self, stream: Stream, callback: WatchCallback) -> None:
        self._stream = stream
        self._callback = callback
        self._fired.clear()
        self._event_buffer.clear()
        self._armed = True
        for w in self._watches:
            w.arm(stream, functools.partial(self._handle_sub_watch, w.id))

    async def _handle_sub_watch(self, watch_id: str, events: list[BaseEvent], ctx: Context) -> None:
        self._fired.add(watch_id)
        self._event_buffer[watch_id] = events
        if len(self._fired) == len(self._watches) and self._callback is not None:
            combined: list[BaseEvent] = []
            for w in self._watches:
                combined.extend(self._event_buffer.get(w.id, []))
            self._fired.clear()
            self._event_buffer.clear()
            await self._callback(combined, ctx)

    def disarm(self) -> None:
        for w in self._watches:
            w.disarm()
        self._fired.clear()
        self._event_buffer.clear()
        self._callback = None
        super().disarm()


class AnyOf(_BaseWatch):
    """Fire when ANY sub-watch fires.

    Example::

        AnyOf(
            EventWatch(Signal.severity == "critical"),
            EventWatch(Signal.severity == "fatal"),
        )
    """

    def __init__(self, *watches: Watch) -> None:
        super().__init__()
        self._watches = watches
        self._callback: WatchCallback | None = None

    def arm(self, stream: Stream, callback: WatchCallback) -> None:
        self._stream = stream
        self._callback = callback
        self._armed = True
        for w in self._watches:
            w.arm(stream, self._handle_any)

    async def _handle_any(self, events: list[BaseEvent], ctx: Context) -> None:
        if self._callback is not None:
            await self._callback(events, ctx)

    def disarm(self) -> None:
        for w in self._watches:
            w.disarm()
        self._callback = None
        super().disarm()


class Sequence(_BaseWatch):
    """Fire when sub-watches fire in order.

    Each sub-watch must fire before the next one is armed.
    When the last sub-watch fires, the callback is invoked and the
    sequence resets.

    Example::

        Sequence(
            EventWatch(DelegationRequest),
            EventWatch(DelegationResult),
        )
    """

    def __init__(self, *watches: Watch) -> None:
        super().__init__()
        self._watches = watches
        self._current_index: int = 0
        self._callback: WatchCallback | None = None
        self._all_events: list[BaseEvent] = []

    def arm(self, stream: Stream, callback: WatchCallback) -> None:
        self._stream = stream
        self._callback = callback
        self._current_index = 0
        self._all_events.clear()
        self._armed = True
        self._arm_current()

    def _arm_current(self) -> None:
        if not self._armed or self._stream is None:
            return
        if self._current_index < len(self._watches):
            self._watches[self._current_index].arm(self._stream, self._step_handler)  # type: ignore[arg-type]

    async def _step_handler(self, events: list[BaseEvent], ctx: Context) -> None:
        if not self._armed:
            return
        self._all_events.extend(events)
        # Disarm current watch before advancing (guard for auto-disarming watches like DelayWatch)
        if self._watches[self._current_index].is_armed:
            self._watches[self._current_index].disarm()
        self._current_index += 1

        if self._current_index >= len(self._watches):
            # All watches have fired in sequence — callback first, then re-arm
            if self._callback is not None:
                collected = self._all_events[:]
                self._all_events.clear()
                self._current_index = 0
                await self._callback(collected, ctx)
                self._arm_current()  # Re-arm AFTER callback completes
        else:
            self._arm_current()

    def disarm(self) -> None:
        for w in self._watches:
            w.disarm()
        self._current_index = 0
        self._all_events.clear()
        self._callback = None
        super().disarm()


_DOW_NAMES = {"SUN": 0, "MON": 1, "TUE": 2, "WED": 3, "THU": 4, "FRI": 5, "SAT": 6}


def _parse_cron_field(spec: str, min_val: int, max_val: int, *, allow_names: bool = False) -> set[int]:
    values: set[int] = set()
    for part in spec.split(","):
        if part == "*":
            values.update(range(min_val, max_val + 1))
        elif "/" in part:
            base, step = part.split("/")
            start = min_val if base == "*" else int(base)
            values.update(range(start, max_val + 1, int(step)))
        elif "-" in part:
            lo, hi = part.split("-")
            values.update(range(int(lo), int(hi) + 1))
        elif allow_names and part.upper() in _DOW_NAMES:
            values.add(_DOW_NAMES[part.upper()])
        else:
            values.add(int(part))
    return values


class CronWatch(_BaseWatch):
    """Fire on a cron schedule expression.

    Uses a simple cron parser. Supports standard 5-field cron expressions:
    ``minute hour day-of-month month day-of-week``

    Example::

        CronWatch("0 9 * * MON")  # Every Monday at 9:00
        CronWatch("*/5 * * * *")  # Every 5 minutes
    """

    def __init__(self, expression: str) -> None:
        super().__init__()
        self._expression = expression
        self._task: asyncio.Task[Any] | None = None
        self._callback: WatchCallback | None = None

    def arm(self, stream: Stream, callback: WatchCallback) -> None:
        self._stream = stream
        self._callback = callback
        self._armed = True
        self._task = asyncio.ensure_future(self._run(stream))

    async def _run(self, stream: Stream) -> None:
        while True:
            now = datetime.datetime.now()
            next_fire = self._next_fire_time(now)
            delay = (next_fire - now).total_seconds()
            if delay > 0:
                await asyncio.sleep(delay)
            if self._callback is not None:
                try:
                    ctx = ContextType(stream=stream)
                    await self._callback([], ctx)
                except Exception:
                    logger.exception("CronWatch %s callback failed", self._id)

    def _next_fire_time(self, now: Any) -> Any:
        """Calculate next fire time from cron expression.

        Simple implementation: parses the 5 fields and finds the next
        matching minute from ``now``.
        """
        fields = self._expression.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid cron expression: {self._expression!r} (need 5 fields)")

        minute_spec, hour_spec, dom_spec, month_spec, dow_spec = fields

        minutes = _parse_cron_field(minute_spec, 0, 59)
        hours = _parse_cron_field(hour_spec, 0, 23)
        doms = _parse_cron_field(dom_spec, 1, 31)
        months = _parse_cron_field(month_spec, 1, 12)
        dow_set = {v % 7 for v in _parse_cron_field(dow_spec, 0, 7, allow_names=True)}  # 0=Sun..6=Sat; 7→0

        # Search forward from now + 1 minute
        candidate = now.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
        for _ in range(525960):  # Max ~1 year of minutes
            if (
                candidate.minute in minutes
                and candidate.hour in hours
                and candidate.day in doms
                and candidate.month in months
                and candidate.isoweekday() % 7 in dow_set
            ):
                return candidate
            candidate += datetime.timedelta(minutes=1)

        # Fallback: 1 hour from now
        return now + datetime.timedelta(hours=1)

    def disarm(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self._callback = None
        super().disarm()
