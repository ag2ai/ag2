# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Watch primitives — unified abstraction for all reactive behavior.

A Watch encapsulates a condition, internal state, check logic, and firing
semantics. It is the universal trigger for event monitoring, time-based
scheduling, and composite conditions.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from autogen.beta.annotations import Context
from autogen.beta.context import ConversationContext as ContextType
from autogen.beta.context import Stream, SubId
from autogen.beta.events import BaseEvent
from autogen.beta.events.conditions import ClassInfo, Condition, TypeCondition

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

    def arm(self, stream: Stream, callback: WatchCallback) -> None:
        async def _handler(event: BaseEvent, ctx: Context) -> None:
            await callback([event], ctx)

        self._stream = stream
        self._sub_id = stream.subscribe(_handler, condition=self._condition)
        self._armed = True


class BatchWatch(_BaseWatch):
    """Buffer N matching events, then fire with the entire batch.

    Example::

        BatchWatch(10, condition=ModelResponse)
    """

    def __init__(self, n: int, condition: ClassInfo | Condition | None = None) -> None:
        super().__init__()
        self._n = n
        self._condition: Condition | None = None
        if condition is not None:
            self._condition = condition if isinstance(condition, Condition) else TypeCondition(condition)
        self._buffer: list[BaseEvent] = []

    def arm(self, stream: Stream, callback: WatchCallback) -> None:
        async def _handler(event: BaseEvent, ctx: Context) -> None:
            self._buffer.append(event)
            if len(self._buffer) >= self._n:
                batch = self._buffer[:]
                self._buffer.clear()
                await callback(batch, ctx)

        self._stream = stream
        self._sub_id = stream.subscribe(_handler, condition=self._condition)
        self._armed = True

    def disarm(self) -> None:
        super().disarm()
        self._buffer.clear()


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
        import logging

        _logger = logging.getLogger(__name__)
        while True:
            await asyncio.sleep(self._seconds)
            if self._callback is not None:
                try:
                    ctx = ContextType(stream=stream)
                    await self._callback([], ctx)
                except Exception:
                    _logger.exception("IntervalWatch %s callback failed", self._id)

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
        import logging

        _logger = logging.getLogger(__name__)
        try:
            await asyncio.sleep(self._seconds)
            ctx = ContextType(stream=stream)
            await callback([], ctx)
        except Exception:
            _logger.exception("DelayWatch %s callback failed", self._id)
        finally:
            # Auto-disarm after firing once (or on error)
            self.disarm()

    def disarm(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
        super().disarm()


# ---------------------------------------------------------------------------
# Composite watches
# ---------------------------------------------------------------------------


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
            w.arm(stream, self._make_handler(w.id))

    def _make_handler(self, watch_id: str) -> WatchCallback:
        async def _handler(events: list[BaseEvent], ctx: Context) -> None:
            self._fired.add(watch_id)
            self._event_buffer[watch_id] = events
            if len(self._fired) == len(self._watches) and self._callback is not None:
                # Collect events from all sub-watches
                combined: list[BaseEvent] = []
                for w in self._watches:
                    combined.extend(self._event_buffer.get(w.id, []))
                self._fired.clear()
                self._event_buffer.clear()
                await self._callback(combined, ctx)

        return _handler

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

        async def _handler(events: list[BaseEvent], ctx: Context) -> None:
            if self._callback is not None:
                await self._callback(events, ctx)

        for w in self._watches:
            w.arm(stream, _handler)

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


# ---------------------------------------------------------------------------
# Advanced watches
# ---------------------------------------------------------------------------


class WindowWatch(_BaseWatch):
    """Collect events in a time window, then fire with the batch.

    Events matching the condition are buffered. After ``seconds`` elapse
    since the first buffered event, the callback fires with all collected
    events and the buffer resets.

    Example::

        WindowWatch(60, condition=ModelResponse)  # Batch responses over 60s
    """

    def __init__(self, seconds: float, condition: ClassInfo | Condition | None = None) -> None:
        super().__init__()
        self._seconds = seconds
        self._condition: Condition | None = None
        if condition is not None:
            self._condition = condition if isinstance(condition, Condition) else TypeCondition(condition)
        self._buffer: list[BaseEvent] = []
        self._timer_task: asyncio.Task[Any] | None = None
        self._callback: WatchCallback | None = None
        self._stream: Stream | None = None

    def arm(self, stream: Stream, callback: WatchCallback) -> None:
        self._stream = stream
        self._callback = callback

        async def _handler(event: BaseEvent, ctx: Context) -> None:
            self._buffer.append(event)
            # Start timer on first event in window
            if self._timer_task is None or self._timer_task.done():
                self._timer_task = asyncio.ensure_future(self._flush(stream))

        self._sub_id = stream.subscribe(_handler, condition=self._condition)
        self._armed = True

    async def _flush(self, stream: Stream) -> None:
        await asyncio.sleep(self._seconds)
        if self._buffer and self._callback is not None:
            batch = self._buffer[:]
            self._buffer.clear()
            ctx = ContextType(stream=stream)
            await self._callback(batch, ctx)

    def disarm(self) -> None:
        if self._timer_task is not None:
            self._timer_task.cancel()
            self._timer_task = None
        self._buffer.clear()
        self._callback = None
        if self._stream is not None and self._sub_id is not None:
            self._stream.unsubscribe(self._sub_id)
        self._sub_id = None
        self._stream = None
        self._armed = False


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
        import datetime
        import logging

        _logger = logging.getLogger(__name__)

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
                    _logger.exception("CronWatch %s callback failed", self._id)

    def _next_fire_time(self, now: Any) -> Any:
        """Calculate next fire time from cron expression.

        Simple implementation: parses the 5 fields and finds the next
        matching minute from ``now``.
        """
        import datetime

        fields = self._expression.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid cron expression: {self._expression!r} (need 5 fields)")

        minute_spec, hour_spec, dom_spec, month_spec, dow_spec = fields

        _dow_names = {"SUN": 0, "MON": 1, "TUE": 2, "WED": 3, "THU": 4, "FRI": 5, "SAT": 6}

        def _parse_field(spec: str, min_val: int, max_val: int, *, allow_names: bool = False) -> set[int]:
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
                elif allow_names and part.upper() in _dow_names:
                    values.add(_dow_names[part.upper()])
                else:
                    values.add(int(part))
            return values

        minutes = _parse_field(minute_spec, 0, 59)
        hours = _parse_field(hour_spec, 0, 23)
        doms = _parse_field(dom_spec, 1, 31)
        months = _parse_field(month_spec, 1, 12)
        dow_set = {v % 7 for v in _parse_field(dow_spec, 0, 7, allow_names=True)}  # 0=Sun..6=Sat; 7→0

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
