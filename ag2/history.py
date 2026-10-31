# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, cast, overload

from ag2.types import ClassInfo

from .annotations import Context
from .context import StreamId
from .events import (
    BaseEvent,
    ModelResponse,
    ToolCallsEvent,
    ToolResult,
    ToolResultEvent,
    ToolResultsEvent,
)
from .events.conditions import Condition, TypeCondition

__all__ = (
    "HUMAN_INPUT_ABANDONED_TOOL_RESULT",
    "History",
    "HistoryQuery",
    "MemoryStorage",
    "Storage",
    "close_unanswered_tool_calls",
    "unanswered_tool_calls",
)

EventT = TypeVar("EventT", bound=BaseEvent, covariant=True)
SelectedEventT = TypeVar("SelectedEventT", bound=BaseEvent)


class Storage(Protocol):
    async def save_event(self, event: "BaseEvent", context: "Context") -> None: ...

    async def get_history(self, stream_id: "StreamId") -> Iterable["BaseEvent"]: ...

    async def set_history(self, stream_id: "StreamId", events: Iterable[BaseEvent]) -> None: ...

    async def drop_history(self, stream_id: "StreamId") -> None: ...


class MemoryStorage(Storage):
    def __init__(self) -> None:
        self.__data: defaultdict[StreamId, list[BaseEvent]] = defaultdict(list)

    async def save_event(self, event: "BaseEvent", context: "Context") -> None:
        self.__data[context.stream.id].append(event)

    async def get_history(self, stream_id: "StreamId") -> Iterable["BaseEvent"]:
        return self.__data[stream_id]

    async def set_history(self, stream_id: "StreamId", events: Iterable["BaseEvent"]) -> None:
        self.__data[stream_id] = list(events)

    async def drop_history(self, stream_id: "StreamId") -> None:
        self.__data.pop(stream_id, None)


class History:
    def __init__(self, stream_id: "StreamId", storage: Storage) -> None:
        self.stream_id = stream_id
        self.storage = storage

    async def get_events(self) -> Iterable["BaseEvent"]:
        return await self.storage.get_history(self.stream_id)

    async def replace(self, events: Iterable["BaseEvent"]) -> None:
        await self.storage.set_history(self.stream_id, events)

    @overload
    def where(self, condition: type[EventT]) -> "HistoryQuery[EventT]": ...

    @overload
    def where(self, condition: ClassInfo | Condition) -> "HistoryQuery[BaseEvent]": ...

    def where(self, condition: ClassInfo | Condition) -> "HistoryQuery[BaseEvent]":
        """Select retained top-level events without reading storage yet."""
        predicate = condition if isinstance(condition, Condition) else TypeCondition(condition)
        return HistoryQuery(self, predicate)


@dataclass(frozen=True, slots=True)
class HistoryQuery(Generic[EventT]):
    """A finite selection of retained history, evaluated afresh by each terminal.

    Queries preserve stored occurrence order, including repeated publications.
    They neither subscribe to future events nor flatten nested event batches.
    Events removed by a history retention policy are no longer available.
    """

    _history: History
    _condition: Condition

    @overload
    def where(self, condition: type[SelectedEventT]) -> "HistoryQuery[SelectedEventT]": ...

    @overload
    def where(self, condition: ClassInfo | Condition) -> "HistoryQuery[EventT]": ...

    def where(self, condition: ClassInfo | Condition) -> "HistoryQuery[BaseEvent]":
        """Return an independent query combining this selection with AND."""
        return HistoryQuery(self._history, self._condition & condition)

    async def all(self) -> list[EventT]:
        """Return a fresh list selected from one snapshot of stored events."""
        events = list(await self._history.get_events())
        return [cast(EventT, event) for event in events if self._condition(event)]

    async def first(self) -> EventT | None:
        """Return the first matching event, or None when no event matches."""
        events = await self.all()
        return events[0] if events else None

    async def last(self) -> EventT | None:
        """Return the last matching event, or None when no event matches."""
        events = await self.all()
        return events[-1] if events else None

    async def count(self) -> int:
        """Count matching retained occurrences, not lifetime events."""
        return len(await self.all())


# Stands in for a tool result nobody will ever produce, because the turn ended
# when a human-input request could not be answered. Lives here rather than with
# any one caller: both the turn boundary and the ACP tool gateway close a call
# off with it, and the model reads the same sentence either way.
HUMAN_INPUT_ABANDONED_TOOL_RESULT = (
    "This tool call was not completed: the turn ended because a human-input request could not be answered."
)


def _tool_call_batches(events: "Sequence[BaseEvent]") -> "list[ToolCallsEvent]":
    """Every batch of tool calls in ``events``, from either place they appear.

    A turn persists the model's :class:`ModelResponse` — which already carries
    ``tool_calls`` — *before* the agent emits the matching
    :class:`ToolCallsEvent`. Dying in that window leaves history holding calls
    that no ``ToolCallsEvent`` describes, so reading only the latter would find
    nothing to repair and leave the transcript with an unanswered call.

    Batches are deduplicated by call id, because the usual case is both records
    present and describing the same calls.
    """
    batches: list[ToolCallsEvent] = []
    seen: set[str] = set()
    for event in events:
        calls = (
            event.tool_calls.calls
            if isinstance(event, ModelResponse) and event.tool_calls
            else event.calls
            if isinstance(event, ToolCallsEvent)
            else []
        )
        fresh = [call for call in calls if call.id not in seen]
        if not fresh:
            continue
        seen.update(call.id for call in fresh)
        batches.append(ToolCallsEvent(fresh))
    return batches


def unanswered_tool_calls(events: "Sequence[BaseEvent]") -> set[str]:
    """Ids of the tool calls in ``events`` that no result answers, loose or wrapped.

    The question :func:`close_unanswered_tool_calls` does not ask, because a
    repair does not need it: it settles a batch only once a ``ToolResultsEvent``
    wraps it, which is right when closing a stopped turn — a partially completed
    batch has to be rebuilt whole. But a turn that ended *cleanly* through a
    ``final=True`` tool never gets that wrapper: it sends its forced
    ``ModelResponse`` and stops. Repairing such a history would append a fresh
    wrapper every time it was asked, rewriting a transcript that was already
    valid.

    So a caller that repairs speculatively — resuming a session it did not
    write, with no way to know how the last turn ended — asks this first. A
    loose result counts as an answer here precisely because it is the thing
    that distinguishes "finished without a wrapper" from "died mid-call".
    """
    answered = {e.parent_id for e in events if isinstance(e, ToolResultEvent)}
    answered.update(r.parent_id for e in events if isinstance(e, ToolResultsEvent) for r in e.results)
    return {call.id for batch in _tool_call_batches(events) for call in batch.calls if call.id not in answered}


async def close_unanswered_tool_calls(history: "History", *, result: str) -> int:
    """Close off tool calls a stopped turn left unanswered. Returns how many.

    A turn that stops mid-flight — cancelled, or failed — stops wherever it
    happened to be, which can be *between* a tool call and its result. That
    leaves history holding an assistant tool-call with nothing answering it —
    and providers reject that shape, so the conversation would fail on its next
    turn even though the history is supposed to stay usable.

    Appending a synthetic result per unanswered call keeps the transcript valid
    and tells the model plainly what happened, rather than rewriting history to
    pretend the call was never made. ``result`` is that stand-in text; each
    caller says why *its* turn stopped.

    Writes through :meth:`History.replace` rather than ``context.send``, because
    a ``ToolResultsEvent`` on the stream is what drives the next model call —
    repairing a dead turn must not start another one.
    """
    events = list(await history.get_events())

    # A batch counts as settled only once a ``ToolResultsEvent`` covers it —
    # that wrapper is what providers serialize. A loose ``ToolResultEvent`` left
    # behind by a tool that *did* finish is not enough on its own: a partially
    # completed batch would otherwise be rebuilt with the finished call missing,
    # and the transcript would carry more tool calls than tool results.
    settled = {r.parent_id for e in events if isinstance(e, ToolResultsEvent) for r in e.results}
    completed = {e.parent_id: e for e in events if isinstance(e, ToolResultEvent) and e.parent_id not in settled}

    repaired: list[ToolResultsEvent] = []
    closed = 0
    for event in _tool_call_batches(events):
        pending = [call for call in event.calls if call.id not in settled]
        if not pending:
            continue
        # Rebuild the *whole* outstanding batch: results that did land, plus a
        # stand-in for each call the turn cut short.
        results = []
        for call in pending:
            done = completed.get(call.id)
            results.append(
                done
                if done is not None
                else ToolResultEvent(
                    parent_id=call.id,
                    name=call.name,
                    result=ToolResult(result),
                )
            )
            closed += int(done is None)
        repaired.append(ToolResultsEvent(results))

    if not repaired:
        return 0

    await history.replace([*events, *repaired])
    return closed
