# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Hold a served agent's turn open across AG-UI exchanges.

Shared by `ag2.ag_ui.stream` and `ag2.a2ui.transports.ag_ui`. A held turn
lives in this process only: resumes must be routed back to it (sticky sessions
are required) and it does not survive a restart.
"""

import asyncio
import functools
import logging
import re
import secrets
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import inf
from typing import Any

from ag_ui.core import (
    PROTOCOL_VERSION,
    BaseEvent,
    EventType,
    Interrupt,
    Metadata,
    ReasoningEndEvent,
    ReasoningMessageEndEvent,
    ReasoningMessageStartEvent,
    ReasoningStartEvent,
    ResumeEntry,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedCancelledOutcome,
    RunFinishedEvent,
    RunFinishedInterruptOutcome,
    RunFinishedSuccessOutcome,
    RunStartedEvent,
    SubagentErrorEvent,
    SubagentFinishedEvent,
    SubagentFinishedSuspendedOutcome,
    SubagentStartedEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    TokenUsage,
    ToolCallChunkEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui.encoder import EventEncoder
from anyio import BrokenResourceError, ClosedResourceError, create_memory_object_stream
from anyio.streams.memory import MemoryObjectSendStream
from typing_extensions import assert_never

from ag2.annotations import Context
from ag2.events import BaseEvent as AG2Event
from ag2.events import HumanInputRequest, HumanMessage, ToolApprovalRequest, UsageEvent
from ag2.exceptions import AG2Error, HumanInputTimeoutError

from .run_input import strip_unrecognised
from .usage import map_usage_events_to_ag_ui

logger = logging.getLogger(__name__)

# What an `Interrupt` raised by `context.input()` says it is. The
# protocol leaves `reason` a free string; a tool call held for approval says
# so separately, because a client renders the two differently — a text box
# against a pair of buttons — and should not have to read the prose to tell.
INPUT_REQUIRED_REASON = "input_required"
TOOL_CALL_REASON = "tool_call"

# `context.input()` is one string in, one string out, so this is the whole of
# what an answer may be. Declared so a client can tell a refused payload from a
# refused interrupt.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "string",
    "title": "Answer",
    "description": "Your answer to the agent's question.",
}

# An approval is still answered as a string — the middleware waiting on it reads
# words like "always" — but a client that drew two buttons has a boolean in
# hand and should not have to know which words this server accepts.
APPROVAL_SCHEMA: dict[str, Any] = {
    "type": ["string", "boolean"],
    "title": "Approval",
    "description": "true to let the tool call go ahead, false to refuse it.",
}

# Codes on the `RUN_ERROR` a refused resume produces, so a client can branch
# without parsing prose.
NOT_COVERED = "INTERRUPT_NOT_COVERED"
PAYLOAD_REFUSED = "INTERRUPT_PAYLOAD_REFUSED"
NOT_PROVEN = "INTERRUPT_NOT_PROVEN"

# The code on the lone `RUN_ERROR` a client declaring another major version of
# the protocol gets, before any run starts.
UNSUPPORTED_PROTOCOL_VERSION = "UNSUPPORTED_PROTOCOL_VERSION"

# The two-component grammar a declared protocol version is read in.
_VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")

# Where this server's own envelope data sits inside a metadata object. Not
# `ag_ui.core.AGUI_METADATA_KEY` (`"ag-ui"`): the protocol reserves that one
# for itself, and every other key is user space.
AG2_METADATA_KEY = "ag2"

# The proof, inside that envelope, that a resume comes from whoever the question
# was put to.
PROOF_KEY = "proof"

# Bytes of randomness behind one proof. Beyond guessing, and short enough to sit
# in a request body without comment.
_PROOF_BYTES = 32


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp_ms() -> int:
    """Now, as the wire spells a timestamp."""
    return int(time.time() * 1000)


@dataclass(frozen=True, slots=True)
class Retention:
    """How long an unanswered question is kept, and how many at once."""

    ttl: float = 900.0
    """Seconds a held turn survives unanswered, after which it is cancelled.

    Also sent to the client as the interrupt's `expiresAt`.
    """

    max_held: int = 128
    """Held turns allowed at once. Holding past this one cancels the oldest."""

    def __post_init__(self) -> None:
        # Rejected here rather than surprising an operator one held turn later:
        # `max_held=0` would have `hold` evict the turn it just took, and a
        # non-positive `ttl` would advertise an `expiresAt` already in the past.
        if self.ttl <= 0:
            raise ValueError(f"Retention.ttl must be positive, got {self.ttl}")
        if self.max_held < 1:
            raise ValueError(f"Retention.max_held must be at least 1, got {self.max_held}")


DEFAULT_RETENTION = Retention()


class ResumeRefusedError(AG2Error):
    """A resume this server will not honour, reaching the client as `RUN_ERROR`."""

    __slots__ = ("code",)

    code: str
    """One of `NOT_COVERED`, `NOT_PROVEN`, `PAYLOAD_REFUSED`.

    Branch on this rather than on the message.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code

    def as_event(self, timestamp: int | None = None) -> RunErrorEvent:
        return RunErrorEvent(message=str(self), code=self.code, timestamp=timestamp)


class TurnOutput:
    """Where a held turn's events go, and under which run.

    Rebound on each exchange that carries the turn, so its lifecycle events
    always name the current run. Between the interrupt that pauses the turn and
    the exchange that resumes it, events are kept and sent first on resume: the
    turn keeps working while it waits — a sibling tool call finishing — and
    none of that may be lost. Sending into an exchange that closed otherwise
    (the client went away mid-run) drops the event rather than failing the turn.

    Also the ledger of what the turn has open on the wire — messages, reasoning,
    tool calls, subagent invocations — read off the events as they are sent, so
    a run can be closed in order however it ends, and of the tool calls the
    turn has left unanswered.

    And the meter of what the turn spent: each run reports what was spent since
    the run before it reported, so a turn carried by several runs is counted once.
    """

    __slots__ = (
        "thread_id",
        "run_id",
        "usage",
        "_metered",
        "_send",
        "_paused",
        "_kept",
        "_open",
        "_announced",
        "_reannounce",
        "_repeats",
        "_calls",
    )

    def __init__(self, *, thread_id: str, run_id: str, send: MemoryObjectSendStream[BaseEvent]) -> None:
        self.thread_id = thread_id
        self.run_id = run_id
        self.usage: list[UsageEvent] = []
        """Everything the turn has spent, as it is spent. The transport running it appends here."""
        self._metered = 0
        """How much of `usage` a run has already reported."""
        self._send = send
        self._paused = False
        self._kept: list[BaseEvent] = []
        self._open: dict[tuple[EventType, str], BaseEvent] = {}
        """Every entity opened and not yet closed, by (kind, id), in opening order."""
        self._announced: set[str] = set()
        """Every subagent invocation id this turn has announced."""
        self._reannounce: list[SubagentStartedEvent] = []
        """Invocations closed as suspended, announced again first thing on resume."""
        self._repeats: dict[str, int] = {}
        """Starts refused for an invocation id still open, each owed one silenced end."""
        self._calls: dict[str, bool] = {}
        """Tool calls the turn started, in call order, and whether it answered them."""

    def rebind(self, *, run_id: str, send: MemoryObjectSendStream[BaseEvent]) -> None:
        """Point the turn at the exchange now carrying it."""
        self.run_id = run_id
        self._send = send
        self._paused = False
        # What an earlier run of this turn left unanswered stays owed: the run
        # that paused did not end as a success, so it reported none of it.
        for event in self._kept:
            self._record_call(event)
        # Ahead of anything kept: an event about a suspended invocation must
        # follow its re-announcement, and the invocation is still open.
        self._kept[:0] = self._take_reannouncements()

    @property
    def paused(self) -> bool:
        """Whether the turn is waiting on a question, its run already ending or ended."""
        return self._paused

    def take_usage(self) -> list[TokenUsage] | None:
        """What the turn spent since a run last reported, for the run reporting now."""
        spent, self._metered = self.usage[self._metered :], len(self.usage)
        return map_usage_events_to_ag_ui(spent)

    def is_open_subagent(self, subagent_run_id: str | None) -> bool:
        """Whether `subagent_run_id` names an invocation open on the wire."""
        return subagent_run_id is not None and (EventType.SUBAGENT_STARTED, subagent_run_id) in self._open

    def success_outcome(self) -> RunFinishedSuccessOutcome:
        """The outcome of the current run finishing, naming the calls it left for the client."""
        # Stated on every run, not only on interrupts: the protocol reads an
        # omitted outcome as a producer predating the interrupt-aware lifecycle,
        # and declaring the capability while behaving as one is not a described
        # state. An empty list is left out, since it would claim nothing.
        pending = [call_id for call_id, answered in self._calls.items() if not answered]
        return RunFinishedSuccessOutcome(pending_tool_call_ids=pending or None)

    async def pause(self, interrupt: RunFinishedEvent) -> None:
        """End the current exchange on `interrupt`, keeping what follows for the next."""
        # Paused before anything is awaited: each send below can wait for the
        # exchange to read it, and whatever the turn sends meanwhile — a sibling
        # delegation finishing, or a new one starting — belongs to the run that
        # resumes it, not between the closes below and the RUN_FINISHED.
        self._paused = True
        owed, self._kept = self._kept, []
        # No run finishes with an invocation open, so each is closed as
        # suspended — naming the interrupts it raised itself — and announced
        # again when the turn resumes. It stays open in the ledger meanwhile,
        # so an end sent while paused is kept for after its re-announcement.
        suspended = [e for e in self._open.values() if isinstance(e, SubagentStartedEvent)]
        self._reannounce.extend(suspended)
        for event in owed:
            await self._deliver(event)
        interrupts = interrupt.outcome.interrupts if isinstance(interrupt.outcome, RunFinishedInterruptOutcome) else []
        for started in suspended:
            raised = [i.id for i in interrupts if i.subagent_run_id == started.subagent_run_id]
            await self._deliver(
                SubagentFinishedEvent(
                    subagent_run_id=started.subagent_run_id,
                    outcome=SubagentFinishedSuspendedOutcome(interrupt_ids=raised or None),
                    timestamp=timestamp_ms(),
                )
            )
        await self._deliver(interrupt)

    async def succeed(self) -> None:
        """Finish the current run as a success, closing first whatever it left open."""
        await self._close_open("the run ended before this invocation finished")
        await self.send(
            RunFinishedEvent(
                thread_id=self.thread_id,
                run_id=self.run_id,
                timestamp=timestamp_ms(),
                usage=self.take_usage(),
                outcome=self.success_outcome(),
            )
        )

    async def stop(self) -> None:
        """Finish the current run as cancelled, because the server stopped it."""
        await self._close_open("the run was cancelled before this invocation finished")
        await self.send(
            RunFinishedEvent(
                thread_id=self.thread_id,
                run_id=self.run_id,
                timestamp=timestamp_ms(),
                usage=self.take_usage(),
                outcome=RunFinishedCancelledOutcome(),
            )
        )

    async def fail(self, error: Exception) -> None:
        """End the current run on `error`. Nothing needs closing: the client abandons it all."""
        await self.send(RunErrorEvent(message=repr(error), timestamp=timestamp_ms(), usage=self.take_usage()))

    async def _close_open(self, message: str) -> None:
        # Every start refused as a repeat is moot once its invocation is closed here.
        self._repeats = {}
        for kind, entity_id in reversed(list(self._open)):
            await self.send(_closing(kind, entity_id, message))

    async def send(self, event: BaseEvent) -> None:
        if not self._admit(event):
            return
        if self._paused:
            self._kept.append(event)
            return
        while self._kept:
            await self._deliver(self._kept.pop(0))
        await self._deliver(event)

    def abandon(self) -> list[BaseEvent]:
        """Everything owed to the run that abandons this paused turn, short of its end.

        What the turn kept while paused, then an end for each entity still
        open, innermost first. Call once the turn has stopped sending.
        """
        owed = [*self._take_reannouncements(), *self._kept]
        owed.extend(
            _closing(kind, entity_id, "the run was cancelled before this invocation finished")
            for kind, entity_id in reversed(self._open)
        )
        self._kept, self._open, self._repeats = [], {}, {}
        return owed

    def _take_reannouncements(self) -> list[BaseEvent]:
        again: list[BaseEvent] = [e.model_copy(update={"timestamp": timestamp_ms()}) for e in self._reannounce]
        self._reannounce = []
        return again

    def _admit(self, event: BaseEvent) -> bool:
        """Record what `event` opens or closes, and whether it may go out at all."""
        # Tracked as sent, not as delivered: kept events are delivered in the
        # order they were sent, so the ledger is what the wire will have seen.
        match event:
            case SubagentStartedEvent():
                if event.subagent_run_id in self._announced:
                    logger.warning(
                        "not announcing subagent invocation %s again: this turn has already announced it",
                        event.subagent_run_id,
                    )
                    # Ends under one id cannot be told apart, so the invocation
                    # stays open until the last delegation under it has ended.
                    if (EventType.SUBAGENT_STARTED, event.subagent_run_id) in self._open:
                        self._repeats[event.subagent_run_id] = self._repeats.get(event.subagent_run_id, 0) + 1
                    return False
                self._announced.add(event.subagent_run_id)
                self._open[EventType.SUBAGENT_STARTED, event.subagent_run_id] = event
            case SubagentFinishedEvent() | SubagentErrorEvent() if event.subagent_run_id in self._repeats:
                if (left := self._repeats.pop(event.subagent_run_id) - 1) > 0:
                    self._repeats[event.subagent_run_id] = left
                return False
            case SubagentFinishedEvent() | SubagentErrorEvent():
                if self._open.pop((EventType.SUBAGENT_STARTED, event.subagent_run_id), None) is None:
                    logger.warning(
                        "not ending subagent invocation %s: it is not open on the wire", event.subagent_run_id
                    )
                    return False
            case TextMessageStartEvent():
                self._open[EventType.TEXT_MESSAGE_START, event.message_id] = event
            case TextMessageEndEvent():
                self._open.pop((EventType.TEXT_MESSAGE_START, event.message_id), None)
            case ReasoningStartEvent():
                self._open[EventType.REASONING_START, event.message_id] = event
            case ReasoningEndEvent():
                self._open.pop((EventType.REASONING_START, event.message_id), None)
            case ReasoningMessageStartEvent():
                self._open[EventType.REASONING_MESSAGE_START, event.message_id] = event
            case ReasoningMessageEndEvent():
                self._open.pop((EventType.REASONING_MESSAGE_START, event.message_id), None)
            case ToolCallStartEvent():
                self._open[EventType.TOOL_CALL_START, event.tool_call_id] = event
            case ToolCallEndEvent():
                self._open.pop((EventType.TOOL_CALL_START, event.tool_call_id), None)
        self._record_call(event)
        return True

    def _record_call(self, event: BaseEvent) -> None:
        match event:
            case ToolCallStartEvent() | ToolCallChunkEvent() if event.tool_call_id is not None:
                self._calls.setdefault(event.tool_call_id, False)
            case ToolCallResultEvent() if event.tool_call_id in self._calls:
                self._calls[event.tool_call_id] = True

    async def _deliver(self, event: BaseEvent) -> None:
        try:
            await self._send.send(event)
        except (BrokenResourceError, ClosedResourceError):
            logger.debug("dropping %s: no AG-UI exchange is carrying this turn", type(event).__name__)

    async def aclose(self) -> None:
        await self._send.aclose()


def _closing(kind: EventType, entity_id: str, message: str) -> BaseEvent:
    """The event that ends an open entity, for a run ending before it did.

    `message` is what an invocation closed this way is told.
    """
    now = timestamp_ms()
    match kind:
        case EventType.SUBAGENT_STARTED:
            return SubagentErrorEvent(subagent_run_id=entity_id, message=message, timestamp=now)
        case EventType.TEXT_MESSAGE_START:
            return TextMessageEndEvent(message_id=entity_id, timestamp=now)
        case EventType.REASONING_START:
            return ReasoningEndEvent(message_id=entity_id, timestamp=now)
        case EventType.REASONING_MESSAGE_START:
            return ReasoningMessageEndEvent(message_id=entity_id, timestamp=now)
        case EventType.TOOL_CALL_START:
            return ToolCallEndEvent(tool_call_id=entity_id, timestamp=now)
    raise AssertionError(f"no closing event for a {kind}")


class ServedTurn:
    """One AG-UI-served agent turn, owned by the server rather than by a request.

    Launched through `start`; from then on its lifetime is `ServedTurns`', so it
    survives the exchange that started it.
    """

    __slots__ = ("output", "asking", "_task", "_outstanding", "_answer")

    def __init__(self, output: TurnOutput) -> None:
        self.output = output
        self.asking = asyncio.Lock()
        """Taken by `ServedTurns.ask` for the question being put."""
        self._task: asyncio.Task[None] | None = None
        self._outstanding: Interrupt | None = None
        self._answer: asyncio.Future[str] | None = None

    @property
    def thread_id(self) -> str:
        return self.output.thread_id

    @property
    def outstanding(self) -> Interrupt | None:
        """The interrupt this turn is waiting on, if it is waiting on one."""
        return self._outstanding

    def start(self, run: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        """Launch the turn. Separate from `__init__`: the coroutine needs this object."""
        task = asyncio.ensure_future(run)
        # A turn that fails with nobody awaiting it — the exchange that started
        # it is long over — would otherwise be reported as an unretrieved
        # exception when the task is finalised.
        task.add_done_callback(_consume_exception)
        self._task = task
        return task

    async def settle(self) -> None:
        """Wait for the turn to end.

        Whatever it raised is not raised here: the turn reported that on the
        wire itself. Cancelled while waiting, it cancels the turn too, since
        nothing else is carrying it. Never call this on a *held* turn: nothing
        will end it but the answer that has not arrived.
        """
        assert self._task is not None, "settle() before start()"
        try:
            await asyncio.wait({self._task})
        except asyncio.CancelledError:
            self.release()
            raise

    def suspend(self, interrupt: Interrupt, answer: "asyncio.Future[str]") -> None:
        """Park the turn on `interrupt` until `answer` is resolved.

        Called by `ServedTurns.ask` under `asking`: one question is outstanding per turn.
        """
        assert self._outstanding is None, "suspend() while another question is outstanding"
        self._outstanding = interrupt
        self._answer = answer

    def wake(self) -> None:
        """Forget the question, however the wait ended."""
        self._outstanding = None
        self._answer = None

    def deliver(self, payload: str) -> None:
        """Hand `payload` to the waiting call."""
        assert self._answer is not None, "deliver() on a turn that is not waiting"
        # Forgotten synchronously, not in the waiting coroutine's own cleanup:
        # that coroutine does not resume until the loop next runs it, and until
        # then the turn must already read as no longer waiting, or the next
        # exchange re-reports the question it just answered.
        answer, self._answer, self._outstanding = self._answer, None, None
        answer.set_result(payload)

    def release(self) -> None:
        """Cancel the turn, wherever it is suspended."""
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def aclose(self) -> None:
        """Cancel the turn and wait for it to unwind."""
        self.release()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)


def _consume_exception(task: "asyncio.Task[Any]") -> None:
    if not task.cancelled():
        task.exception()


class ServedTurns:
    """Every turn this process is running, and which of them are held.

    Held turns are keyed by thread, not by run: a resume carries a new run id
    under the same thread. Membership is also what keeps a turn alive — a bare
    `asyncio` task nobody references may be collected mid-flight.
    """

    __slots__ = ("_live", "_held", "retention", "require_proof", "_now")

    def __init__(
        self,
        *,
        retention: Retention = DEFAULT_RETENTION,
        require_proof: bool = False,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        self._live: set[ServedTurn] = set()
        self._held: dict[str, ServedTurn] = {}
        self.retention = retention
        self.require_proof = require_proof
        self._now = now

    def track(self, turn: ServedTurn, task: "asyncio.Task[None]") -> None:
        """Own `turn`'s lifetime until its task completes."""
        self._live.add(turn)
        task.add_done_callback(_Discard(self, turn))

    async def ask(self, turn: ServedTurn, interrupt_for: "Callable[[datetime], Interrupt]") -> str:
        """Put a question to the client, hold `turn`, and return the answer.

        Concurrent askers on one turn — parallel tool calls, parallel subtasks —
        are put one at a time: each next question is the outcome of the run
        that answers the one before. `interrupt_for` builds the question once
        its turn comes, from when the caller started waiting, so the deadline
        it advertises is current.
        """
        since = self._now()
        async with turn.asking:
            return await self._put(turn, interrupt_for(since))

    async def _put(self, turn: ServedTurn, interrupt: Interrupt) -> str:
        answer: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        turn.suspend(interrupt, answer)
        # Held before the question is emitted, never after: the exchange ends on
        # that very event, so a resume can be in flight the moment it lands, and
        # a turn not yet held is unreachable by retrieval and eviction.
        self.hold(turn)
        seconds = _seconds_until(interrupt, self._now())
        try:
            await turn.output.pause(
                RunFinishedEvent(
                    thread_id=turn.output.thread_id,
                    run_id=turn.output.run_id,
                    timestamp=timestamp_ms(),
                    # Spent up to the pause: the run resuming the turn reports
                    # only what it spends itself.
                    usage=turn.output.take_usage(),
                    outcome=RunFinishedInterruptOutcome(interrupts=[interrupt]),
                )
            )
            # The deadline the client was shown is kept here, by the call that
            # advertised it. A held turn is a suspended coroutine, so the
            # coroutine is what has to stop waiting: nothing else knows when, and
            # a registry swept only by later traffic would hold an unanswered
            # question for as long as the process stayed quiet.
            return await asyncio.wait_for(answer, seconds)
        except asyncio.TimeoutError as error:  # a separate class from the builtin on Python 3.10
            # The same exception `context.input(timeout=)` raises, because it is
            # the same event: `deadline()` advertised whichever bound fell first,
            # and the caller should not have to tell them apart.
            assert seconds is not None, "an interrupt with no deadline cannot time out"
            raise HumanInputTimeoutError(seconds) from error
        finally:
            turn.wake()
            self.discard(turn)

    def hold(self, turn: ServedTurn) -> None:
        """Hold `turn` for its thread, evicting whatever that thread held before.

        Stamps the retention clock, so each new question buys a full bound.
        """
        previous = self._held.get(turn.thread_id)
        if previous is not None and previous is not turn:
            logger.info("releasing a turn held for thread %s: it has been superseded", turn.thread_id)
            previous.release()
        self._held[turn.thread_id] = turn
        self._evict_expired()
        while len(self._held) > self.retention.max_held:
            # Nearest its deadline, rather than longest held: under one `ttl` the
            # two are the same turn, and where callers pass their own `timeout`
            # this drops the one with least life left. Read off the interrupt,
            # which `restore` does not touch, so a stream of refused resumes
            # cannot move a turn down the queue.
            soonest = min(self._held.values(), key=_deadline_of)
            logger.warning(
                "releasing the held AG-UI turn nearest its deadline: %d already held", self.retention.max_held
            )
            del self._held[soonest.thread_id]
            soonest.release()

    def take(self, thread_id: str) -> "ServedTurn | None":
        """Remove and return the turn held for `thread_id`, or `None`.

        Removed rather than looked up, so two resumes racing one thread cannot
        both drive it. An expired turn is still returned — the caller decides
        what to tell a late answer — while everything else aged out is reclaimed.
        """
        turn = self._held.pop(thread_id, None)
        self._evict_expired()
        return turn

    def restore(self, turn: ServedTurn) -> None:
        """Put back a turn taken for a resume that was refused.

        Not `hold`: the turn keeps the deadline it was holding on, so a stream of
        bad resumes can neither extend its life nor move it down the queue for
        eviction at another tenant's expense.
        """
        self._held[turn.thread_id] = turn

    def discard(self, turn: ServedTurn) -> None:
        """Stop holding `turn`, if this is still the turn its thread holds."""
        if self._held.get(turn.thread_id) is turn:
            del self._held[turn.thread_id]

    def retire(self, turn: ServedTurn) -> None:
        """Forget `turn` entirely, because its task has ended.

        Held as well as live: a turn that dies while held — cancelled, or failed
        somewhere its own `ask` could not clean up after — would otherwise leave
        an entry no resume can do anything with but rebind to a task that is gone.
        """
        self._live.discard(turn)
        self.discard(turn)

    def holds(self, thread_id: str) -> bool:
        """Whether `thread_id` holds a question that can still be answered."""
        self._evict_expired()
        return thread_id in self._held

    async def release_all(self) -> None:
        """Cancel every turn this process is running, and wait for them to unwind.

        Awaited rather than fired off, so a turn's cleanup runs before the
        process holding it goes.
        """
        self._held.clear()
        live, self._live = tuple(self._live), set()
        await asyncio.gather(*(turn.aclose() for turn in live), return_exceptions=True)

    def expired(self, turn: ServedTurn) -> bool:
        """Whether the question `turn` is held on can still be answered."""
        return _expired(turn, self._now())

    def deadline(self, timeout: float | None = None, *, since: datetime | None = None) -> datetime:
        """When an interrupt raised now stops being answerable.

        The earlier of :attr:`Retention.ttl` from now and `timeout` from `since`
        (default now), the caller's own bound — clients reject late answers
        locally against this figure, so it has to be the one that will in fact
        apply. `since` is when the caller started waiting: a question queued
        behind another has spent part of its `timeout` before it is put.
        """
        now = self._now()
        retained = now + timedelta(seconds=self.retention.ttl)
        if timeout is None:
            return retained
        return min(retained, (since or now) + timedelta(seconds=timeout))

    def _evict_expired(self) -> None:
        now = self._now()
        for thread_id, turn in [(t, h) for t, h in self._held.items() if _expired(h, now)]:
            logger.info("releasing an expired held AG-UI turn for thread %s", thread_id)
            del self._held[thread_id]
            turn.release()


def _deadline_of(turn: ServedTurn) -> float:
    """When `turn`'s question stops being answerable, as a POSIX timestamp.

    `inf` for a turn waiting on nothing, or on an interrupt with no deadline:
    the protocol reads an absent `expiresAt` as a promise never to expire, and
    that promise is infinity rather than a date distant enough to pass for it.
    Comparable and orderable against any other deadline, which is all either
    caller needs.
    """
    outstanding = turn.outstanding
    if outstanding is None or outstanding.expires_at is None:
        return inf
    return _parse(outstanding.expires_at).timestamp()


def _seconds_until(interrupt: Interrupt, now: datetime) -> float | None:
    """How long there is left to answer `interrupt`, or `None` if it never expires."""
    if interrupt.expires_at is None:
        return None
    return max((_parse(interrupt.expires_at) - now).total_seconds(), 0.0)


def _expired(turn: ServedTurn, now: datetime) -> bool:
    return _deadline_of(turn) <= now.timestamp()


def _parse(expires_at: str) -> datetime:
    return datetime.fromisoformat(expires_at)


class _Discard:
    # A done-callback retiring one turn from the registry. A class rather than a
    # closure: `track` runs once per turn, which is a runtime execution path.

    __slots__ = ("_turns", "_turn")

    def __init__(self, turns: "ServedTurns", turn: ServedTurn) -> None:
        self._turns = turns
        self._turn = turn

    def __call__(self, _task: "asyncio.Task[Any]") -> None:
        self._turns.retire(self._turn)


class ClientInterrupter:
    """Answers a served agent's `context.input()` from the human at the AG-UI client.

    Registered as the turn's stream interrupter, and only when no human-input
    hook was supplied: a caller who passed one keeps answering in process.
    """

    __slots__ = ("_turn", "_turns")

    def __init__(self, turn: ServedTurn, turns: ServedTurns) -> None:
        self._turn = turn
        self._turns = turns

    async def __call__(self, event: HumanInputRequest, context: Context) -> "AG2Event | None":
        answer = await self._turns.ask(self._turn, functools.partial(self.interrupt_for, event))
        await context.send(HumanMessage.ensure_message(answer, parent_id=event.id))
        return None

    def interrupt_for(self, event: HumanInputRequest, since: datetime | None = None) -> Interrupt:
        """The wire interrupt for one human-input request, under the request's own id.

        `since` is when the request started waiting; see `ServedTurns.deadline`.
        """
        # A request gating a tool call says so and names the call, so a client
        # can offer buttons rather than a text box without parsing the prose.
        approval = event if isinstance(event, ToolApprovalRequest) else None
        # Tagged only with an invocation the client was told about: a transport
        # that does not announce delegations has nothing to attribute it to.
        output = self._turn.output
        return Interrupt(
            subagent_run_id=event.task_id if output.is_open_subagent(event.task_id) else None,
            id=event.id,
            reason=INPUT_REQUIRED_REASON if approval is None else TOOL_CALL_REASON,
            message=event.content,
            tool_call_id=None if approval is None else approval.tool_call_id,
            response_schema=ANSWER_SCHEMA if approval is None else APPROVAL_SCHEMA,
            expires_at=self._turns.deadline(event.timeout, since=since).isoformat(),
            metadata={AG2_METADATA_KEY: {PROOF_KEY: issue_proof()}},
        )


def issue_proof() -> str:
    """A fresh secret, left with one interrupt and asked for on its answer."""
    # A capability, not a signature: the question it proves is suspended in this
    # process, so the value to compare against is in memory beside it — no key,
    # nothing to rotate. Per interrupt, so a leaked proof does not carry over.
    return secrets.token_urlsafe(_PROOF_BYTES)


def check_proof(entry: ResumeEntry, interrupt: Interrupt, *, required: bool) -> None:
    """Verify the proof `entry` carries against the one `interrupt` was issued with.

    A proof that is present must be the one issued. One that is absent is
    refused only when `required`: a standard client copies nothing from an
    interrupt's metadata, and the protocol does not ask a producer to check
    what it was told to keep.

    Raises `ResumeRefusedError` under `NOT_PROVEN` if the check fails.
    """
    if not required and not _has_proof(entry.metadata):
        return
    if not secrets.compare_digest(_proof_in(entry.metadata), _proof_in(interrupt.metadata)):
        raise ResumeRefusedError(
            NOT_PROVEN,
            f"the resume for interrupt {interrupt.id} does not carry the proof it was issued with",
        )


def _has_proof(metadata: Metadata | None) -> bool:
    envelope = (metadata or {}).get(AG2_METADATA_KEY)
    return isinstance(envelope, dict) and PROOF_KEY in envelope


def _proof_in(metadata: Metadata | None) -> bytes:
    # A malformed proof and a missing one are one case when a proof is required:
    # both mean nothing was proved, and telling them apart would only say which
    # half to fix. Bytes, not str: a proof off the wire is arbitrary text and
    # `compare_digest` raises TypeError on a non-ASCII str rather than reporting
    # a mismatch.
    envelope = (metadata or {}).get(AG2_METADATA_KEY)
    proof = envelope.get(PROOF_KEY) if isinstance(envelope, dict) else None
    return proof.encode("utf-8", "surrogatepass") if isinstance(proof, str) else b""


def resume_entry(incoming: RunAgentInput) -> "list[ResumeEntry]":
    """The resume entries this run carries. Empty means an ordinary new run."""
    return list(incoming.resume or ())


def answer_from(entry: ResumeEntry, interrupt: Interrupt) -> str:
    """The answer `entry` carries, as the string the waiting call will read.

    A `bool` is accepted for an approval and only for an approval; anything else
    raises `ResumeRefusedError` under `PAYLOAD_REFUSED`.
    """
    # The client drew two buttons and has a bool; the approval middleware reads
    # words. Translating here beats teaching either side the other's vocabulary.
    if isinstance(entry.payload, bool):
        if interrupt.reason == TOOL_CALL_REASON:
            return "y" if entry.payload else "n"
    elif isinstance(entry.payload, str):
        return entry.payload
    raise ResumeRefusedError(
        PAYLOAD_REFUSED,
        f"interrupt {interrupt.id} asked for a string answer, got {type(entry.payload).__name__}",
    )


@dataclass(frozen=True, slots=True)
class Abandoned:
    """A held turn whose question the client gave up on, for its run to close as cancelled."""

    turn: "ServedTurn"


def resume_held_turn(
    turns: ServedTurns,
    incoming: RunAgentInput,
    send: MemoryObjectSendStream[BaseEvent],
) -> "ServedTurn | Abandoned | None":
    """Hand a resume to the turn it addresses, and point that turn at this exchange.

    Returns the turn now carrying the run, `Abandoned` when the client gave up
    on the question (the turn is to be stopped, not carried), or `None` when
    the thread holds no question to answer.

    A resume that cannot be honoured raises `ResumeRefusedError`, and the turn
    is put back with its deadline unchanged, so a legitimate answer arriving in
    time still resumes it.
    """
    turn = turns.take(incoming.thread_id)
    outstanding = turn.outstanding if turn is not None else None
    if turn is None or outstanding is None or turns.expired(turn):
        # Unknown, already answered, expired, or held by another process: which
        # of them is timing or routing, and none is the client's error. What it
        # sent answers nothing this server asked.
        if turn is not None:
            turn.release()
        return None

    entry = None
    for candidate in resume_entry(incoming):
        if candidate.interrupt_id == outstanding.id and entry is None:
            entry = candidate
        else:
            # An answer to something this thread is not asking: the protocol
            # says to carry on without it, and to say so.
            logger.warning(
                "ignoring an AG-UI resume entry for interrupt %r: thread %s is not waiting on it",
                candidate.interrupt_id,
                incoming.thread_id,
            )
    if entry is None:
        turns.restore(turn)
        raise ResumeRefusedError(NOT_COVERED, _uncovered(incoming.thread_id))

    try:
        # Before the payload is so much as looked at, and before "cancelled" is
        # honoured: ending someone else's turn is not a lesser act than
        # answering it.
        check_proof(entry, outstanding, required=turns.require_proof)

        match entry.status:
            case "cancelled":
                return Abandoned(turn)
            case "resolved":
                payload = answer_from(entry, outstanding)
            case _:
                assert_never(entry.status)
    except Exception:
        # Anything short of delivering the answer puts the turn back, so a
        # legitimate resume inside the deadline still reaches it. Not only
        # `ResumeRefusedError`: a turn taken and then dropped on an unexpected
        # failure is unreachable by retrieval, sweep and eviction alike.
        turns.restore(turn)
        raise

    turn.output.rebind(run_id=incoming.run_id, send=send)
    turn.deliver(payload)
    return turn


async def serve_exchange(
    turns: ServedTurns,
    incoming: RunAgentInput,
    encoder: EventEncoder,
    start: "Callable[[TurnOutput], ServedTurn]",
) -> AsyncIterator[str]:
    """Drive one AG-UI exchange over a turn, and yield its encoded events.

    Begins or resumes the turn, carries it to the run's terminating event, and
    leaves a held turn running. `start` is how the calling transport launches
    a fresh turn writing to the `TurnOutput` it is handed.

    Wrap in `contextlib.aclosing`: this holds a channel open across yields.
    """
    # Here as well as where a body is read: a caller building the input itself
    # has the SDK's models, which keep what they do not recognise.
    strip_unrecognised(incoming)
    if (unsupported := refuse_protocol_version(incoming)) is not None:
        # Before RUN_STARTED: no run starts that this server cannot speak to.
        yield encoder.encode(unsupported)  # noqa: ASYNC119
        return

    send, receive = create_memory_object_stream[BaseEvent]()

    # Before RUN_STARTED, like the version check: a resume that cannot be
    # honoured is refused before anything is sent.
    try:
        turn = begin_turn(turns, incoming, send, start)
    except ResumeRefusedError as refused:
        yield encoder.encode(refused.as_event(timestamp_ms()))  # noqa: ASYNC119
        return
    except Exception as error:
        # A stream that just stops leaves the client with no outcome to act on.
        logger.exception("failed to begin an AG-UI turn for thread %s", incoming.thread_id)
        yield encoder.encode(  # noqa: ASYNC119
            RunErrorEvent(message=str(error) or type(error).__name__, timestamp=timestamp_ms())
        )
        return

    # Emitted by the exchange, not by the turn: a resumed turn started under an
    # earlier run id in an earlier exchange, and it is *this* run that is
    # starting. The version is this server's own, never an echo of the client's.
    yield encoder.encode(  # noqa: ASYNC119
        RunStartedEvent(
            thread_id=incoming.thread_id,
            run_id=incoming.run_id,
            protocol_version=PROTOCOL_VERSION,
            timestamp=timestamp_ms(),
        )
    )

    if isinstance(turn, Abandoned):
        async for chunk in _cancel(turn.turn, incoming, encoder):
            yield chunk  # noqa: ASYNC119
        return

    ended = held = False
    async with receive:
        async for event in receive:
            yield encoder.encode(event)  # noqa: ASYNC119
            if isinstance(event, (RunFinishedEvent, RunErrorEvent)):
                # The exchange ends on the event that terminates the run, not on
                # the turn's own end: a turn can outlive this response.
                ended, held = True, is_interrupt(event)
                break

    if not ended:
        # The turn let go of the run without ending it — stopped while it was
        # pausing, which leaves nothing that can be closed in order. A run that
        # started on the wire is always ended on it.
        logger.error("an AG-UI turn for thread %s stopped without ending run %s", incoming.thread_id, incoming.run_id)
        yield encoder.encode(  # noqa: ASYNC119
            RunErrorEvent(message="the run was stopped before it could finish", timestamp=timestamp_ms())
        )

    if not held:
        # A held turn is waiting for an answer this exchange will not bring, so
        # awaiting it would never return.
        await turn.settle()


async def _cancel(turn: ServedTurn, incoming: RunAgentInput, encoder: EventEncoder) -> AsyncIterator[str]:
    """Stop an abandoned turn, and close its run as cancelled.

    Stopping is what abandonment means here; the agent does not carry on past
    the question. The protocol forbids reporting a stopped run as success, and
    a run finishing with anything open, so the run first sends what the turn
    kept while paused and an end for everything still open.
    """
    try:
        await turn.aclose()
        owed = turn.output.abandon()
    except Exception as error:
        logger.exception("failed to stop an abandoned AG-UI turn for thread %s", incoming.thread_id)
        yield encoder.encode(RunErrorEvent(message=str(error) or type(error).__name__, timestamp=timestamp_ms()))
        return

    for event in owed:
        yield encoder.encode(event)
    yield encoder.encode(
        RunFinishedEvent(
            thread_id=incoming.thread_id,
            run_id=incoming.run_id,
            timestamp=timestamp_ms(),
            usage=turn.output.take_usage(),
            outcome=RunFinishedCancelledOutcome(),
        )
    )


async def drive_run(output: TurnOutput, work: Coroutine[Any, Any, None]) -> None:
    """Carry out a turn's `work`, then end its run on the wire however the work ended.

    A failure is logged and reported as `RUN_ERROR`, not raised: the run has
    already answered, and raising would cut its body short behind the event. A
    stop the server makes — shutdown, eviction — closes what the run opened and
    finishes it as cancelled, unless the turn was pausing, whose run is ending
    on its interrupt already.
    """
    try:
        await work
    except asyncio.CancelledError:
        if not output.paused:
            await output.stop()
        raise
    except Exception as error:
        logger.exception("AG-UI run %s on thread %s failed", output.run_id, output.thread_id)
        await output.fail(error)
    else:
        await output.succeed()
    finally:
        # The exchange reading this turn ends on its terminating event, but
        # the channel is the turn's: closed here, once there is nothing more
        # to say, on every path including cancellation while held.
        await output.aclose()


def refuse_protocol_version(incoming: RunAgentInput) -> RunErrorEvent | None:
    """The refusal for a client declaring a protocol major other than this server's, if it does.

    An absent version is a client predating 1.0, served quietly. A newer minor,
    or a version that cannot be read, is served with a warning: the protocol
    forbids rejecting either.
    """
    declared = incoming.protocol_version
    if declared is None:
        return None

    ours = _parse_version(PROTOCOL_VERSION)
    theirs = _parse_version(declared)
    if theirs is None or ours is None:
        logger.warning("serving an AG-UI client declaring protocol version %r, which cannot be read", declared)
        return None
    if theirs[0] != ours[0]:
        return RunErrorEvent(
            message=f"this server speaks AG-UI {PROTOCOL_VERSION}, and cannot serve a client on {declared}",
            code=UNSUPPORTED_PROTOCOL_VERSION,
            timestamp=timestamp_ms(),
        )
    if theirs > ours:
        logger.warning(
            "serving an AG-UI client on protocol %s with %s: what the newer minor adds will not be used",
            declared,
            PROTOCOL_VERSION,
        )
    return None


def _parse_version(version: str) -> tuple[int, int] | None:
    match = _VERSION.fullmatch(version)
    return None if match is None else (int(match[1]), int(match[2]))


def begin_turn(
    turns: ServedTurns,
    incoming: RunAgentInput,
    send: MemoryObjectSendStream[BaseEvent],
    start: "Callable[[TurnOutput], ServedTurn]",
) -> "ServedTurn | Abandoned":
    """The turn this run drives — a held one resumed, or a fresh one started.

    `Abandoned` when the run gives up on the question it addresses. Raises
    `ResumeRefusedError` if it addresses one that cannot be honoured, or leaves
    the question the thread holds unanswered. A resume on a thread holding none
    is dropped with a warning, and the run starts as an ordinary one.
    """
    if resume_entry(incoming):
        if (held := resume_held_turn(turns, incoming, send)) is not None:
            return held
        # Answers to nothing this server holds: the protocol has a producer
        # treat them as unrecognised, proceed without them and say so — a
        # restart, or another worker, is no reason to fail the run.
        for entry in resume_entry(incoming):
            logger.warning(
                "ignoring an AG-UI resume entry for interrupt %r: thread %s holds no interrupt",
                entry.interrupt_id,
                incoming.thread_id,
            )
        incoming.resume = None

    # Omission is not abandonment: the question stays held until it is answered,
    # given up with a "cancelled" entry, or expires.
    if turns.holds(incoming.thread_id):
        raise ResumeRefusedError(NOT_COVERED, _uncovered(incoming.thread_id))
    return start(TurnOutput(thread_id=incoming.thread_id, run_id=incoming.run_id, send=send))


def _uncovered(thread_id: str) -> str:
    return (
        f"thread {thread_id} is waiting on an interrupt this run does not answer: "
        'resume it, or give it up with a "cancelled" entry'
    )


def is_interrupt(event: BaseEvent) -> bool:
    """Whether `event` is the `RUN_FINISHED` that pauses a run rather than ends it."""
    return isinstance(event, RunFinishedEvent) and isinstance(event.outcome, RunFinishedInterruptOutcome)


__all__ = (
    "AG2_METADATA_KEY",
    "DEFAULT_RETENTION",
    "INPUT_REQUIRED_REASON",
    "NOT_COVERED",
    "NOT_PROVEN",
    "PAYLOAD_REFUSED",
    "PROOF_KEY",
    "TOOL_CALL_REASON",
    "UNSUPPORTED_PROTOCOL_VERSION",
    "ClientInterrupter",
    "Retention",
    "ServedTurn",
    "ServedTurns",
    "TurnOutput",
    "drive_run",
    "serve_exchange",
    "timestamp_ms",
    "utc_now",
)
