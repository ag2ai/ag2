# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Hold a served agent's turn open across AG-UI exchanges.

Shared by `ag2.ag_ui.stream` and `ag2.a2ui.transports.ag_ui`. A held turn
lives in this process only: resumes must be routed back to it (sticky sessions
are required) and it does not survive a restart.
"""

import asyncio
import logging
import secrets
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import inf
from typing import Any

from ag_ui.core import (
    AgentCapabilities,
    BaseEvent,
    HumanInTheLoopCapabilities,
    IdentityCapabilities,
    Interrupt,
    ResumeEntry,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunFinishedInterruptOutcome,
    RunFinishedSuccessOutcome,
    RunStartedEvent,
)
from ag_ui.encoder import EventEncoder
from anyio import BrokenResourceError, ClosedResourceError, create_memory_object_stream
from anyio.streams.memory import MemoryObjectSendStream

from ag2.annotations import Context
from ag2.events import BaseEvent as AG2Event
from ag2.events import HumanInputRequest, HumanMessage, ToolApprovalRequest
from ag2.exceptions import AG2Error, HumanInputError, HumanInputTimeoutError

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
NO_HELD_TURN = "INTERRUPT_NOT_HELD"
NOT_OUTSTANDING = "INTERRUPT_NOT_OUTSTANDING"
PAYLOAD_REFUSED = "INTERRUPT_PAYLOAD_REFUSED"
NOT_PROVEN = "INTERRUPT_NOT_PROVEN"

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
    """One of `NO_HELD_TURN`, `NOT_OUTSTANDING`, `NOT_PROVEN`, `PAYLOAD_REFUSED`.

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
    always name the current run. Sending into an exchange that has already
    closed drops the event rather than failing the turn.
    """

    __slots__ = ("thread_id", "run_id", "_send")

    def __init__(self, *, thread_id: str, run_id: str, send: MemoryObjectSendStream[BaseEvent]) -> None:
        self.thread_id = thread_id
        self.run_id = run_id
        self._send = send

    def rebind(self, *, run_id: str, send: MemoryObjectSendStream[BaseEvent]) -> None:
        """Point the turn at the exchange now carrying it."""
        self.run_id = run_id
        self._send = send

    async def send(self, event: BaseEvent) -> None:
        try:
            await self._send.send(event)
        except (BrokenResourceError, ClosedResourceError):
            logger.debug("dropping %s: no AG-UI exchange is carrying this turn", type(event).__name__)

    async def aclose(self) -> None:
        await self._send.aclose()


class ServedTurn:
    """One AG-UI-served agent turn, owned by the server rather than by a request.

    Launched through `start`; from then on its lifetime is `ServedTurns`', so it
    survives the exchange that started it.
    """

    __slots__ = ("output", "_task", "_outstanding", "_answer")

    def __init__(self, output: TurnOutput) -> None:
        self.output = output
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

    async def result(self) -> None:
        """Wait for the turn to end, re-raising whatever it raised.

        Never call this on a *held* turn: nothing will end it but the answer
        that has not arrived.
        """
        assert self._task is not None, "result() before start()"
        await self._task

    def suspend(self, interrupt: Interrupt, answer: "asyncio.Future[str]") -> None:
        """Park the turn on `interrupt` until `answer` is resolved.

        One question is outstanding per turn, so a second ask raises
        `HumanInputError` rather than displacing the first.
        """
        if self._outstanding is not None:
            # Two questions at once means concurrent work under one turn —
            # parallel subtasks. Overwriting the slot would orphan the first
            # future, hanging that branch until the turn's deadline, and the
            # second question would go out on an exchange that already ended on
            # the first. Serving several at once needs one outcome carrying them
            # all and a resume routed per interrupt; until then, say so.
            #
            # Logged as well as raised: this fails the turn, and the turn's own
            # exchange ended on the first question, so the raise reaches no
            # client. The operator is the only one who can hear it.
            logger.error(
                "an AG-UI turn on thread %s asked a second question (interrupt %s) while still "
                "waiting on interrupt %s; the turn will fail",
                self.thread_id,
                interrupt.id,
                self._outstanding.id,
            )
            raise HumanInputError(
                f"This AG-UI turn is already waiting on interrupt {self._outstanding.id}. "
                "One question can be outstanding per turn, so concurrent asking — "
                "parallel subtasks that each ask a human — is not supported yet."
            )
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

    __slots__ = ("_live", "_held", "retention", "_now")

    def __init__(
        self,
        *,
        retention: Retention = DEFAULT_RETENTION,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        self._live: set[ServedTurn] = set()
        self._held: dict[str, ServedTurn] = {}
        self.retention = retention
        self._now = now

    def track(self, turn: ServedTurn, task: "asyncio.Task[None]") -> None:
        """Own `turn`'s lifetime until its task completes."""
        self._live.add(turn)
        task.add_done_callback(_Discard(self, turn))

    async def ask(self, turn: ServedTurn, interrupt: Interrupt) -> str:
        """Put `interrupt` to the client, hold `turn`, and return the answer."""
        answer: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        turn.suspend(interrupt, answer)
        # Held before the question is emitted, never after: the exchange ends on
        # that very event, so a resume can be in flight the moment it lands, and
        # a turn not yet held is unreachable by retrieval and eviction.
        self.hold(turn)
        seconds = _seconds_until(interrupt, self._now())
        try:
            await turn.output.send(
                RunFinishedEvent(
                    thread_id=turn.output.thread_id,
                    run_id=turn.output.run_id,
                    timestamp=timestamp_ms(),
                    outcome=RunFinishedInterruptOutcome(interrupts=[interrupt]),
                )
            )
            # The deadline the client was shown is kept here, by the call that
            # advertised it. A held turn is a suspended coroutine, so the
            # coroutine is what has to stop waiting: nothing else knows when, and
            # a registry swept only by later traffic would hold an unanswered
            # question for as long as the process stayed quiet.
            return await asyncio.wait_for(answer, seconds)
        except TimeoutError as error:
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

    def release_thread(self, thread_id: str) -> None:
        """Cancel whatever `thread_id` was holding, because it has moved on."""
        turn = self._held.pop(thread_id, None)
        if turn is not None:
            turn.release()

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

    def deadline(self, timeout: float | None = None) -> datetime:
        """When an interrupt raised now stops being answerable.

        The earlier of :attr:`Retention.ttl` and `timeout`, the caller's own
        bound — clients reject late answers locally against this figure, so it
        has to be the one that will in fact apply.
        """
        seconds = self.retention.ttl if timeout is None else min(self.retention.ttl, timeout)
        return self._now() + timedelta(seconds=seconds)

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
        answer = await self._turns.ask(self._turn, self.interrupt_for(event))
        await context.send(HumanMessage.ensure_message(answer, parent_id=event.id))
        return None

    def interrupt_for(self, event: HumanInputRequest) -> Interrupt:
        """The wire interrupt for one human-input request, under the request's own id."""
        # A request gating a tool call says so and names the call, so a client
        # can offer buttons rather than a text box without parsing the prose.
        approval = event if isinstance(event, ToolApprovalRequest) else None
        return Interrupt(
            id=event.id,
            reason=INPUT_REQUIRED_REASON if approval is None else TOOL_CALL_REASON,
            message=event.content,
            tool_call_id=None if approval is None else approval.tool_call_id,
            response_schema=ANSWER_SCHEMA if approval is None else APPROVAL_SCHEMA,
            expires_at=self._turns.deadline(event.timeout).isoformat(),
            metadata={AG2_METADATA_KEY: {PROOF_KEY: issue_proof()}},
        )


def issue_proof() -> str:
    """A fresh secret, left with one interrupt and asked for on its answer."""
    # A capability, not a signature: the question it proves is suspended in this
    # process, so the value to compare against is in memory beside it — no key,
    # nothing to rotate. Per interrupt, so a leaked proof does not carry over.
    return secrets.token_urlsafe(_PROOF_BYTES)


def check_proof(entry: ResumeEntry, interrupt: Interrupt) -> None:
    """Verify that `entry` carries the proof `interrupt` was issued with.

    Raises `ResumeRefusedError` under `NOT_PROVEN` if it does not.
    """
    if not secrets.compare_digest(_proof_in(entry.metadata), _proof_in(interrupt.metadata)):
        raise ResumeRefusedError(
            NOT_PROVEN,
            f"the resume for interrupt {interrupt.id} does not carry the proof it was issued with",
        )


def _proof_in(metadata: "dict[str, Any] | None") -> bytes:
    # Absent and malformed are one case: both mean nothing was proved, and
    # telling them apart would only say which half to fix. Bytes, not str: a
    # proof off the wire is arbitrary text and `compare_digest` raises
    # TypeError on a non-ASCII str rather than reporting a mismatch.
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


def resume_held_turn(
    turns: ServedTurns,
    incoming: RunAgentInput,
    send: MemoryObjectSendStream[BaseEvent],
) -> "ServedTurn | None":
    """Hand a resume to the turn it addresses, and point that turn at this exchange.

    Returns the turn now carrying the run, or `None` when the client gave up on
    the question: the turn is ended and this exchange has nothing to carry.

    A resume that cannot be honoured raises `ResumeRefusedError`. Except under
    `NO_HELD_TURN` the turn is put back with its deadline unchanged, so a
    legitimate answer arriving in time still resumes it.
    """
    turn = turns.take(incoming.thread_id)
    if turn is None or turns.expired(turn):
        # Refused under the same code either way: whether a turn past its
        # deadline is still in the registry or was already swept by someone
        # else's traffic is timing, and a client cannot be told two different
        # things about one answer arriving too late.
        if turn is not None:
            turn.release()
        raise ResumeRefusedError(
            NO_HELD_TURN,
            f"thread {incoming.thread_id} is not holding an interrupt: it is unknown, already answered, or expired",
        )

    outstanding = turn.outstanding
    entry = next((e for e in resume_entry(incoming) if e.interrupt_id == outstanding.id), None) if outstanding else None
    if outstanding is None or entry is None:
        turns.restore(turn)
        raise ResumeRefusedError(
            NOT_OUTSTANDING,
            f"thread {incoming.thread_id} is not waiting on any interrupt this run addresses",
        )

    try:
        # Before the payload is so much as looked at, and before "cancelled" is
        # honoured: ending someone else's turn is not a lesser act than
        # answering it.
        check_proof(entry, outstanding)

        if entry.status == "cancelled":
            turn.release()
            return None

        payload = answer_from(entry, outstanding)
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
    send, receive = create_memory_object_stream[BaseEvent]()

    # Emitted by the exchange, not by the turn: a resumed turn started under an
    # earlier run id in an earlier exchange, and it is *this* run that is
    # starting.
    yield encoder.encode(  # noqa: ASYNC119
        RunStartedEvent(thread_id=incoming.thread_id, run_id=incoming.run_id, timestamp=timestamp_ms())
    )

    try:
        turn = begin_turn(turns, incoming, send, start)
    except ResumeRefusedError as refused:
        yield encoder.encode(refused.as_event(timestamp_ms()))  # noqa: ASYNC119
        return
    except Exception as error:
        # A run already started on the wire is always terminated on the wire: a
        # stream that just stops leaves the client with no outcome to act on.
        logger.exception("failed to begin an AG-UI turn for thread %s", incoming.thread_id)
        yield encoder.encode(  # noqa: ASYNC119
            RunErrorEvent(message=str(error) or type(error).__name__, timestamp=timestamp_ms())
        )
        return

    if turn is None:
        # The client gave up on the question. The turn is gone and will say
        # nothing further, so this run has only its own ending to report.
        yield encoder.encode(  # noqa: ASYNC119
            RunFinishedEvent(
                thread_id=incoming.thread_id,
                run_id=incoming.run_id,
                timestamp=timestamp_ms(),
                outcome=success_outcome(),
            )
        )
        return

    held = False
    async with receive:
        async for event in receive:
            yield encoder.encode(event)  # noqa: ASYNC119
            if isinstance(event, (RunFinishedEvent, RunErrorEvent)):
                # The exchange ends on the event that terminates the run, not on
                # the turn's own end: a turn can outlive this response.
                held = is_interrupt(event)
                break

    if not held:
        # A held turn is waiting for an answer this exchange will not bring, so
        # awaiting it would never return.
        await turn.result()


def begin_turn(
    turns: ServedTurns,
    incoming: RunAgentInput,
    send: MemoryObjectSendStream[BaseEvent],
    start: "Callable[[TurnOutput], ServedTurn]",
) -> "ServedTurn | None":
    """The turn this run drives — a held one resumed, or a fresh one started.

    `None` when the run abandons the question it addresses. Raises
    `ResumeRefusedError` if it addresses one that cannot be honoured.
    """
    if resume_entry(incoming):
        return resume_held_turn(turns, incoming, send)

    # A fresh run on a thread still holding a question has abandoned it — the
    # protocol's own client will not send one — and the turn behind it would
    # otherwise sit here until its deadline.
    turns.release_thread(incoming.thread_id)
    return start(TurnOutput(thread_id=incoming.thread_id, run_id=incoming.run_id, send=send))


def success_outcome() -> RunFinishedSuccessOutcome:
    """The outcome of a run that finished."""
    # Stated on every run, not only on interrupts: the protocol reads an omitted
    # outcome as a producer predating the interrupt-aware lifecycle, and
    # declaring the capability while behaving as one is not a described state.
    return RunFinishedSuccessOutcome()


def is_interrupt(event: BaseEvent) -> bool:
    """Whether `event` is the `RUN_FINISHED` that pauses a run rather than ends it."""
    return isinstance(event, RunFinishedEvent) and isinstance(event.outcome, RunFinishedInterruptOutcome)


def interrupt_capabilities(agent_name: str) -> AgentCapabilities:
    """What this agent tells a client it can do, at connect time."""
    return AgentCapabilities(
        identity=IdentityCapabilities(name=agent_name, type="ag2"),
        human_in_the_loop=HumanInTheLoopCapabilities(supported=True, interrupts=True),
    )


__all__ = (
    "AG2_METADATA_KEY",
    "DEFAULT_RETENTION",
    "INPUT_REQUIRED_REASON",
    "NOT_OUTSTANDING",
    "NOT_PROVEN",
    "NO_HELD_TURN",
    "PAYLOAD_REFUSED",
    "PROOF_KEY",
    "TOOL_CALL_REASON",
    "ClientInterrupter",
    "Retention",
    "ServedTurn",
    "ServedTurns",
    "TurnOutput",
    "interrupt_capabilities",
    "serve_exchange",
    "success_outcome",
    "timestamp_ms",
    "utc_now",
)
