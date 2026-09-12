# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Hold a served agent's turn open across AG-UI exchanges.

A question the agent asks leaves as the terminating ``RUN_FINISHED`` of the
exchange it was raised in, carrying an *interrupt*; the client answers it by
addressing that interrupt in the ``resume`` array of a later run on the **same
thread**. The turn itself never restarts — it stays suspended inside the agent's
own Python function, in this process, until the answer arrives.

For an operator: a resume must reach the process holding the turn, so **sticky
routing is required**; a held turn does not survive a restart; and how long one
is kept is :class:`Retention`, which is also what a client is shown as the
interrupt's deadline.

Shared by both AG-UI transports — ``ag2.ag_ui.stream`` and
``ag2.a2ui.transports.ag_ui`` — which differ only in how they emit and ingest
frames.
"""

import asyncio
import logging
import secrets
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
from ag2.exceptions import AG2Error, HumanInputError

logger = logging.getLogger(__name__)

# What an :class:`Interrupt` raised by ``context.input()`` says it is. The
# protocol leaves ``reason`` a free string; a tool call held for approval says
# so separately, because a client renders the two differently — a text box
# against a pair of buttons — and should not have to read the prose to tell.
HUMAN_INPUT_REASON = "human_input"
TOOL_APPROVAL_REASON = "tool_approval"

# ``context.input()`` is one string in, one string out, so this is the whole of
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

# Codes on the ``RUN_ERROR`` a refused resume produces, so a client can branch
# without parsing prose.
NO_HELD_TURN = "INTERRUPT_NOT_HELD"
NOT_OUTSTANDING = "INTERRUPT_NOT_OUTSTANDING"
PAYLOAD_REFUSED = "INTERRUPT_PAYLOAD_REFUSED"
NOT_PROVEN = "INTERRUPT_NOT_PROVEN"

# Where this server's own envelope data sits inside a metadata object. Not
# ``ag_ui.core.AGUI_METADATA_KEY`` (``"ag-ui"``): the protocol reserves that one
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
    """Now, as the wire spells a timestamp. Shared so both transports agree."""
    return int(time.time() * 1000)


@dataclass(frozen=True, slots=True)
class Retention:
    """How long an unanswered question is kept, and how many at once.

    Not an internal constant: ``ttl`` is what the client is shown as the
    interrupt's deadline, so it is part of the wire contract rather than a
    memory-management detail. Memory held by a pausing server is proportional
    to both fields — each held turn is a suspended coroutine with its history,
    not a record.

    Attributes:
        ttl: Seconds a held turn survives unanswered. Past it the turn is gone
            and its task cancelled.
        max_held: How many turns may be held at once. Registering past this
            evicts the *oldest* and cancels it.
    """

    ttl: float = 900.0
    max_held: int = 128


DEFAULT_RETENTION = Retention()


class ResumeRefusedError(AG2Error):
    """A resume this server will not honour, and the code it is refused under.

    Carried to the client as the protocol's ``RUN_ERROR``, which is what the
    specification names for exactly these cases — never as a silent stream that
    will produce nothing further.
    """

    __slots__ = ("code",)

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code

    def as_event(self, timestamp: int | None = None) -> RunErrorEvent:
        return RunErrorEvent(message=str(self), code=self.code, timestamp=timestamp)


class TurnOutput:
    """Where a held turn's events go, and under which run.

    Rebound on every exchange. A response's stream closes when the exchange
    ends and the turn outlives it, so the run id on the lifecycle events it
    emits next is the *resuming* run's, not the run that started it.

    Sending into an exchange that has already gone drops the event: a client
    that disconnected mid-run must not kill the turn from the outside, and on
    the interrupt path nothing is emitted between the question leaving and the
    next exchange attaching.
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

    The turn is launched through :meth:`start` and from then on its lifetime is
    :class:`ServedTurns`': it survives the exchange that started it, and is
    cancelled on every exit path — answered, abandoned, expired, evicted, or
    shut down.
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
        """Launch the turn (separate from ``__init__``: the coroutine needs this object)."""
        task = asyncio.ensure_future(run)
        # A turn that fails with nobody awaiting it — the exchange that started
        # it is long over — would otherwise be reported as an unretrieved
        # exception when the task is finalised.
        task.add_done_callback(_consume_exception)
        self._task = task
        return task

    async def result(self) -> None:
        """Wait for the turn to end, re-raising whatever it raised.

        For the exchange still holding the response open: the turn ran outside
        its scope, so a failure reaches the caller only by being asked for.
        Never call this on a turn that is *held* — nothing will end it but the
        answer that has not arrived.
        """
        assert self._task is not None, "result() before start()"
        await self._task

    def suspend(self, interrupt: Interrupt, answer: "asyncio.Future[str]") -> None:
        """Park the turn on ``interrupt`` until ``answer`` is resolved.

        A turn carries one question at a time, so asking a second while the
        first is outstanding is refused rather than allowed to overwrite it.
        """
        if self._outstanding is not None:
            # Two questions at once means concurrent work under one turn —
            # parallel subtasks. Overwriting the slot would orphan the first
            # future, hanging that branch until the turn's deadline, and the
            # second question would go out on an exchange that already ended on
            # the first. Serving several at once needs one outcome carrying them
            # all and a resume routed per interrupt; until then, say so.
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
        """Hand ``payload`` to the waiting call.

        The question is forgotten *synchronously*, not in the waiting
        coroutine's own cleanup: that coroutine does not resume until the loop
        next runs it, and until then the turn must already read as no longer
        waiting, or the next exchange re-reports the question it just answered.
        """
        assert self._answer is not None, "deliver() on a turn that is not waiting"
        answer, self._answer, self._outstanding = self._answer, None, None
        answer.set_result(payload)

    def release(self) -> None:
        """Cancel the turn, wherever it is suspended."""
        if self._task is not None and not self._task.done():
            self._task.cancel()


def _consume_exception(task: "asyncio.Task[Any]") -> None:
    if not task.cancelled():
        task.exception()


class ServedTurns:
    """Every turn this process is running, and which of them are held.

    Membership is what keeps a turn alive: a bare ``asyncio`` task nobody
    references may be collected mid-flight, so the server holds one until the
    turn is done.

    Held turns are keyed by **thread**, not by run: the resuming request carries
    a *new* run id under the same thread, which is what the protocol's examples
    show and what its own clients do.
    """

    __slots__ = ("_live", "_held", "retention", "_now")

    def __init__(
        self,
        *,
        retention: Retention = DEFAULT_RETENTION,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        self._live: set[ServedTurn] = set()
        self._held: OrderedDict[str, ServedTurn] = OrderedDict()
        self.retention = retention
        self._now = now

    def now(self) -> datetime:
        return self._now()

    def track(self, turn: ServedTurn, task: "asyncio.Task[None]") -> None:
        """Own ``turn``'s lifetime until its task completes."""
        self._live.add(turn)
        task.add_done_callback(_discarder(self._live, turn))

    async def ask(self, turn: ServedTurn, interrupt: Interrupt) -> str:
        """Put ``interrupt`` to the client, hold ``turn``, and return the answer.

        Held *before* the question is emitted, never after: the exchange ends on
        that very event, so a resume can be in flight the moment it lands, and a
        turn not yet held would be unreachable by every one of retrieval, sweep
        and eviction.
        """
        answer: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        turn.suspend(interrupt, answer)
        self.hold(turn)
        try:
            await turn.output.send(
                RunFinishedEvent(
                    thread_id=turn.output.thread_id,
                    run_id=turn.output.run_id,
                    outcome=RunFinishedInterruptOutcome(interrupts=[interrupt]),
                )
            )
            return await answer
        finally:
            turn.wake()
            self.discard(turn)

    def hold(self, turn: ServedTurn) -> None:
        """Hold ``turn`` for its thread, evicting whatever that thread held before.

        The retention clock is the deadline on the question being held, stamped
        as it was asked — so each new question buys a full bound, and a turn
        whose client answered inside the window is never reclaimed on the
        strength of a question it already answered.
        """
        previous = self._held.get(turn.thread_id)
        if previous is not None and previous is not turn:
            logger.info("releasing a turn held for thread %s: it has been superseded", turn.thread_id)
            previous.release()
        self._held[turn.thread_id] = turn
        self._held.move_to_end(turn.thread_id)
        self._evict_expired()
        while len(self._held) > self.retention.max_held:
            _, oldest = self._held.popitem(last=False)
            logger.warning("releasing the oldest held AG-UI turn: %d already held", self.retention.max_held)
            oldest.release()

    def take(self, thread_id: str) -> "ServedTurn | None":
        """Remove and return the turn held for ``thread_id``, or ``None``.

        Removed rather than looked up, so two resumes racing one thread cannot
        both drive the one turn: the loser finds nothing and is refused.

        An expired turn is still returned, so a resume that arrives too late is
        told it is too late rather than told nothing was ever there; the caller
        decides. Everything *else* that has aged out is reclaimed here.
        """
        turn = self._held.pop(thread_id, None)
        self._evict_expired()
        return turn

    def restore(self, turn: ServedTurn) -> None:
        """Put back a turn taken for a resume that was refused.

        Deliberately not :meth:`hold`: the retention clock is not restamped, so
        a stream of bad resumes cannot keep a turn alive past the deadline its
        client was shown.
        """
        self._held[turn.thread_id] = turn

    def discard(self, turn: ServedTurn) -> None:
        """Stop holding ``turn``, if this is still the turn its thread holds."""
        if self._held.get(turn.thread_id) is turn:
            del self._held[turn.thread_id]

    def release_thread(self, thread_id: str) -> None:
        """Cancel whatever ``thread_id`` was holding, because it has moved on.

        A run that starts on a thread without covering its outstanding interrupt
        has abandoned it — the protocol's own client refuses to send one — and
        the turn behind it would otherwise sit there until its deadline.
        """
        turn = self._held.pop(thread_id, None)
        if turn is not None:
            turn.release()

    async def release_all(self) -> None:
        """Cancel every turn this process is running, on the way down.

        Nothing else does: a turn nobody comes back for is only reclaimed by a
        later request, and on the way down there is none.
        """
        self._held.clear()
        for turn in tuple(self._live):
            turn.release()
        self._live.clear()

    def expired(self, turn: ServedTurn) -> bool:
        """Whether the question ``turn`` is held on can still be answered."""
        return _expired(turn, self._now())

    def deadline(self, timeout: float | None = None) -> datetime:
        """When an interrupt raised now stops being answerable.

        The **earlier** of the retention bound and the deadline implied by the
        timeout the caller passed when asking. The protocol makes the consumer
        the judge of expiry and treats an absent deadline as a promise that the
        interrupt never expires — its client rejects a late answer locally on
        the strength of that — so a deadline is always stated, and it is the one
        that will in fact apply.
        """
        seconds = self.retention.ttl if timeout is None else min(self.retention.ttl, timeout)
        return self._now() + timedelta(seconds=seconds)

    def _evict_expired(self) -> None:
        """Reclaim held turns past their deadline (lazily, on registry traffic)."""
        now = self._now()
        for thread_id, turn in [(t, h) for t, h in self._held.items() if _expired(h, now)]:
            logger.info("releasing an expired held AG-UI turn for thread %s", thread_id)
            del self._held[thread_id]
            turn.release()


def _expired(turn: ServedTurn, now: datetime) -> bool:
    outstanding = turn.outstanding
    return outstanding is not None and outstanding.expires_at is not None and _parse(outstanding.expires_at) <= now


def _parse(expires_at: str) -> datetime:
    return datetime.fromisoformat(expires_at)


def _discarder(live: "set[ServedTurn]", turn: ServedTurn) -> "Callable[[asyncio.Task[Any]], None]":
    """A done-callback dropping ``turn`` from ``live`` (closure built once per turn)."""

    def discard(_task: "asyncio.Task[Any]") -> None:
        live.discard(turn)

    return discard


class ClientInterrupter:
    """Answers a served agent's ``context.input()`` from the human at the AG-UI client.

    This is the transport's own human-input strategy, registered as a stream
    interrupter before the turn starts. It is registered *only* when nobody
    supplied a hook: a caller who did keeps today's behaviour exactly, and a
    caller who did not now has the question put to the client instead of the
    turn dying with "nobody could be asked".
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
        """The wire interrupt for one human-input request.

        The id is the request's own. A question raised inside AG2 already
        carries one, and its answer is already matched by naming that id as its
        parent — which is exactly what an interrupt id means on the wire, so no
        second correlation identifier is introduced.

        The id alone is not proof of anything, so the interrupt also leaves with
        a secret of its own in its envelope: see :func:`issue_proof`.

        A request that gates a tool call says so, and names the call: same
        lifecycle, different occasion, and the client can offer the right UI for
        it without parsing the question.
        """
        approval = event if isinstance(event, ToolApprovalRequest) else None
        return Interrupt(
            id=event.id,
            reason=HUMAN_INPUT_REASON if approval is None else TOOL_APPROVAL_REASON,
            message=event.content,
            tool_call_id=None if approval is None else approval.tool_call_id,
            response_schema=ANSWER_SCHEMA if approval is None else APPROVAL_SCHEMA,
            expires_at=self._turns.deadline(event.timeout).isoformat(),
            metadata={AG2_METADATA_KEY: {PROOF_KEY: issue_proof()}},
        )


def issue_proof() -> str:
    """A fresh secret, to leave with one interrupt and be asked for on its answer.

    A capability, not a signature: the question this proves is suspended in this
    process, so the value it is checked against is already held in memory beside
    it. There is nothing to sign with, no key to manage and nothing to rotate —
    a held turn does not outlive the process that holds it.

    Issued per interrupt rather than per turn or per thread, so a proof that
    leaked with one answered question cannot be used on the next one.
    """
    return secrets.token_urlsafe(_PROOF_BYTES)


def check_proof(entry: ResumeEntry, interrupt: Interrupt) -> None:
    """Verify that ``entry`` comes from whoever ``interrupt`` was put to.

    Raises:
        ResumeRefusedError: the proof is absent, malformed, or not the one issued.
    """
    if not secrets.compare_digest(_proof_in(entry.metadata), _proof_in(interrupt.metadata)):
        raise ResumeRefusedError(
            NOT_PROVEN,
            f"the resume for interrupt {interrupt.id} does not carry the proof it was issued with",
        )


def _proof_in(metadata: "dict[str, Any] | None") -> str:
    """The proof inside one metadata envelope, or ``""`` if there is not one.

    Absent and malformed are one case on purpose: both mean nothing was proved,
    and telling them apart would only tell an attacker which half to fix. The
    empty string never compares equal to an issued proof, which is never empty.
    """
    envelope = (metadata or {}).get(AG2_METADATA_KEY)
    proof = envelope.get(PROOF_KEY) if isinstance(envelope, dict) else None
    return proof if isinstance(proof, str) else ""


def resume_entry(incoming: RunAgentInput) -> "list[ResumeEntry]":
    """The resume entries this run carries, if it is a resume at all.

    An empty array is not a resume: it addresses no interrupt, so the run is an
    ordinary new one.
    """
    return list(incoming.resume or ())


def answer_from(entry: ResumeEntry, interrupt: Interrupt) -> str:
    """The answer ``entry`` carries, as the string the waiting call will read.

    A ``bool`` is accepted for an approval and only for an approval: the client
    that drew two buttons has one, the middleware waiting on the other side
    reads words, and translating between them here is better than either
    teaching every client this server's vocabulary or teaching the middleware
    a second one.

    Raises:
        ResumeRefusedError: the payload is not what the interrupt asked for.
    """
    if isinstance(entry.payload, bool):
        if interrupt.reason == TOOL_APPROVAL_REASON:
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

    Shared by both AG-UI transports: the registry, the correlation and the
    refusals are one implementation, and only the frames around them differ.

    Returns the turn now carrying the run, or ``None`` when the client gave up
    on the question — an outcome, not a failure: the turn is ended and no
    payload is delivered, so this exchange has nothing further to carry.

    Raises:
        ResumeRefusedError: nothing addressed here can be honoured. A turn refused
            for a reason that is not its own — a payload that does not fit, an
            answer for a question it is no longer waiting on — is put back, so a
            legitimate answer arriving inside the deadline still resumes it.
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

    # Before the payload is so much as looked at, and before "cancelled" is
    # honoured: ending someone else's turn is not a lesser act than answering it.
    try:
        check_proof(entry, outstanding)
    except ResumeRefusedError:
        turns.restore(turn)
        raise

    if entry.status == "cancelled":
        turn.release()
        return None

    try:
        payload = answer_from(entry, outstanding)
    except ResumeRefusedError:
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

    The whole shape of a run that can pause: begin it or resume it, carry its
    events until the run terminates, and leave a held turn running. Shared by
    both AG-UI transports, which differ only in ``start`` — how each launches a
    turn of its own writing to the :class:`TurnOutput` it is handed. A client
    that cannot tell the two apart is the point: this is the part it sees.

    ASYNC119 throughout: this is a true streaming generator that must hold the
    channel open across yields; consumers are expected to use
    ``contextlib.aclosing`` for timely cleanup.
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

    ``None`` when the run abandons the question it addresses: there is then no
    turn left for the exchange to carry.

    Raises:
        ResumeRefusedError: the run addresses an interrupt that cannot be honoured.
    """
    if resume_entry(incoming):
        return resume_held_turn(turns, incoming, send)

    # A fresh run on a thread still holding a question has abandoned it — the
    # protocol's own client will not send one — and the turn behind it would
    # otherwise sit here until its deadline.
    turns.release_thread(incoming.thread_id)
    return start(TurnOutput(thread_id=incoming.thread_id, run_id=incoming.run_id, send=send))


def success_outcome() -> RunFinishedSuccessOutcome:
    """The outcome of a run that finished.

    Stated on **every** run, not only on interrupts: the protocol treats an
    omitted outcome as marking a producer written before the interrupt-aware
    lifecycle, and declaring the interrupt capability while behaving as one is a
    state the protocol does not describe.
    """
    return RunFinishedSuccessOutcome()


def is_interrupt(event: BaseEvent) -> bool:
    """Whether ``event`` is the ``RUN_FINISHED`` that pauses a run rather than ends it."""
    return isinstance(event, RunFinishedEvent) and isinstance(event.outcome, RunFinishedInterruptOutcome)


def interrupt_capabilities(agent_name: str) -> AgentCapabilities:
    """What this agent tells a client it can do, at connect time.

    Only the part this work is responsible for: that the agent participates in
    the interrupt protocol, so a frontend can decide up front whether to offer
    the UI for it.
    """
    return AgentCapabilities(
        identity=IdentityCapabilities(name=agent_name, type="ag2"),
        human_in_the_loop=HumanInTheLoopCapabilities(supported=True, interrupts=True),
    )


__all__ = (
    "AG2_METADATA_KEY",
    "ANSWER_SCHEMA",
    "APPROVAL_SCHEMA",
    "DEFAULT_RETENTION",
    "HUMAN_INPUT_REASON",
    "NOT_OUTSTANDING",
    "NOT_PROVEN",
    "NO_HELD_TURN",
    "PAYLOAD_REFUSED",
    "PROOF_KEY",
    "TOOL_APPROVAL_REASON",
    "ClientInterrupter",
    "ResumeRefusedError",
    "Retention",
    "ServedTurn",
    "ServedTurns",
    "TurnOutput",
    "answer_from",
    "begin_turn",
    "check_proof",
    "interrupt_capabilities",
    "is_interrupt",
    "issue_proof",
    "resume_entry",
    "resume_held_turn",
    "serve_exchange",
    "success_outcome",
    "timestamp_ms",
    "utc_now",
)
