# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Hold a served agent's turn open across a modern-era client round trip.

Protocol revision 2026-07-28 removed the server-to-client request union outright
— the schema defines no ``ServerRequest`` for it. So the only way a served agent
can ask its caller anything is to return the question *as the result of the
call*, in an ``InputRequiredResult``, and continue when the client retries with
the answer.

Continuing means either keeping the suspended coroutine alive or replaying the
work that led to it, and replay is not available here: re-running
:meth:`Agent.ask` re-issues LLM calls, re-runs tool side effects, and re-spends
tokens. (This is exactly why the ``mcp`` SDK's own resolver mechanism, whose
resolver bodies re-run every round, cannot serve the conversational tool.) So the
run is **held in this process** between the two calls.

That is forced by the protocol, not chosen, and it has three consequences that
belong in an operator's head rather than in a bug report:

* **Resuming must reach the same process.** Sticky routing is required.
* **A pause does not survive a restart.**
* **A multi-instance deployment must supply a shared** ``requestState`` **key**,
  since the default policy mints a process-local one.

Retention is bounded by the lifetime of the state token that names the run: once
that has expired no client can resume, so the run is unreclaimable and is
reclaimed. One number, not two that can disagree.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from mcp.types import CallToolResult, InputRequest, InputResponse

from ag2.stream import MemoryStream

logger = logging.getLogger(__name__)

# How many suspended runs this process will hold at once. A backstop, not the
# bound that matters: the state token's TTL is what actually reclaims runs, and
# this only stops a flood of never-answered pauses growing without limit. The
# oldest is reclaimed first, which is also the one closest to expiry.
MAX_PAUSED_RUNS = 256

_STATE_VERSION = 1


@dataclass(frozen=True, slots=True)
class PauseState:
    """The plaintext this server puts inside ``requestState``.

    Never seen by a client: the ``RequestStateBoundary`` seals it on the way out
    and verifies the echo on the way back, so a handler only ever reads
    plaintext it minted. Kept deliberately small — it names the paused run and
    the question, and nothing a client could learn anything from.

    ``conversation`` is the handle the run's conversation goes by (``None`` under
    ``sessions=False``, which mints none). It is carried so a run whose
    conversation has since been evicted is refused rather than resumed into a
    history that is gone; no second notion of continuity is introduced.
    """

    version: int
    pause: str
    question: str
    digest: str
    conversation: str | None

    def encode(self) -> str:
        return json.dumps(
            {
                "v": self.version,
                "p": self.pause,
                "q": self.question,
                "d": self.digest,
                "c": self.conversation,
            },
            separators=(",", ":"),
        )

    @classmethod
    def mint(cls, *, pause: str, question: str, digest: str, conversation: str | None) -> "PauseState":
        return cls(version=_STATE_VERSION, pause=pause, question=question, digest=digest, conversation=conversation)

    @classmethod
    def decode(cls, raw: str) -> "PauseState | None":
        """Read state this server minted, or ``None`` when it is not that.

        The string arrives boundary-authenticated, so anything unreadable here is
        drift inside the operator's own fleet — a rolling upgrade across a shared
        key, say — and is treated as naming no paused run.
        """
        try:
            data = json.loads(raw)
            if data["v"] != _STATE_VERSION:
                return None
            return cls(
                version=_STATE_VERSION,
                pause=str(data["p"]),
                question=str(data["q"]),
                digest=str(data["d"]),
                conversation=data["c"] if data["c"] is None else str(data["c"]),
            )
        except (ValueError, KeyError, TypeError):
            return None


def question_digest(request: InputRequest) -> str:
    """Pin an answer to the exact question it was written for.

    The digest travels in the state and is re-checked against the question the
    run is *currently* waiting on, so an answer minted for a different or
    reworded question is not consumed — the question is asked again instead.
    """
    params = request.params
    rendered = json.dumps(
        params.model_dump(mode="json", by_alias=True, exclude_none=True) if params else None,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode()).hexdigest()[:32]


class SuspendedTurn:
    """One served agent turn, held open in this process across client round trips.

    Two halves talk to each other through it. Inside the turn, whatever needs
    something from the client — the elicitor with a question, the peer-backed
    model with a completion to run — calls :meth:`ask` and blocks. Outside, the
    serving path calls :meth:`advance` to run the turn until it either finishes
    or asks something, and :meth:`answer` to hand an answer in before advancing
    again. What kind of request it is never matters here: this holds the run,
    and the asker decides what to make of the response it gets back.
    """

    __slots__ = ("id", "conversation", "stream", "created", "_task", "_outstanding", "_answer", "_raised")

    def __init__(self, *, conversation: str | None, stream: "MemoryStream", created: float) -> None:
        self.id = uuid4().hex
        self.conversation = conversation
        # The run's own stream, carried so a later round can re-attach progress
        # forwarding to it without going back to the conversation registry (whose
        # turn lock this pause has deliberately let go of).
        self.stream = stream
        self.created = created
        self._task: asyncio.Task[CallToolResult] | None = None
        self._outstanding: tuple[str, InputRequest] | None = None
        self._answer: asyncio.Future[InputResponse] | None = None
        self._raised = asyncio.Event()

    def start(self, run: Coroutine[Any, Any, CallToolResult]) -> None:
        """Launch the turn. Separate from ``__init__`` because the coroutine needs
        this object: the elicitor inside it asks *through* the turn it belongs to."""
        task = asyncio.ensure_future(run)
        # A turn that fails while nobody is awaiting it — an input timeout
        # elapsing mid-pause — would otherwise be reported by asyncio as an
        # unretrieved exception at collection time. Retrieving it here consumes
        # that complaint; ``result()`` still raises it for the retry that comes
        # asking.
        task.add_done_callback(_consume_exception)
        self._task = task

    async def ask(self, request: InputRequest) -> InputResponse:
        """Put a question to the client and suspend until the retry answers it.

        Called from inside the turn. The ``context.input(timeout=)`` the caller
        passed wraps this await, so that timeout now spans the client's side of
        the round trip — and still ends the turn through the existing
        human-input timeout when it elapses first.
        """
        key = uuid4().hex
        answer: asyncio.Future[InputResponse] = asyncio.get_running_loop().create_future()
        self._outstanding = (key, request)
        self._answer = answer
        self._raised.set()
        try:
            return await answer
        finally:
            # Only if this is still *our* question: a cancelled wait (an input
            # timeout) must not wipe a question a later round has since raised.
            if self._outstanding is not None and self._outstanding[0] == key:
                self._outstanding = None
                self._answer = None

    @property
    def outstanding(self) -> "tuple[str, InputRequest] | None":
        """The question this run is waiting on, if it is waiting on one."""
        return self._outstanding

    async def advance(self) -> "CallToolResult | tuple[str, InputRequest]":
        """Run until the turn produces a result or asks the client something.

        Returns the finished :class:`CallToolResult`, or the outstanding
        ``(key, request)`` when the turn is suspended on a question.

        Raises:
            Exception: Whatever the turn raised — which is how a declined
                elicitation, an input timeout, or an agent failure reaches the
                caller that is waiting on this round.
        """
        assert self._task is not None, "advance() before start()"
        # A finished turn wins over a question: the only way both hold at once is
        # a question whose wait was cut short (an input timeout), and then the
        # result — the failure — is what there is to report.
        if self._task.done():
            return self._task.result()
        # Still parked on a question nobody answered this round: re-ask it rather
        # than wait for one that will never be raised again.
        if self._outstanding is not None:
            return self._outstanding
        self._raised.clear()
        raised = asyncio.ensure_future(self._raised.wait())
        try:
            await asyncio.wait({self._task, raised}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            raised.cancel()
        if self._task.done():
            return self._task.result()
        assert self._outstanding is not None
        return self._outstanding

    def answer(self, key: str, digest: str, result: InputResponse) -> bool:
        """Hand an answer to the outstanding question.

        ``False`` when it is not an answer to *this* question — a stale key, or a
        digest naming a different rendering — in which case nothing is consumed
        and the caller re-asks whatever the run is actually waiting on.
        """
        if self._answer is None or self._outstanding is None:
            return False
        outstanding_key, request = self._outstanding
        if outstanding_key != key or question_digest(request) != digest:
            logger.info("discarding an answer for a question this run is not waiting on")
            return False
        answer = self._answer
        # Cleared here, synchronously, rather than in ``ask``'s ``finally``: the
        # suspended coroutine does not resume until the loop next runs it, and
        # until then the run must already read as no longer waiting — otherwise
        # this round would report the answered question all over again.
        self._outstanding = None
        self._answer = None
        answer.set_result(result)
        return True

    def reclaim(self) -> None:
        """Cancel the held run and close its turn scope."""
        if self._task is not None and not self._task.done():
            self._task.cancel()


def _consume_exception(task: "asyncio.Task[Any]") -> None:
    if not task.cancelled():
        task.exception()


class PausedRuns:
    """Bounded registry of the runs this process is holding open.

    Retention is the lifetime of the state that names a run: ``ttl`` is the
    ``requestState`` TTL, so a run whose state no client can present any more is
    reclaimed rather than left as garbage nothing can reach. The count bound is a
    backstop — see :data:`MAX_PAUSED_RUNS`.

    Sweeping happens on every registry operation rather than on a timer: the
    boundary already refuses an expired token before a handler runs, so an
    expired run is unreachable the instant it expires and only its memory is
    left to reclaim.
    """

    __slots__ = ("_runs", "_ttl", "_max", "_clock")

    def __init__(
        self,
        *,
        ttl: float,
        max_runs: int = MAX_PAUSED_RUNS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._runs: OrderedDict[str, SuspendedTurn] = OrderedDict()
        self._ttl = ttl
        self._max = max_runs
        self._clock = clock

    @property
    def ttl(self) -> float:
        return self._ttl

    def now(self) -> float:
        return self._clock()

    def register(self, turn: SuspendedTurn) -> None:
        self._sweep()
        self._runs[turn.id] = turn
        while len(self._runs) > self._max:
            _, oldest = self._runs.popitem(last=False)
            logger.warning("reclaiming the oldest paused MCP run: %d already held", self._max)
            oldest.reclaim()

    def take(self, pause: str) -> SuspendedTurn | None:
        """Remove and return the run ``pause`` names, or ``None``.

        Removed rather than looked up: a run is held by exactly one round at a
        time, and re-registered only if it pauses again. Two retries racing the
        same state therefore cannot both drive the same turn.
        """
        self._sweep()
        return self._runs.pop(pause, None)

    def discard_conversation(self, handle: str) -> None:
        """Reclaim any run belonging to a conversation that has just gone away.

        A run abandoned without either bound elapsing is still reclaimed here,
        when the conversation registry evicts the handle that names it.
        """
        for pause in [p for p, t in self._runs.items() if t.conversation == handle]:
            self._runs.pop(pause).reclaim()

    def _sweep(self) -> None:
        cutoff = self._clock() - self._ttl
        for pause in [p for p, t in self._runs.items() if t.created <= cutoff]:
            self._runs.pop(pause).reclaim()


__all__ = (
    "MAX_PAUSED_RUNS",
    "PauseState",
    "PausedRuns",
    "SuspendedTurn",
    "question_digest",
)
