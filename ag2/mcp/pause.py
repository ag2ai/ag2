# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Hold a served agent's turn open across a modern-era client round trip.

Revision 2026-07-28 defines no server-to-client request, so a question comes back
as the *result* of the call and the run is held here until the client retries.
Held rather than replayed because replaying a conversational turn re-issues LLM
calls, re-runs tool side effects and re-spends tokens.

Three consequences for an operator: the retry must reach the process holding the
run, so **sticky routing is required**; a pause does not survive a restart; and
more than one replica needs a shared ``requestState`` key, since the default
policy mints a process-local one.

Retention is the lifetime of that state token and nothing else — once no client
can present a resumable one the run is unreachable, so it is reclaimed. One
number, not two that can disagree.
"""

import asyncio
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar
from uuid import uuid4

from mcp.types import CallToolResult, InputRequest, InputResponse

from ag2.stream import MemoryStream

logger = logging.getLogger(__name__)

_T = TypeVar("_T", bound=InputResponse)

# A backstop against a flood of never-answered pauses, not the bound that
# matters: the state token's TTL is what reclaims runs.
MAX_PAUSED_RUNS = 256

_STATE_VERSION = 1


@dataclass(frozen=True, slots=True)
class PauseState:
    """The plaintext this server puts inside ``requestState``.

    Sealed by the ``RequestStateBoundary`` on the way out and verified on the way
    back, so a handler only ever reads plaintext it minted. The run's
    conversation is deliberately absent: eviction already reclaims the run, so
    there is no second notion of continuity to keep in step.

    Attributes:
        run_id: The :class:`SuspendedTurn` this state names.
        request_key: The key of the question that run is waiting on, and the key
            its answer comes back under in ``inputResponses``.
    """

    version: int
    run_id: str
    request_key: str

    def encode(self) -> str:
        return json.dumps(
            {
                "v": self.version,
                "p": self.run_id,
                "q": self.request_key,
            },
            separators=(",", ":"),
        )

    @classmethod
    def mint(cls, *, run_id: str, request_key: str) -> "PauseState":
        return cls(version=_STATE_VERSION, run_id=run_id, request_key=request_key)

    @classmethod
    def decode(cls, raw: str) -> "PauseState | None":
        """Read state this server minted, or ``None`` when it is not that.

        Arrives boundary-authenticated, so anything unreadable is drift within
        the operator's own fleet (a rolling upgrade across a shared key) and
        names no paused run.
        """
        try:
            data = json.loads(raw)
            if data["v"] != _STATE_VERSION:
                return None
            return cls(
                version=_STATE_VERSION,
                run_id=str(data["p"]),
                request_key=str(data["q"]),
            )
        except (ValueError, KeyError, TypeError):
            return None


class SuspendedTurn:
    """One served agent turn, held open across client round trips.

    Inside the turn, whatever needs something from the client calls :meth:`ask`
    and blocks; outside, the serving path drives :meth:`advance` and
    :meth:`answer`. What kind of request it is never matters here — this holds
    the run, and the asker judges the response.
    """

    __slots__ = ("id", "conversation", "stream", "created", "_task", "_outstanding", "_answer", "_raised")

    def __init__(self, *, conversation: str | None, stream: "MemoryStream", created: float) -> None:
        self.id = uuid4().hex
        self.conversation = conversation
        # Carried so a later round can re-attach progress forwarding without
        # going back to the conversation registry, whose lock this pause released.
        self.stream = stream
        # When the state naming this run was last minted — see ``register``.
        self.created = created
        self._task: asyncio.Task[CallToolResult] | None = None
        self._outstanding: tuple[str, InputRequest] | None = None
        self._answer: asyncio.Future[InputResponse] | None = None
        self._raised = asyncio.Event()

    def start(self, run: Coroutine[Any, Any, CallToolResult]) -> None:
        """Launch the turn. Separate from ``__init__`` because the coroutine needs
        this object: the elicitor inside it asks *through* the turn it belongs to."""
        task = asyncio.ensure_future(run)
        # A turn that fails while nobody awaits it (an input timeout elapsing
        # mid-pause) would be reported as an unretrieved exception; ``result()``
        # still raises it for the retry that comes asking.
        task.add_done_callback(_consume_exception)
        self._task = task

    async def ask(self, request: InputRequest) -> InputResponse:
        """Put a question to the client and suspend until the retry answers it.

        Called from inside the turn, so any ``context.input(timeout=)`` wraps
        this await and thus spans the client's side of the round trip.
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

    async def ask_for(self, request: InputRequest, expected: type[_T]) -> "_T | None":
        """:meth:`ask`, and ``None`` when the answer is not of the kind asked for.

        Reports the mismatch and decides nothing: what to do about one differs
        per asker (fall through to the agent's hook, or refuse the turn).
        """
        answered = await self.ask(request)
        if isinstance(answered, expected):
            return answered
        logger.warning(
            "MCP client answered a %s with %s",
            type(request).__name__,
            type(answered).__name__,
        )
        return None

    @property
    def outstanding(self) -> "tuple[str, InputRequest] | None":
        """The question this run is waiting on, if it is waiting on one."""
        return self._outstanding

    async def advance(self) -> "CallToolResult | tuple[str, InputRequest]":
        """Run until the turn produces a result or asks the client something.

        Returns the finished result, or the outstanding ``(key, request)``.

        Raises:
            Exception: Whatever the turn raised — a declined elicitation, an
                input timeout, an agent failure — for this round's caller.
        """
        assert self._task is not None, "advance() before start()"
        # A finished turn wins over a question: both hold at once only when a
        # question's wait was cut short, and then the failure is what to report.
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

    def answer(self, key: str, result: InputResponse) -> bool:
        """Hand an answer to the outstanding question, or refuse it.

        The key **is** the pinning, and nothing further is needed: :meth:`ask`
        mints a fresh one per question, so a matching key can only have come from
        the question now outstanding. ``False`` consumes nothing, and the caller
        re-asks whatever the run is actually waiting on.
        """
        if self._answer is None or self._outstanding is None:
            return False
        outstanding_key, _ = self._outstanding
        if outstanding_key != key:
            logger.info("discarding an answer for a question this run is not waiting on")
            return False
        answer = self._answer
        # Cleared synchronously, not in ``ask``'s ``finally``: the coroutine does
        # not resume until the loop next runs it, and until then the run must
        # already read as no longer waiting, or this round re-reports the
        # question it just answered.
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

    ``ttl`` is the ``requestState`` TTL: a run whose state no client can present
    any more is unreachable, so it is reclaimed. Sweeping is **lazy**, on each
    registry operation rather than on a timer, so an idle server keeps an expired
    run's task parked until its next call — deliberate, since a timer would add a
    second clock to keep in step with this one.
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

    def now(self) -> float:
        return self._clock()

    def register(self, turn: SuspendedTurn) -> None:
        """Hold a run that has just paused under freshly minted state.

        Restamps ``created``, because the token naming the run is minted now.
        Measuring from the *first* pause instead reclaims a run that pauses
        repeatedly while the token its client holds is still one the boundary
        accepts — do not simplify this back.
        """
        self._sweep()
        turn.created = self._clock()
        self._runs[turn.id] = turn
        while len(self._runs) > self._max:
            _, oldest = self._runs.popitem(last=False)
            logger.warning("reclaiming the oldest paused MCP run: %d already held", self._max)
            oldest.reclaim()

    def take(self, run_id: str) -> SuspendedTurn | None:
        """Remove and return the run ``run_id`` names, or ``None``.

        Removed rather than looked up, so two retries racing the same state
        cannot both drive the one turn.
        """
        self._sweep()
        return self._runs.pop(run_id, None)

    def holds_conversation(self, handle: str | None) -> bool:
        """Whether a run of ``handle``'s conversation is paused right now.

        ``None`` — a stateless call — never matches: each has its own stream.
        """
        if handle is None:
            return False
        self._sweep()
        return any(t.conversation == handle for t in self._runs.values())

    def reclaim_all(self) -> None:
        """Cancel every run this process is holding, on the way down.

        Nothing else does: sweeping is lazy, and on the way down there is no next
        call. Without this each held task is destroyed pending, closing none of
        the turn scopes its tools opened.
        """
        while self._runs:
            _, turn = self._runs.popitem(last=False)
            turn.reclaim()

    def discard_conversation(self, handle: str) -> None:
        """Reclaim any run of a conversation that has just been evicted.

        The third way out, for a run abandoned without either bound elapsing.
        """
        for run_id in [r for r, t in self._runs.items() if t.conversation == handle]:
            self._runs.pop(run_id).reclaim()

    def _sweep(self) -> None:
        cutoff = self._clock() - self._ttl
        for run_id in [r for r, t in self._runs.items() if t.created <= cutoff]:
            self._runs.pop(run_id).reclaim()


__all__ = (
    "MAX_PAUSED_RUNS",
    "PauseState",
    "PausedRuns",
    "SuspendedTurn",
)
