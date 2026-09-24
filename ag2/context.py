# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping
from contextlib import AbstractAsyncContextManager, AbstractContextManager, suppress
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, cast, overload, runtime_checkable
from uuid import UUID

import anyio.from_thread
from fast_depends import Provider

from ag2.types import ClassInfo, SendableMessage

from .events import BaseEvent, HumanInputRequest, HumanMessage, Input, MessageEnqueued, ModelRequest
from .events.conditions import Condition
from .exceptions import HumanInputError, HumanInputFailedError, HumanInputTimeoutError

logger = logging.getLogger(__name__)

StreamId: TypeAlias = UUID
SubId: TypeAlias = UUID


@runtime_checkable
class Stream(Protocol):
    id: StreamId

    pending_messages: list[ModelRequest]
    """Inbox of follow-up turns produced asynchronously (e.g. by background
    tasks). The agent loop drains this before each model call; whatever lands
    here while no ``ask`` is running is consumed by the next ``ask`` on this
    stream and merged into its initial request."""

    async def send(self, event: BaseEvent, context: "ConversationContext") -> None: ...

    def enqueue(self, *content: "SendableMessage | Input") -> None:
        """Append a follow-up turn to this stream's inbox.

        Low-level: it only appends and announces nothing. Code that hands a
        running agent input goes through ``ConversationContext.enqueue``,
        which also publishes ``MessageEnqueued``.
        """
        ...

    def spawn_background(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        """Start a fire-and-forget task in this stream's scope.

        The task is not awaited by the agent loop. Tasks deliver their results
        via ``self.enqueue(...)`` — anything enqueued while an ``ask`` is live
        feeds the next model call; anything enqueued after ``ask`` returned
        sits in ``pending_messages`` and is consumed by the next ``ask`` on
        the same stream.
        """
        ...

    def where(self, condition: ClassInfo | Condition) -> "Stream": ...

    def join(
        self,
        *,
        max_events: int | None = None,
    ) -> AbstractContextManager[AsyncIterator[BaseEvent]]: ...

    @overload
    def subscribe(
        self,
        func: Callable[..., Any],
        *,
        interrupt: bool = False,
        sync_to_thread: bool = True,
        condition: Condition | None = None,
    ) -> SubId: ...

    @overload
    def subscribe(
        self,
        func: None = None,
        *,
        interrupt: bool = False,
        sync_to_thread: bool = True,
        condition: Condition | None = None,
    ) -> Callable[[Callable[..., Any]], SubId]: ...

    def subscribe(
        self,
        func: Callable[..., Any] | None = None,
        *,
        interrupt: bool = False,
        sync_to_thread: bool = True,
        condition: Condition | None = None,
    ) -> Callable[[Callable[..., Any]], SubId] | SubId: ...

    def unsubscribe(self, sub_id: SubId) -> None: ...

    def sub_scope(
        self,
        func: Callable[..., Any],
        *,
        interrupt: bool = False,
        sync_to_thread: bool = True,
    ) -> AbstractContextManager[None]: ...

    def get(
        self,
        condition: ClassInfo | Condition,
    ) -> AbstractAsyncContextManager[asyncio.Future[BaseEvent]]: ...


@dataclass(slots=True)
class ConversationContext:
    stream: Stream = field(repr=False)
    dependency_provider: "Provider | None" = field(default=None, repr=False)

    # store Context Variables as separated serializable field. Keys under
    # ``RESERVED_VARIABLE_PREFIXES`` are the framework's own control-plane state
    # and are never authored by a remote peer — see ``strip_reserved_variables``.
    variables: dict[str, Any] = field(default_factory=dict)

    dependencies: dict[Any, Any] = field(default_factory=dict)

    prompt: list[str] = field(default_factory=list)

    @property
    def pending_messages(self) -> list[ModelRequest]:
        """Read-through view of the underlying stream's inbox."""
        return self.stream.pending_messages

    def spawn_background(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        """Forward to ``self.stream.spawn_background``.

        Background tasks live in the stream's scope (not the per-run Context),
        so a task that finishes after this ``ask`` returns still delivers its
        result — the next ``ask`` on the same stream picks it up.
        """
        return self.stream.spawn_background(coro)

    def enqueue(self, *content: "SendableMessage | Input") -> None:
        """Append a follow-up turn to the stream's inbox and announce it.

        The inbox lives on the stream, so a message enqueued here survives the
        end of the current run and feeds the next ``ask`` on the same stream.
        The append is immediate; ``MessageEnqueued`` follows from a background
        task, so this stays safe to call from a stream subscriber and from a
        sync tool running in a worker thread. From a thread that is not an
        event-loop worker the message is appended but not announced.
        """
        if not content:
            return
        self.stream.enqueue(*content)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # A sync tool runs in an anyio worker thread; the announcement has to
            # be scheduled on the event loop that owns the stream.
            with suppress(RuntimeError):
                anyio.from_thread.run_sync(self._announce_enqueued)
        else:
            self._announce_enqueued()

    def _announce_enqueued(self) -> None:
        self.spawn_background(self.send(MessageEnqueued()))

    async def input(self, message: str, timeout: float | None = None) -> str:
        """Put ``message`` to a human and return their answer.

        Every way this can fail to produce an answer leaves as a
        :class:`~ag2.exceptions.HumanInputError`, and this is the only place
        that decides so: the hook raising, nobody answering within *timeout*,
        or the send itself breaking. Normalising here rather than at each
        catch site downstream is what keeps the distinction from evaporating —
        the alternative, tagging whatever the hook threw, is lost the moment
        anything in between re-raises it as its own error.

        *timeout* covers whoever is actually answering, which is not always
        someone in this process. When the agent is served over a protocol that
        routes the question to the calling client — ``ag2.mcp`` does, and on
        the 2026-07-28 revision the question goes back as the result of the
        call — the wait spans the client's side of a round trip: the network,
        and a human reading the question in another application. Size it for
        that, not for a local prompt. It stays independent of any lifetime the
        transport puts on the pause; whichever elapses first ends the turn.
        """
        request_msg = HumanInputRequest(message)
        async with self.stream.get(HumanMessage.parent_id == request_msg.id) as response:
            try:
                # The hook runs inline inside ``send``, so *timeout* has to
                # cover the asking as well as the waiting. Timing only
                # ``response`` starts the clock after the hook has already
                # returned, which makes the timeout unreachable and leaves a
                # hook that hangs hanging the turn forever.
                result = await asyncio.wait_for(_ask_human(self, request_msg, response), timeout)

            except HumanInputError:
                raise  # classified already, by _ask_human or by the hook itself

            except asyncio.TimeoutError as exc:
                # Only ``wait_for`` can reach this: anything the channel raised,
                # timeouts included, left _ask_human as a HumanInputError.
                raise HumanInputTimeoutError(timeout) from exc  # type: ignore[arg-type]

        return result.content

    async def send(self, event: BaseEvent) -> None:
        await self.stream.send(event, self)


async def _ask_human(
    context: "ConversationContext",
    request: HumanInputRequest,
    response: "asyncio.Future[BaseEvent]",
) -> HumanMessage:
    """Ask, then wait — as one awaitable, so a timeout can cover both.

    Module level rather than nested in ``Context.input`` (per AGENTS.md: no
    nested functions in runtime execution paths).
    """
    try:
        await context.send(request)

    except HumanInputError:
        raise  # already says the channel failed, and says it precisely

    # Classifying the hook's exceptions here, inside the awaited coroutine,
    # rather than around ``wait_for`` is what keeps a hook that raises
    # ``TimeoutError`` of its own from being reported as nobody answering: after
    # this, the only bare timeout the caller can see is ``wait_for``'s.
    except Exception as exc:
        raise HumanInputFailedError(exc) from exc

    # The future is filtered on ``HumanMessage.parent_id``, so this is one.
    return cast(HumanMessage, await response)


def drop_background_task(tasks: set[asyncio.Task[None]], task: asyncio.Task[Any]) -> None:
    """Done-callback for ``Stream.spawn_background``: remove the task from the
    live set and surface any exception to the log (so asyncio doesn't warn
    about an unretrieved exception when the task is GC'd).
    """
    tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.exception("Background task raised", exc_info=exc)


# Variable namespaces the framework reserves for its own control-plane state:
# the ``approval_required`` allow-always bypass (``ag:approval_required:always``),
# the A2A context-id bookkeeping and the per-call tenant override
# (``a2a:tenant``) all live in ``ConversationContext.variables``. A transport
# that syncs variables with a peer must not let that peer author them — a caller
# able to write the bypass key pre-approves a gated tool and the human is never
# asked. Every wire-originated merge goes through ``strip_reserved_variables``.
RESERVED_VARIABLE_PREFIXES = ("ag:", "a2a:")


def strip_reserved_variables(payload: Mapping[str, Any], *, source: str, warn: bool = True) -> dict[str, Any]:
    """Return *payload* without the framework's reserved variable keys.

    Transports call this on every variables payload that arrives from — or
    leaves for — a remote peer, so control-plane state stays locally authored;
    ``warn`` is off on the outbound side, where stripping is routine rather
    than a peer overstepping.
    """
    if not payload:
        return {}
    kept = {key: value for key, value in payload.items() if not str(key).startswith(RESERVED_VARIABLE_PREFIXES)}
    if warn and len(kept) != len(payload):
        logger.warning(
            "Dropped reserved context variables from %s: %s",
            source,
            sorted(str(key) for key in payload if key not in kept),
        )
    return kept
