# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, cast, overload, runtime_checkable
from uuid import UUID

from fast_depends import Provider

from ag2.types import ClassInfo, SendableMessage

from .events import BaseEvent, HumanInputRequest, HumanMessage, Input, ModelRequest
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
        """Append a follow-up turn to this stream's inbox."""
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

    # store Context Variables as separated serializable field
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
        """Forward to ``self.stream.enqueue``.

        The inbox lives on the stream, so a message enqueued here survives the
        end of the current run and feeds the next ``ask`` on the same stream.
        """
        self.stream.enqueue(*content)

    async def input(self, message: str, timeout: float | None = None) -> str:
        """Put ``message`` to a human and return their answer.

        Every way this can fail to produce an answer leaves as a
        :class:`~ag2.exceptions.HumanInputError`, and this is the only place
        that decides so: the hook raising, nobody answering within *timeout*,
        or the send itself breaking. Normalising here rather than at each
        catch site downstream is what keeps the distinction from evaporating —
        the alternative, tagging whatever the hook threw, is lost the moment
        anything in between re-raises it as its own error.
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
