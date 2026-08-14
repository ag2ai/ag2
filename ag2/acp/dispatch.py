# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Make inbound ACP notifications handled in wire order, and waitable.

The SDK's receive loop is not order-preserving across message kinds. A response
is correlated inline, while a notification is published to a queue whose
consumer dispatches it as a fire-and-forget task and marks the queue entry done
the moment that task is *created*. Two consequences, both visible to a caller:
``session/prompt`` can return while ``session/update``s that preceded it on the
wire are still unhandled, so the turn read out at that moment is short by
whatever has not landed; and concurrent handler tasks can interleave, so chunks
can be handled out of the order the agent sent them.

``Connection`` takes both the queue and the dispatcher as arguments, which is the
seam used here: :class:`_InOrderDispatcher` *awaits* each notification instead of
spawning it, so the dispatcher's own sequential loop becomes the ordering, and
the queue entry is marked done only once the handler has returned. That is what
makes :meth:`InboundUpdates.settle` exact — ``join()`` returns when every
notification read off the wire has been handled, with nothing to poll or time.

The trade-off: an inbound *request* published behind a notification now waits for
that notification's handler. For AG2 that means a ``session/request_permission``
arriving mid-stream is delayed by one ``handle_update``, which sends an event to
the run's stream — bounded by whatever the caller's subscribers do. Requests
themselves still run concurrently once dispatched, so a permission prompt
awaiting a human never blocks the updates behind it.
"""

from typing import Any

from acp.task import (
    DefaultMessageDispatcher,
    InMemoryMessageQueue,
    MessageDispatcher,
    MessageQueue,
    MessageStateStore,
    NotificationRunner,
    RequestRunner,
    TaskSupervisor,
)

__all__ = ["InboundUpdates"]


class _InOrderDispatcher(DefaultMessageDispatcher):
    """``DefaultMessageDispatcher``, but notifications are awaited, not spawned."""

    def __init__(self, *, notification_runner: NotificationRunner, **kwargs: Any) -> None:
        super().__init__(notification_runner=notification_runner, **kwargs)
        # Held separately rather than reaching for the base class's own copy.
        self._run_notification = notification_runner

    async def _dispatch_notification(self, message: dict[str, Any]) -> None:
        # The base class's loop calls ``task_done()`` once this returns, so
        # awaiting here is also what makes "done" mean "handled".
        await self._run_notification(message)


class InboundUpdates:
    """Per-connection notification plumbing, and the wait a turn does on it.

    One of these is owned by the bridge (:class:`~.bridge.BridgeState`), whose
    connection hooks splat :attr:`connection_kwargs` into the SDK call that builds
    the connection. A bridge with no connection yet — or a test double that never
    builds a real one — simply never has anything to settle.
    """

    def __init__(self) -> None:
        self._queue: MessageQueue | None = None

    def connection_kwargs(self) -> dict[str, Any]:
        """What to pass to ``spawn_agent_process`` / ``connect_to_agent``.

        The queue is minted here rather than left to ``Connection``, because it is
        what :meth:`settle` waits on and so this side needs the reference. One
        fresh queue per call: a queue is closed along with the connection that
        drained it, so a second connection cannot be handed the first one's.
        """
        self._queue = InMemoryMessageQueue()
        return {"queue": self._queue, "dispatcher_factory": self._make_dispatcher}

    def _make_dispatcher(
        self,
        queue: MessageQueue,
        supervisor: TaskSupervisor,
        store: MessageStateStore,
        request_runner: RequestRunner,
        notification_runner: NotificationRunner,
    ) -> MessageDispatcher:
        return _InOrderDispatcher(
            queue=queue,
            supervisor=supervisor,
            store=store,
            request_runner=request_runner,
            notification_runner=notification_runner,
        )

    async def settle(self) -> None:
        """Wait until every notification read off the wire has been handled.

        Every notification the agent sent before its ``session/prompt`` response
        was published to the queue before that response was read, so a caller
        that reaches here after the response awaits exactly this turn's updates.
        A bridge whose connection never went through here — an in-process test
        double calls ``session_update`` directly — has nothing to wait for.
        """
        if self._queue is not None:
            await self._queue.join()
