# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Observer the eval runner attaches inside the agent loop.

The eval framework's foundational claim is that the agent's event
stream — ``ModelRequest`` / ``ModelResponse`` / ``ToolCallEvent`` /
``ToolResultEvent`` / ``HaltEvent`` / ``TaskStarted`` / … — is the
lowest-level, most complete substrate for evaluation. A :class:`Trace`
is just the events that landed on the stream during one task. Scorers
grade what's in the trace.

:class:`EventCapture` is how the runner consumes that substrate. It
satisfies the framework's :class:`~ag2.observers.Observer`
protocol — the same extension point ``LoopDetector`` and ``TokenMonitor``
use — so it composes with any user-supplied observers without conflict.

The subscription is *direct* (``stream.subscribe``), not stack-scoped
(``stream.sub_scope``). That distinction is load-bearing: it means the
subscription persists across *multiple* ``Agent._execute`` calls on
the same stream, so a continuation pattern like ``reply.ask(...)``
(which re-enters ``_execute`` with a fresh ``ExitStack`` the runner
didn't pre-attach to) is still captured. The stream is per-task and
discarded by the runner after the task ends, so the subscription dies
with the stream — no explicit unsubscribe needed.

Per-task wall-clock duration is measured in :func:`run_agent` directly with
``time.perf_counter`` — that captures the whole agent-side workload
including any internal ``reply.ask`` continuations, which a per-turn
middleware can't see.

The same stream-subscription pattern is what makes the v0 → v1 online-
evaluation path natural: replace "subscribe to a per-task in-memory
stream" with "subscribe to a live production stream (sampled)" and the
scorers don't change. The boundary between offline and online is just
where the stream came from.
"""

from contextlib import ExitStack

from ag2.annotations import Context
from ag2.events import BaseEvent

__all__ = ("EventCapture",)


class EventCapture:
    """Observer that records every event emitted during one task.

    Subscribes once to the agent's stream on first ``register`` and
    accumulates events into :attr:`events` for the runner to read after
    ``agent.ask(...)`` returns. Idempotent: if ``register`` is called
    again (re-entering ``_execute`` for a continuation turn), the
    existing subscription is left in place rather than duplicated.
    """

    __slots__ = ("events", "_subscribed")

    def __init__(self) -> None:
        self.events: list[BaseEvent] = []
        self._subscribed = False

    def register(self, stack: ExitStack, context: Context) -> None:
        # Subscribe directly (not via sub_scope) so the subscription
        # outlives this ExitStack and continues capturing events from
        # any follow-up ``reply.ask`` on the same stream. The unused
        # ``stack`` parameter is part of the Observer protocol contract.
        del stack
        if self._subscribed:
            return
        context.stream.subscribe(self._on_event, sync_to_thread=False)
        self._subscribed = True

    async def _on_event(self, event: BaseEvent) -> None:
        self.events.append(event)
