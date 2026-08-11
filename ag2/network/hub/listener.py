# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``HubListener`` — read-only Protocol for hub state-transition notifications.

A listener attaches via :meth:`Hub.register_listener`. The hub fires
listener methods after the corresponding state change has committed —
they are observers, not gatekeepers. Decision-making lives in
:class:`HubArbiter` (see ``arbiter.py``).

Every method has a default ``pass`` body so implementations only override
what they care about. The hub wraps each listener call in
``try/except``; a buggy listener cannot break dispatch.

Conventions:

* All methods are ``async``. Hub awaits them sequentially in
  registration order. Listeners that need to do I/O should keep it
  short (or schedule work onto their own queue) — slow listeners stall
  the dispatch path.
* Method names use the past tense (``on_envelope_posted``,
  ``on_channel_closed``) because they fire after the fact.
* Exceptions raised inside a listener are logged at ``ERROR`` and
  swallowed.
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..channel import ChannelMetadata, Expectation
    from ..envelope import Envelope
    from ..errors import NetworkError
    from .expectations import Violation

__all__ = ("HubListener",)


@runtime_checkable
class HubListener(Protocol):
    """Observer Protocol for hub state transitions.

    Register via :meth:`Hub.register_listener`. Implementations override
    only the methods they care about; defaults are no-ops.
    """

    async def on_envelope_posted(
        self,
        envelope: "Envelope",
        metadata: "ChannelMetadata",
    ) -> None:
        """An envelope was validated, WAL-appended, folded, and dispatched."""

    async def on_envelope_rejected(
        self,
        envelope: "Envelope",
        reason: "NetworkError",
    ) -> None:
        """An envelope was rejected before WAL append.

        ``reason`` is the typed error (``AccessDeniedError``,
        ``ProtocolError``, ``InboxFull``, ``RateLimited``, …) the
        sender saw. Use this for tenant-side metrics / alerting on
        rejection rates.
        """

    async def on_dispatch_failed(
        self,
        envelope: "Envelope",
        recipient_id: str,
        reason: BaseException,
    ) -> None:
        """Dispatch of an accepted envelope to a specific recipient failed.

        Per-recipient — the rest of the audience may have received
        normally. Causes typically reflect a closed endpoint or a
        downstream link error. Sender still sees the
        ``post_envelope`` return as success because the WAL committed.
        """

    async def on_channel_event(
        self,
        channel_id: str,
        kind: str,
        payload: dict,
    ) -> None:
        """A channel-lifecycle event fired.

        ``kind`` is one of ``"opened"``, ``"closed"``, ``"expired"``,
        ``"participant_removed"``, ``"participant_hidden"``. ``payload``
        carries event-specific fields (close ``reason``, expired
        ``at``, etc.).
        """

    async def on_agent_event(
        self,
        agent_id: str,
        kind: str,
        payload: dict,
    ) -> None:
        """An identity-lifecycle event fired.

        ``kind`` is one of ``"registered"``, ``"unregistered"``,
        ``"resume_set"``, ``"skill_set"``, ``"rule_set"``,
        ``"observation_recorded"``. ``payload`` carries
        kind-specific fields (e.g. ``{"passport": Passport}`` for
        ``"registered"``).
        """

    async def on_expectation_fired(
        self,
        channel_id: str,
        expectation: "Expectation",
        violation: "Violation",
    ) -> None:
        """An expectation evaluator emitted a violation.

        Fires once per ``(channel, expectation, violator)`` per
        evaluator tick (the hub dedupes repeat fires of the same
        violation key).
        """

    async def on_turn_failed(
        self,
        channel_id: str,
        agent_id: str,
        envelope_id: str,
        exc: BaseException,
    ) -> None:
        """A notify handler raised while processing an inbound envelope.

        The default notify handler traps ``agent.ask`` and
        ``build_round_envelope`` exceptions and emits this event. The
        channel stays alive; no reply envelope is posted. The
        application chooses how to react (retry, escalate, surface to
        a UI).
        """

    async def on_task_event(
        self,
        task_id: str,
        kind: str,
        payload: dict,
    ) -> None:
        """A task-lifecycle event fired.

        ``kind`` is one of ``"started"``, ``"progress"``,
        ``"completed"``, ``"failed"``, ``"expired"``, ``"cancelled"``,
        ``"mirror_failed"``. The mirror-failed kind signals that
        ``TaskMirror`` could not forward an observation to the hub —
        the task itself may still be advancing locally.
        """

    async def on_inbox_pressure(
        self,
        agent_id: str,
        pending: int,
        cap: int,
    ) -> None:
        """A recipient's inbox crossed the high-water mark.

        Fired when ``pending`` first crosses
        ``Rule.limits.inbox.high_water`` (default: 80% of
        ``max_pending``). Fires at most once per crossing — does not
        re-fire on every subsequent envelope while above the mark.
        Operators wire this to a backpressure dashboard / alert.
        """

    # Default implementations: every method is no-op so subclasses /
    # Protocol implementers can override only the events they care
    # about. The hub's fan-out loop swallows exceptions either way.

    # Implemented as method bodies below so a concrete class that does
    # ``class MyListener(HubListener): pass`` doesn't error at the
    # first hub fan-out.


# Concrete default impl. Protocols are structural so a subclass is not
# required, but exposing a base lets users do `class Foo(BaseHubListener)`
# and override only what they care about without declaring the full
# Protocol surface.
class BaseHubListener:
    """No-op base implementation. Override only the events you care about."""

    async def on_envelope_posted(self, envelope, metadata) -> None:  # noqa: ARG002
        return None

    async def on_envelope_rejected(self, envelope, reason) -> None:  # noqa: ARG002
        return None

    async def on_dispatch_failed(self, envelope, recipient_id, reason) -> None:  # noqa: ARG002
        return None

    async def on_channel_event(self, channel_id, kind, payload) -> None:  # noqa: ARG002
        return None

    async def on_agent_event(self, agent_id, kind, payload) -> None:  # noqa: ARG002
        return None

    async def on_expectation_fired(self, channel_id, expectation, violation) -> None:  # noqa: ARG002
        return None

    async def on_turn_failed(self, channel_id, agent_id, envelope_id, exc) -> None:  # noqa: ARG002
        return None

    async def on_task_event(self, task_id, kind, payload) -> None:  # noqa: ARG002
        return None

    async def on_inbox_pressure(self, agent_id, pending, cap) -> None:  # noqa: ARG002
        return None


__all__ = ("BaseHubListener", "HubListener")
